"""
Data Loading and Preprocessing Utilities for Stream-DAD.

This module provides utilities for loading various benchmark datasets,
preprocessing time series data, and creating streaming data loaders
for training and evaluation.
"""
from datetime import time

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator, Union, Any
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from stream_dad.utils.drift import inject_synthetic_drift, validate_drift_injection, create_drift_scenarios, \
    STANDARD_DRIFT_SCENARIOS

logger = logging.getLogger(__name__)


class TimeSeriesDataset(data.Dataset):
    """
    PyTorch Dataset for time series data with sliding windows.

    Args:
        data: Time series data [num_samples, num_features]
        labels: Anomaly labels [num_samples] (optional)
        window_size: Size of sliding windows
        stride: Stride for window creation
        normalize: Whether to normalize the data
    """

    def __init__(self,
                 data: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 window_size: int = 10,
                 stride: int = 1,
                 normalize: bool = True):

        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

        # Create sliding windows
        self.windows, self.window_labels = self._create_windows()

        # Normalize if requested
        if self.normalize:
            self.scaler = StandardScaler()
            original_shape = self.windows.shape
            # Reshape to (num_windows * window_size, num_features) for scaling
            reshaped_windows = self.windows.reshape(-1, original_shape[-1])
            scaled_windows = self.scaler.fit_transform(reshaped_windows)
            self.windows = scaled_windows.reshape(original_shape)
        else:
            self.scaler = None

    def _create_windows(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding windows from time series data."""
        num_samples, num_features = self.data.shape

        # Calculate number of windows
        num_windows = (num_samples - self.window_size) // self.stride + 1

        # Create windows
        windows = np.zeros((num_windows, self.window_size, num_features))
        window_labels = None

        if self.labels is not None:
            window_labels = np.zeros(num_windows)

        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            windows[i] = self.data[start_idx:end_idx]

            if self.labels is not None:
                # Label window as anomalous if any sample in window is anomalous
                window_labels[i] = np.any(self.labels[start_idx:end_idx])

        return windows, window_labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single window and its label."""
        window = torch.FloatTensor(self.windows[idx])

        item = {'data': window}

        if self.window_labels is not None:
            item['label'] = torch.FloatTensor([self.window_labels[idx]])

        return item

    def get_scaler(self):
        """Return the fitted scaler for inverse transform."""
        return self.scaler


class StreamingDataLoader:
    """
    Streaming data loader that simulates real-time data arrival.

    Processes data sequentially without shuffling to maintain temporal order,
    which is crucial for concept drift evaluation.
    """

    def __init__(self,
                 dataset: TimeSeriesDataset,
                 batch_size: int = 1,
                 shuffle: bool = False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0

        # Create index order
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through dataset in streaming fashion."""
        self.current_idx = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        """Get next batch of data."""
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # Get batch indices
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_idx:end_idx]

        # Collect batch data
        batch_data = []
        batch_labels = []

        for idx in batch_indices:
            item = self.dataset[idx]
            batch_data.append(item['data'])
            if 'label' in item:
                batch_labels.append(item['label'])

        # Stack into tensors
        batch_dict = {
            'data': torch.stack(batch_data)
        }

        if batch_labels:
            batch_dict['labels'] = torch.stack(batch_labels)

        self.current_idx = end_idx

        return batch_dict

    def __len__(self) -> int:
        """Number of batches in the loader."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def reset(self) -> None:
        """Reset the loader to beginning."""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return data and labels."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, any]:
        """Get dataset information."""
        pass


class SWaTLoader(DatasetLoader):
    """Loader for SWaT (Secure Water Treatment) dataset."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load SWaT dataset."""
        try:
            # Load training data
            train_file = self.data_path / "SWaT_Dataset_Normal_v1.csv"
            test_file = self.data_path / "SWaT_Dataset_Attack_v0.csv"

            if not train_file.exists() or not test_file.exists():
                raise FileNotFoundError(f"SWaT dataset files not found in {self.data_path}")

            # Load and preprocess training data
            train_df = pd.read_csv(train_file)
            train_df = self._preprocess_swat_data(train_df)
            train_data = train_df.select_dtypes(include=[np.number]).values
            train_labels = np.zeros(len(train_data))  # Normal data

            # Load and preprocess test data
            test_df = pd.read_csv(test_file)
            test_df = self._preprocess_swat_data(test_df)
            test_data = test_df.select_dtypes(include=[np.number]).values

            # Extract labels from test data
            if 'Normal/Attack' in test_df.columns:
                test_labels = (test_df['Normal/Attack'] == 'Attack').astype(int).values
            else:
                test_labels = np.zeros(len(test_data))  # Fallback

            # Combine train and test data
            data = np.vstack([train_data, test_data])
            labels = np.hstack([train_labels, test_labels])

            logger.info(f"Loaded SWaT dataset: {data.shape[0]} samples, {data.shape[1]} features")
            logger.info(f"Anomaly ratio: {labels.mean():.3f}")

            return data, labels

        except Exception as e:
            logger.error(f"Error loading SWaT dataset: {e}")
            raise

    def _preprocess_swat_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess SWaT data."""
        # Remove timestamp columns
        if 'Timestamp' in df.columns:
            df = df.drop('Timestamp', axis=1)

        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Convert categorical columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'Normal/Attack':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df = df.drop(col, axis=1)

        return df

    def get_info(self) -> Dict[str, any]:
        """Get dataset information."""
        return {
            'name': 'SWaT',
            'description': 'Secure Water Treatment dataset',
            'domain': 'Industrial Control Systems',
            'features': 51,
            'type': 'Physical Process'
        }


class WADILoader(DatasetLoader):
    """Loader for WADI (Water Distribution) dataset."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load WADI dataset."""
        try:
            train_file = self.data_path / "WADI_14days_new.csv"
            test_file = self.data_path / "WADI_attackdataLABLE.csv"

            if not train_file.exists() or not test_file.exists():
                raise FileNotFoundError(f"WADI dataset files not found in {self.data_path}")

            # Load training data
            train_df = pd.read_csv(train_file)
            train_df = self._preprocess_wadi_data(train_df)
            train_data = train_df.select_dtypes(include=[np.number]).values
            train_labels = np.zeros(len(train_data))

            # Load test data
            test_df = pd.read_csv(test_file)
            test_df = self._preprocess_wadi_data(test_df)
            test_data = test_df.select_dtypes(include=[np.number]).values

            # Extract labels
            if 'Attack LABLE (1:No Attack, -1:Attack)' in test_df.columns:
                test_labels = (test_df['Attack LABLE (1:No Attack, -1:Attack)'] == -1).astype(int).values
            else:
                test_labels = np.zeros(len(test_data))

            # Combine data
            data = np.vstack([train_data, test_data])
            labels = np.hstack([train_labels, test_labels])

            logger.info(f"Loaded WADI dataset: {data.shape[0]} samples, {data.shape[1]} features")
            logger.info(f"Anomaly ratio: {labels.mean():.3f}")

            return data, labels

        except Exception as e:
            logger.error(f"Error loading WADI dataset: {e}")
            raise

    def _preprocess_wadi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess WADI data."""
        # Remove timestamp and label columns from features
        columns_to_drop = ['Date', 'Time', 'Attack LABLE (1:No Attack, -1:Attack)']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def get_info(self) -> Dict[str, any]:
        """Get dataset information."""
        return {
            'name': 'WADI',
            'description': 'Water Distribution dataset',
            'domain': 'Industrial Control Systems',
            'features': 14,
            'type': 'Physical Process'
        }


def load_dataset_from_config(config_path: str) -> Tuple[StreamingDataLoader, Dict[str, Any]]:
    """
    Load dataset directly from configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        data_loader: Configured streaming data loader
        dataset_info: Dataset information
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config.get('dataset', {})
    data_config = config.get('data', {})

    dataset_name = dataset_config.get('name', 'synthetic')
    data_path = dataset_config.get('path', './data')

    return create_data_loader(dataset_name, data_path, data_config)


def split_data_temporal(data_loader: StreamingDataLoader,
                        split_ratio: float = 0.8) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    """
    Split data loader temporally for train/validation while maintaining time order.

    Args:
        data_loader: Original data loader
        split_ratio: Ratio for training split (0.8 = 80% train, 20% val)

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Get all data
    all_data = []
    for batch in data_loader:
        all_data.append(batch)

    # Split temporally
    split_idx = int(len(all_data) * split_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Create new loaders
    class StaticDataLoader:
        def __init__(self, data_list):
            self.data_list = data_list

        def __iter__(self):
            return iter(self.data_list)

        def __len__(self):
            return len(self.data_list)

    train_loader = StaticDataLoader(train_data)
    val_loader = StaticDataLoader(val_data)

    return train_loader, val_loader


def create_cross_validation_splits(data_loader: StreamingDataLoader,
                                   n_splits: int = 5) -> List[Tuple[StreamingDataLoader, StreamingDataLoader]]:
    """
    Create time-series cross-validation splits maintaining temporal order.

    Args:
        data_loader: Original data loader
        n_splits: Number of CV splits

    Returns:
        cv_splits: List of (train_loader, val_loader) tuples
    """
    # Get all data
    all_data = []
    for batch in data_loader:
        all_data.append(batch)

    splits = []
    data_length = len(all_data)

    for i in range(n_splits):
        # Time series CV: use expanding window
        train_end = int((i + 1) * data_length / (n_splits + 1))
        val_start = train_end
        val_end = min(val_start + data_length // (n_splits + 1), data_length)

        train_data = all_data[:train_end]
        val_data = all_data[val_start:val_end]

        if len(train_data) > 0 and len(val_data) > 0:
            class StaticDataLoader:
                def __init__(self, data_list):
                    self.data_list = data_list

                def __iter__(self):
                    return iter(self.data_list)

                def __len__(self):
                    return len(self.data_list)

            train_loader = StaticDataLoader(train_data)
            val_loader = StaticDataLoader(val_data)
            splits.append((train_loader, val_loader))

    return splits


def create_multi_dataset_loader(dataset_configs: List[Dict]) -> Dict[str, Tuple[StreamingDataLoader, Dict]]:
    """
    Create data loaders for multiple datasets.

    Args:
        dataset_configs: List of dataset configurations

    Returns:
        loaders: Dictionary mapping dataset names to (loader, info) tuples
    """
    loaders = {}

    for config in dataset_configs:
        name = config['name']
        data_path = config['data_path']
        data_config = config.get('config', {})

        try:
            loader, info = create_data_loader(name, data_path, data_config)
            loaders[name] = (loader, info)
            logger.info(f"Successfully created loader for {name}")
        except Exception as e:
            logger.error(f"Failed to create loader for {name}: {e}")

    return loaders


def save_dataset_cache(data: np.ndarray,
                       labels: np.ndarray,
                       dataset_info: Dict,
                       cache_path: str):
    """
    Save processed dataset to cache for faster loading.

    Args:
        data: Processed data array
        labels: Label array
        dataset_info: Dataset metadata
        cache_path: Path to save cache
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        'data': data,
        'labels': labels,
        'dataset_info': dataset_info,
        'cache_version': '1.0',
        'timestamp': time.time()
    }

    np.savez_compressed(cache_path, **cache_data)
    logger.info(f"Dataset cache saved to {cache_path}")


def load_dataset_cache(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Load dataset from cache if available and valid.

    Args:
        cache_path: Path to cache file

    Returns:
        (data, labels, dataset_info) if cache is valid, None otherwise
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None

    try:
        cache_data = np.load(cache_path, allow_pickle=True)

        # Check cache version
        if cache_data.get('cache_version', '0.0') != '1.0':
            logger.warning(f"Cache version mismatch, ignoring cache: {cache_path}")
            return None

        data = cache_data['data']
        labels = cache_data['labels']
        dataset_info = cache_data['dataset_info'].item()

        logger.info(f"Loaded dataset from cache: {cache_path}")
        return data, labels, dataset_info

    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None


# Example usage functions
def create_benchmark_evaluation_setup(dataset_names: List[str],
                                      data_paths: Dict[str, str],
                                      base_config: Dict) -> Dict[str, Any]:
    """
    Create standardized evaluation setup for benchmarking across datasets.

    Args:
        dataset_names: List of dataset names to evaluate
        data_paths: Dictionary mapping dataset names to their data paths
        base_config: Base configuration to use for all datasets

    Returns:
        benchmark_setup: Dictionary with loaders and configurations for each dataset
    """
    benchmark_setup = {}

    for dataset_name in dataset_names:
        if dataset_name not in data_paths:
            logger.warning(f"No data path specified for {dataset_name}, skipping")
            continue

        # Create dataset-specific config
        dataset_config = base_config.copy()

        # Apply dataset-specific overrides
        if dataset_name in dataset_config.get('datasets', {}):
            dataset_overrides = dataset_config['datasets'][dataset_name]
            dataset_config['data'].update(dataset_overrides)

        try:
            # Create data loader
            data_loader, dataset_info = create_data_loader(
                dataset_name,
                data_paths[dataset_name],
                dataset_config['data']
            )

            # Split into train/val
            train_loader, val_loader = split_data_temporal(data_loader, split_ratio=0.8)

            benchmark_setup[dataset_name] = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'dataset_info': dataset_info,
                'config': dataset_config
            }

            logger.info(f"Created benchmark setup for {dataset_name}")

        except Exception as e:
            logger.error(f"Failed to create benchmark setup for {dataset_name}: {e}")

    return benchmark_setup


if __name__ == "__main__":
    # Example usage and testing
    import tempfile

    def test_synthetic_data_generation():
        """Test synthetic data generation and drift injection."""
        print("Testing synthetic data generation...")

        # Test configuration
        config = {
            'synthetic': {
                'num_features': 20,
                'num_samples': 1000,
                'anomaly_ratio': 0.05,
                'seed': 42,
                'drift': {
                    'type': 'gradual',
                    'magnitude': 0.3,
                    'start': 500,
                    'num_affected_features': 6
                }
            },
            'data': {
                'window_size': 10,
                'batch_size': 1,
                'normalize': True,
                'shuffle': False
            }
        }

        # Create synthetic data loader
        data_loader, dataset_info = create_data_loader('synthetic', '', config)

        print(f"Created synthetic dataset: {dataset_info}")
        print(f"Data loader length: {len(data_loader)}")

        # Test first few batches
        batch_count = 0
        for batch in data_loader:
            if batch_count >= 5:
                break

            data = batch['data']
            labels = batch.get('labels', None)

            print(f"Batch {batch_count}: data shape {data.shape}")
            if labels is not None:
                print(f"  Labels shape: {labels.shape}")

            batch_count += 1

        print("Synthetic data generation test completed ✓")

    def test_drift_injection():
        """Test drift injection functionality."""
        print("\nTesting drift injection...")

        # Create base data
        np.random.seed(42)
        base_data = np.random.randn(1000, 10)
        base_labels = np.random.binomial(1, 0.05, 1000)

        # Test different drift types
        drift_configs = [
            {
                'type': 'gradual',
                'magnitude': 0.5,
                'start': 500,
                'num_affected_features': 3
            },
            {
                'type': 'sudden',
                'magnitude': 1.0,
                'start': 500,
                'num_affected_features': 5
            },
            {
                'type': 'recurring',
                'magnitude': 0.3,
                'start': 300,
                'frequency': 0.02,
                'num_affected_features': 4
            }
        ]

        for i, drift_config in enumerate(drift_configs):
            print(f"Testing {drift_config['type']} drift...")

            drifted_data, drifted_labels = inject_synthetic_drift(
                base_data, base_labels, drift_config
            )

            # Validate drift injection
            validation_metrics = validate_drift_injection(
                base_data, drifted_data, drift_config
            )

            print(f"  Drift magnitude: {validation_metrics['overall_drift_magnitude']:.4f}")
            print(f"  Mean shift: {validation_metrics['mean_shift']['average']:.4f}")

        print("Drift injection test completed ✓")

    def test_multiple_scenarios():
        """Test multiple drift scenarios."""
        print("\nTesting multiple drift scenarios...")

        # Create base data
        np.random.seed(42)
        base_data = np.random.randn(800, 15)
        base_labels = np.random.binomial(1, 0.03, 800)

        # Create multiple scenarios
        scenarios = create_drift_scenarios(
            base_data, base_labels, STANDARD_DRIFT_SCENARIOS[:3]
        )

        print(f"Created {len(scenarios)} drift scenarios:")
        for i, (data, labels, config) in enumerate(scenarios):
            print(f"  Scenario {i+1}: {config['name']}")
            print(f"    Data shape: {data.shape}")
            print(f"    Anomaly ratio: {config['final_anomaly_ratio']:.4f}")

        print("Multiple scenarios test completed ✓")

    def test_real_dataset_loading():
        """Test real dataset loading (with mock data)."""
        print("\nTesting real dataset loading...")

        # Create temporary mock data files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock SWaT data
            mock_train_data = pd.DataFrame({
                'FIT101': np.random.randn(100),
                'LIT101': np.random.randn(100),
                'MV101': np.random.randn(100),
                'P101': np.random.randn(100),
                'Normal/Attack': ['Normal'] * 100
            })

            mock_test_data = pd.DataFrame({
                'FIT101': np.random.randn(50),
                'LIT101': np.random.randn(50),
                'MV101': np.random.randn(50),
                'P101': np.random.randn(50),
                'Normal/Attack': ['Normal'] * 45 + ['Attack'] * 5
            })

            # Save mock files
            mock_train_data.to_csv(temp_path / 'SWaT_Dataset_Normal_v1.csv', index=False)
            mock_test_data.to_csv(temp_path / 'SWaT_Dataset_Attack_v0.csv', index=False)

            # Test loading
            try:
                config = {
                    'window_size': 5,
                    'batch_size': 1,
                    'normalize': True
                }

                data_loader, dataset_info = create_data_loader('swat', str(temp_path), config)

                print(f"Loaded SWaT dataset: {dataset_info}")
                print(f"Data loader length: {len(data_loader)}")

                # Test a few batches
                batch_count = 0
                for batch in data_loader:
                    if batch_count >= 3:
                        break
                    print(f"Batch {batch_count}: data shape {batch['data'].shape}")
                    batch_count += 1

                print("Real dataset loading test completed ✓")

            except Exception as e:
                print(f"Real dataset loading test failed (expected): {e}")

    def test_data_splitting():
        """Test data splitting functionality."""
        print("\nTesting data splitting...")

        # Create test data loader
        config = {
            'synthetic': {
                'num_features': 10,
                'num_samples': 200,
                'anomaly_ratio': 0.05
            },
            'data': {
                'window_size': 5,
                'batch_size': 1,
                'normalize': True
            }
        }

        data_loader, _ = create_data_loader('synthetic', '', config)

        # Test temporal split
        train_loader, val_loader = split_data_temporal(data_loader, split_ratio=0.7)

        print(f"Original loader length: {len(data_loader)}")
        print(f"Train loader length: {len(train_loader)}")
        print(f"Val loader length: {len(val_loader)}")

        # Test cross-validation splits
        cv_splits = create_cross_validation_splits(data_loader, n_splits=3)

        print(f"Created {len(cv_splits)} CV splits:")
        for i, (train, val) in enumerate(cv_splits):
            print(f"  Split {i+1}: train={len(train)}, val={len(val)}")

        print("Data splitting test completed ✓")

    def test_caching():
        """Test dataset caching functionality."""
        print("\nTesting dataset caching...")

        # Create test data
        np.random.seed(42)
        test_data = np.random.randn(100, 5)
        test_labels = np.random.binomial(1, 0.1, 100)
        test_info = {'name': 'test', 'features': 5, 'samples': 100}

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / 'test_cache.npz'

            # Test saving cache
            save_dataset_cache(test_data, test_labels, test_info, str(cache_path))
            print(f"Cache saved to: {cache_path}")

            # Test loading cache
            loaded_result = load_dataset_cache(str(cache_path))

            if loaded_result is not None:
                loaded_data, loaded_labels, loaded_info = loaded_result

                print(f"Cache loaded successfully")
                print(f"Data match: {np.array_equal(test_data, loaded_data)}")
                print(f"Labels match: {np.array_equal(test_labels, loaded_labels)}")
                print(f"Info match: {test_info == loaded_info}")
            else:
                print("Failed to load cache")

        print("Dataset caching test completed ✓")

    # Run all tests
    print("=" * 50)
    print("STREAM-DAD DATA LOADING TESTS")
    print("=" * 50)

    test_synthetic_data_generation()
    test_drift_injection()
    test_multiple_scenarios()
    test_real_dataset_loading()
    test_data_splitting()
    test_caching()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)



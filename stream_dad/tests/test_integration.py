"""
Integration tests for Stream-DAD.

Tests the complete pipeline from data loading through training to evaluation.
"""

import pytest
import torch
import numpy as np
import tempfile
import yaml
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from stream_dad.core.models import StreamDAD, create_stream_dad_model
from stream_dad.utils.data_loading import create_data_loader, SyntheticDataGenerator
from stream_dad.utils.evaluation import ComprehensiveEvaluator, evaluate_anomaly_detection
from stream_dad.experiments.train import StreamDADTrainer


class TestStreamDADIntegration:
    """Integration tests for the complete Stream-DAD pipeline."""

    @pytest.fixture
    def config(self):
        """Default test configuration."""
        return {
            'model': {
                'hidden_dim': 32,
                'num_layers': 2,
                'dropout': 0.1,
                'lambda_ewc': 0.01,
                'lambda_cons': 0.001,
                'lambda_sparsity': 0.0001,
                'lambda_entropy': 0.001,
                'correlation_threshold': 0.3,
                'gating': {
                    'dropout': 0.1,
                    'importance_ema_alpha': 0.9,
                    'lambda_l1': 0.0001,
                    'lambda_entropy': 0.001
                },
                'drift': {
                    'window_size': 20,
                    'drift_threshold': 0.5,
                    'adaptation_rate': 0.1
                },
                'continual': {
                    'fisher': {
                        'fisher_update_freq': 50,
                        'fisher_decay': 0.95
                    },
                    'buffer': {
                        'capacity': 100,
                        'diversity_threshold': 0.1
                    },
                    'ewc_lambda': 0.01,
                    'fisher_update_interval': 50
                }
            },
            'data': {
                'window_size': 5,
                'stride': 1,
                'batch_size': 1,
                'normalize': True,
                'shuffle': False
            },
            'training': {
                'max_steps': 100,
                'eval_interval': 25,
                'save_interval': 50,
                'grad_clip': 1.0,
                'adaptation_threshold': 0.3
            },
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-4
            },
            'evaluation': {
                'window_size': 20
            },
            'synthetic': {
                'num_features': 20,
                'num_samples': 500,
                'anomaly_ratio': 0.1,
                'seed': 42,
                'drift': {
                    'type': 'gradual',
                    'magnitude': 0.2,
                    'start': 250,
                    'affected_features': None,
                    'num_affected_features': 5
                }
            }
        }

    @pytest.fixture
    def synthetic_data(self, config):
        """Generate synthetic test data."""
        generator = SyntheticDataGenerator(config['synthetic'])
        data, labels = generator.generate_multivariate_gaussian_drift()
        return data, labels

    def test_model_creation(self, config):
        """Test model creation and basic forward pass."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        assert model is not None
        assert model.input_dim == input_dim
        assert model.hidden_dim == config['model']['hidden_dim']

        # Test forward pass
        batch_size = 2
        window_size = config['data']['window_size']
        x = torch.randn(batch_size, window_size, input_dim)

        output = model(x, return_gates=True)

        assert 'reconstructed' in output
        assert 'anomaly_scores' in output
        assert 'drift_signal' in output
        assert 'gates' in output

        assert output['reconstructed'].shape == x.shape
        assert output['anomaly_scores'].shape == (batch_size, window_size)
        assert output['gates'].shape == (batch_size, input_dim)

    def test_data_loading(self, config, synthetic_data):
        """Test data loading and preprocessing."""
        data, labels = synthetic_data

        # Save temporary data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            np.save(data_path / 'data.npy', data)
            np.save(data_path / 'labels.npy', labels)

            # Test synthetic data loader
            data_loader, dataset_info = create_data_loader(
                dataset_name='synthetic',
                data_path=str(data_path),
                config=config['data']
            )

            assert dataset_info is not None
            assert 'features' in dataset_info
            assert len(data_loader) > 0

            # Test batch loading
            batch = next(iter(data_loader))
            assert 'data' in batch
            assert batch['data'].shape[2] == config['synthetic']['num_features']

    def test_loss_computation(self, config):
        """Test loss computation with all components."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        batch_size = 2
        window_size = config['data']['window_size']
        x = torch.randn(batch_size, window_size, input_dim)

        # Forward pass
        output = model(x, return_gates=True)

        # Compute losses
        loss_dict = model.compute_loss(x, output)

        assert 'total_loss' in loss_dict
        assert 'recon_loss' in loss_dict
        assert 'ewc_loss' in loss_dict
        assert 'consistency_loss' in loss_dict
        assert 'sparsity_loss' in loss_dict

        # Check that losses are scalars
        for loss_name, loss_value in loss_dict.items():
            assert loss_value.dim() == 0, f"{loss_name} should be a scalar"
            assert torch.isfinite(loss_value), f"{loss_name} should be finite"

    def test_drift_detection(self, config):
        """Test drift detection functionality."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        # Generate data with known drift
        window_size = config['data']['window_size']

        # Normal data
        normal_data = torch.randn(10, window_size, input_dim)

        # Drifted data (shift in mean)
        drifted_data = torch.randn(10, window_size, input_dim) + 2.0

        # Process normal data
        normal_drift_signals = []
        for i in range(len(normal_data)):
            output = model(normal_data[i:i+1])
            drift_signal = output['drift_signal']
            normal_drift_signals.append(drift_signal.mean().item())

        # Process drifted data
        drifted_drift_signals = []
        for i in range(len(drifted_data)):
            output = model(drifted_data[i:i+1])
            drift_signal = output['drift_signal']
            drifted_drift_signals.append(drift_signal.mean().item())

        # Drift signals should be higher for drifted data
        avg_normal_drift = np.mean(normal_drift_signals[-5:])  # Last 5 samples
        avg_drifted_drift = np.mean(drifted_drift_signals[-5:])  # Last 5 samples

        assert avg_drifted_drift > avg_normal_drift, "Drift detector should detect higher drift in shifted data"

    def test_continual_learning(self, config):
        """Test continual learning components."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        # Initial Fisher information should be empty
        assert len(model.continual_learner.fisher_computer.fisher_dict) == 0

        # Generate some data for Fisher computation
        x = torch.randn(5, config['data']['window_size'], input_dim)

        # Update Fisher information
        model.continual_learner.update_fisher_information(x)

        # Fisher information should now be populated
        assert len(model.continual_learner.fisher_computer.fisher_dict) > 0

        # Test EWC loss computation
        ewc_loss = model.continual_learner.compute_ewc_loss()
        assert torch.isfinite(ewc_loss)
        assert ewc_loss >= 0  # EWC loss should be non-negative

    def test_feature_gating(self, config):
        """Test dynamic feature gating."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        batch_size = 3
        window_size = config['data']['window_size']
        x = torch.randn(batch_size, window_size, input_dim)

        # Forward pass with gates
        output = model(x, return_gates=True)
        gates = output['gates']

        # Check gate properties
        assert gates.shape == (batch_size, input_dim)
        assert torch.all(gates >= 0) and torch.all(gates <= 1), "Gates should be in [0, 1]"

        # Test gate statistics
        gate_stats = model.gating_network.get_gate_statistics(gates)
        assert 'mean_gate_value' in gate_stats
        assert 'sparsity_ratio' in gate_stats
        assert 'active_features' in gate_stats

        # Sparsity ratio should be reasonable (not all features selected)
        assert 0 <= gate_stats['sparsity_ratio'] <= 1

    def test_evaluation_metrics(self, config, synthetic_data):
        """Test evaluation metrics computation."""
        data, labels = synthetic_data
        input_dim = data.shape[1]

        model = create_stream_dad_model(input_dim, config['model'])
        evaluator = ComprehensiveEvaluator(config['evaluation'])

        # Simulate evaluation loop
        window_size = config['data']['window_size']
        for i in range(0, min(100, len(data) - window_size), 10):
            # Create window
            window_data = torch.FloatTensor(data[i:i+window_size]).unsqueeze(0)
            window_labels = labels[i:i+window_size]

            # Forward pass
            output = model(window_data, return_gates=True)

            # Extract results
            anomaly_scores = output['anomaly_scores'].detach().numpy()
            predictions = (anomaly_scores > np.percentile(anomaly_scores, 90)).astype(int)
            gates = output['gates'].detach().numpy()
            selected_features = (gates > 0.5).astype(int)

            # Update evaluator
            evaluator.update(
                predictions=predictions.flatten(),
                labels=window_labels,
                scores=anomaly_scores.flatten(),
                selected_features=selected_features[0],
                timestamp=i
            )

        # Compute comprehensive results
        results = evaluator.evaluate_comprehensive()

        assert 'anomaly_detection' in results
        assert 'feature_selection' in results
        assert 'computational' in results

        # Check anomaly detection metrics
        ad_metrics = results['anomaly_detection']
        assert 'f1_score' in ad_metrics
        assert 'auroc' in ad_metrics
        assert 'false_alarm_rate' in ad_metrics

        # Check feature selection metrics
        fs_metrics = results['feature_selection']
        assert 'stability' in fs_metrics
        assert 'sparsity' in fs_metrics

    def test_end_to_end_training(self, config, synthetic_data):
        """Test end-to-end training pipeline."""
        data, labels = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save temporary data
            data_path = Path(temp_dir)
            np.save(data_path / 'data.npy', data)
            np.save(data_path / 'labels.npy', labels)

            # Update config for test
            config['experiment'] = {'save_dir': str(data_path / 'experiments')}

            # Create data loader
            train_loader, dataset_info = create_data_loader(
                dataset_name='synthetic',
                data_path=str(data_path),
                config=config['data']
            )

            # Initialize trainer
            trainer = StreamDADTrainer(config)
            trainer.setup_model(dataset_info['features'])

            # Run short training
            config['training']['max_steps'] = 20  # Very short for testing
            results = trainer.train_streaming(train_loader, train_loader)

            # Check that training completed
            assert results is not None
            assert 'anomaly_detection' in results

            # Check that model was updated
            assert trainer.current_step > 0

    def test_checkpointing(self, config):
        """Test model checkpointing and loading."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'test_checkpoint.pt'

            # Save checkpoint
            model.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Create new model and load checkpoint
            new_model = create_stream_dad_model(input_dim, config['model'])
            new_model.load_checkpoint(str(checkpoint_path))

            # Test that models produce same output
            x = torch.randn(1, config['data']['window_size'], input_dim)

            with torch.no_grad():
                output1 = model(x)
                output2 = new_model(x)

            # Outputs should be very close (allowing for minor floating point differences)
            assert torch.allclose(output1['reconstructed'], output2['reconstructed'], atol=1e-6)

    def test_memory_efficiency(self, config):
        """Test that memory usage remains bounded during training."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Simulate training loop
        for i in range(50):
            x = torch.randn(1, config['data']['window_size'], input_dim)
            if torch.cuda.is_available():
                x = x.cuda()
                model = model.cuda()

            output = model(x, return_gates=True)
            loss_dict = model.compute_loss(x, output)

            # Simulate backward pass
            loss_dict['total_loss'].backward()
            model.zero_grad()

            # Check memory growth
            if torch.cuda.is_available() and i % 10 == 0:
                current_memory = torch.cuda.memory_allocated()
                memory_growth = current_memory - initial_memory

                # Memory shouldn't grow excessively (allow some growth for caching)
                assert memory_growth < 100 * 1024 * 1024, f"Memory grew by {memory_growth / (1024*1024):.1f} MB"

    def test_device_compatibility(self, config):
        """Test model works on different devices."""
        input_dim = 20
        model = create_stream_dad_model(input_dim, config['model'])

        # Test CPU
        x_cpu = torch.randn(1, config['data']['window_size'], input_dim)
        output_cpu = model(x_cpu)
        assert output_cpu['reconstructed'].device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = model_cuda(x_cuda)
            assert output_cuda['reconstructed'].device.type == 'cuda'

            # Results should be close between devices
            with torch.no_grad():
                diff = torch.abs(output_cpu['reconstructed'] - output_cuda['reconstructed'].cpu())
                assert torch.max(diff) < 1e-4, "Results should be consistent across devices"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
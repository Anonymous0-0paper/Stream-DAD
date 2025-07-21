import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def inject_synthetic_drift(data: np.ndarray,
                           labels: np.ndarray,
                           drift_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject controlled drift into existing dataset.

    Args:
        data: Original data [num_samples, num_features]
        labels: Original labels [num_samples]
        drift_config: Configuration for drift injection

    Returns:
        drifted_data: Data with injected drift
        drift_labels: Updated labels
    """
    if not drift_config:
        logger.warning("No drift config provided, returning original data")
        return data, labels

    drift_type = drift_config.get('type', 'gradual')
    drift_start = drift_config.get('start', len(data) // 2)
    drift_magnitude = drift_config.get('magnitude', 0.5)
    affected_features = drift_config.get('affected_features', None)

    logger.info(f"Injecting {drift_type} drift starting at sample {drift_start}")
    logger.info(f"Drift magnitude: {drift_magnitude}")

    drifted_data = data.copy()
    drifted_labels = labels.copy()

    # Convert relative start position to absolute if needed
    if isinstance(drift_start, float) and 0 <= drift_start <= 1:
        drift_start = int(drift_start * len(data))

    if affected_features is None:
        # Affect random subset of features
        num_affected = drift_config.get('num_affected_features', data.shape[1] // 3)
        if isinstance(num_affected, float) and 0 <= num_affected <= 1:
            num_affected = int(num_affected * data.shape[1])

        np.random.seed(drift_config.get('seed', 42))
        affected_features = np.random.choice(data.shape[1], num_affected, replace=False)
        logger.info(f"Randomly selected {num_affected} features to affect: {affected_features}")
    else:
        logger.info(f"Affecting specified features: {affected_features}")

    # Apply drift based on type
    for i in range(drift_start, len(data)):
        progress = (i - drift_start) / (len(data) - drift_start)

        if drift_type == 'gradual':
            # Linear increase in mean
            shift = drift_magnitude * progress
        elif drift_type == 'sudden':
            # Step change
            shift = drift_magnitude
        elif drift_type == 'recurring':
            # Sinusoidal pattern
            frequency = drift_config.get('frequency', 0.01)
            shift = drift_magnitude * np.sin(2 * np.pi * frequency * i)
        elif drift_type == 'exponential':
            # Exponential drift
            decay_rate = drift_config.get('decay_rate', 0.01)
            shift = drift_magnitude * (1 - np.exp(-decay_rate * progress))
        elif drift_type == 'polynomial':
            # Polynomial drift (quadratic by default)
            degree = drift_config.get('degree', 2)
            shift = drift_magnitude * (progress ** degree)
        else:
            logger.warning(f"Unknown drift type: {drift_type}, using gradual")
            shift = drift_magnitude * progress

        # Apply shift to affected features
        for feature_idx in affected_features:
            if drift_config.get('drift_mode', 'mean') == 'mean':
                # Shift mean
                drifted_data[i, feature_idx] += shift
            elif drift_config.get('drift_mode', 'mean') == 'variance':
                # Scale variance
                drifted_data[i, feature_idx] *= (1 + shift)
            elif drift_config.get('drift_mode', 'mean') == 'both':
                # Both mean and variance
                drifted_data[i, feature_idx] = drifted_data[i, feature_idx] * (1 + shift * 0.5) + shift * 0.5
            elif drift_config.get('drift_mode', 'mean') == 'rotation':
                # Rotate feature relationships (only works with pairs)
                if len(affected_features) >= 2 and feature_idx < len(affected_features) - 1:
                    idx1, idx2 = affected_features[feature_idx], affected_features[feature_idx + 1]
                    angle = shift * np.pi / 4  # Max 45 degree rotation

                    x1, x2 = drifted_data[i, idx1], drifted_data[i, idx2]
                    drifted_data[i, idx1] = x1 * np.cos(angle) - x2 * np.sin(angle)
                    drifted_data[i, idx2] = x1 * np.sin(angle) + x2 * np.cos(angle)

    # Add drift-induced anomalies if specified
    if drift_config.get('add_drift_anomalies', False):
        drift_anomaly_ratio = drift_config.get('drift_anomaly_ratio', 0.02)
        drift_anomaly_magnitude = drift_config.get('drift_anomaly_magnitude', 2.0)

        # Create drift anomalies in the drift period
        drift_period_length = len(data) - drift_start
        num_drift_anomalies = int(drift_period_length * drift_anomaly_ratio)

        # Randomly select time points for drift anomalies
        np.random.seed(drift_config.get('seed', 42) + 1)
        drift_anomaly_indices = np.random.choice(
            range(drift_start, len(data)),
            num_drift_anomalies,
            replace=False
        )

        logger.info(f"Adding {num_drift_anomalies} drift-induced anomalies")

        for idx in drift_anomaly_indices:
            # Add anomaly as extreme deviation in random direction
            anomaly_direction = np.random.randn(len(affected_features))
            anomaly_direction /= np.linalg.norm(anomaly_direction)

            for j, feature_idx in enumerate(affected_features):
                drifted_data[idx, feature_idx] += drift_anomaly_magnitude * anomaly_direction[j]

            # Mark as anomaly
            drifted_labels[idx] = 1

    logger.info(f"Drift injection completed. Affected {len(affected_features)} features")

    return drifted_data, drifted_labels


def validate_drift_injection(original_data: np.ndarray,
                             drifted_data: np.ndarray,
                             drift_config: Dict) -> Dict[str, float]:
    """
    Validate that drift was properly injected by measuring distributional changes.

    Args:
        original_data: Data before drift injection
        drifted_data: Data after drift injection
        drift_config: Configuration used for drift injection

    Returns:
        validation_metrics: Dictionary of validation metrics
    """
    drift_start = drift_config.get('start', len(original_data) // 2)

    # Convert relative start to absolute
    if isinstance(drift_start, float) and 0 <= drift_start <= 1:
        drift_start = int(drift_start * len(original_data))

    affected_features = drift_config.get('affected_features', None)

    if affected_features is None:
        num_affected = drift_config.get('num_affected_features', original_data.shape[1] // 3)
        if isinstance(num_affected, float) and 0 <= num_affected <= 1:
            num_affected = int(num_affected * original_data.shape[1])
        # Approximate the affected features (we can't know exactly which were randomly selected)
        affected_features = list(range(num_affected))

    # Compare distributions before and after drift
    before_drift = original_data[:drift_start]
    after_drift_original = original_data[drift_start:]
    after_drift_modified = drifted_data[drift_start:]

    validation_metrics = {}

    # Measure mean shift
    mean_shifts = []
    for feature_idx in affected_features:
        if feature_idx < original_data.shape[1]:
            before_mean = np.mean(before_drift[:, feature_idx])
            after_mean_original = np.mean(after_drift_original[:, feature_idx])
            after_mean_modified = np.mean(after_drift_modified[:, feature_idx])

            # Measure the change introduced by drift injection
            drift_induced_shift = abs(after_mean_modified - after_mean_original)
            # Also measure relative to before-drift baseline
            baseline_shift = abs(after_mean_modified - before_mean)

            mean_shifts.append({
                'drift_induced': drift_induced_shift,
                'baseline_relative': baseline_shift,
                'feature_idx': feature_idx
            })

    if mean_shifts:
        drift_induced_shifts = [ms['drift_induced'] for ms in mean_shifts]
        baseline_shifts = [ms['baseline_relative'] for ms in mean_shifts]

        validation_metrics['mean_shift'] = {
            'drift_induced_avg': float(np.mean(drift_induced_shifts)),
            'drift_induced_max': float(np.max(drift_induced_shifts)),
            'drift_induced_min': float(np.min(drift_induced_shifts)),
            'baseline_relative_avg': float(np.mean(baseline_shifts)),
            'baseline_relative_max': float(np.max(baseline_shifts)),
            'baseline_relative_min': float(np.min(baseline_shifts))
        }

    # Measure variance change
    var_changes = []
    for feature_idx in affected_features:
        if feature_idx < original_data.shape[1]:
            before_var = np.var(before_drift[:, feature_idx])
            after_var_original = np.var(after_drift_original[:, feature_idx])
            after_var_modified = np.var(after_drift_modified[:, feature_idx])

            # Relative variance change due to drift
            if after_var_original > 1e-8:
                var_change = abs(after_var_modified - after_var_original) / after_var_original
            else:
                var_change = 0.0

            var_changes.append(var_change)

    if var_changes:
        validation_metrics['variance_change'] = {
            'average': float(np.mean(var_changes)),
            'max': float(np.max(var_changes)),
            'min': float(np.min(var_changes)),
            'std': float(np.std(var_changes))
        }

    # Measure KL divergence (simplified using histograms)
    try:
        kl_divergences = []

        for feature_idx in affected_features:
            if feature_idx < original_data.shape[1]:
                # Create histograms for before and after drift periods
                feature_data_before = before_drift[:, feature_idx]
                feature_data_after_original = after_drift_original[:, feature_idx]
                feature_data_after_modified = after_drift_modified[:, feature_idx]

                # Use same bins for fair comparison
                all_data = np.concatenate([feature_data_before, feature_data_after_original, feature_data_after_modified])
                bins = np.linspace(all_data.min(), all_data.max(), 50)

                hist_original, _ = np.histogram(feature_data_after_original, bins=bins, density=True)
                hist_modified, _ = np.histogram(feature_data_after_modified, bins=bins, density=True)

                # Add small epsilon to avoid log(0)
                hist_original += 1e-8
                hist_modified += 1e-8

                # Normalize to probabilities
                hist_original /= hist_original.sum()
                hist_modified /= hist_modified.sum()

                # Compute KL divergence
                kl_div = entropy(hist_modified, hist_original)
                kl_divergences.append(kl_div)

        if kl_divergences:
            validation_metrics['kl_divergence'] = {
                'average': float(np.mean(kl_divergences)),
                'max': float(np.max(kl_divergences)),
                'min': float(np.min(kl_divergences)),
                'std': float(np.std(kl_divergences))
            }
    except Exception as e:
        logger.warning(f"Failed to compute KL divergence: {e}")

    # Wasserstein distance (simplified 1D version)
    try:
        wasserstein_distances = []

        for feature_idx in affected_features:
            if feature_idx < original_data.shape[1]:
                original_samples = after_drift_original[:, feature_idx]
                modified_samples = after_drift_modified[:, feature_idx]

                # Sort samples for Wasserstein distance computation
                original_sorted = np.sort(original_samples)
                modified_sorted = np.sort(modified_samples)

                # Make arrays same length for comparison
                min_len = min(len(original_sorted), len(modified_sorted))
                original_sorted = original_sorted[:min_len]
                modified_sorted = modified_sorted[:min_len]

                # Compute 1D Wasserstein distance
                wasserstein_dist = np.mean(np.abs(original_sorted - modified_sorted))
                wasserstein_distances.append(wasserstein_dist)

        if wasserstein_distances:
            validation_metrics['wasserstein_distance'] = {
                'average': float(np.mean(wasserstein_distances)),
                'max': float(np.max(wasserstein_distances)),
                'min': float(np.min(wasserstein_distances))
            }
    except Exception as e:
        logger.warning(f"Failed to compute Wasserstein distance: {e}")

    # Overall drift magnitude (composite score)
    if 'mean_shift' in validation_metrics and 'variance_change' in validation_metrics:
        overall_drift = np.mean([
            validation_metrics['mean_shift']['drift_induced_avg'],
            validation_metrics['variance_change']['average']
        ])
        validation_metrics['overall_drift_magnitude'] = float(overall_drift)

    # Drift detection success rate
    expected_magnitude = drift_config.get('magnitude', 0.5)
    detected_magnitude = validation_metrics.get('overall_drift_magnitude', 0)

    validation_metrics['drift_detection_success'] = {
        'expected_magnitude': float(expected_magnitude),
        'detected_magnitude': float(detected_magnitude),
        'detection_ratio': float(detected_magnitude / expected_magnitude) if expected_magnitude > 0 else 0,
        'detection_success': bool(detected_magnitude > expected_magnitude * 0.1)  # At least 10% of expected
    }

    logger.info(f"Drift validation completed:")
    logger.info(f"  Mean shift (drift-induced): {validation_metrics.get('mean_shift', {}).get('drift_induced_avg', 0):.4f}")
    logger.info(f"  Variance change: {validation_metrics.get('variance_change', {}).get('average', 0):.4f}")
    logger.info(f"  Overall drift magnitude: {validation_metrics.get('overall_drift_magnitude', 0):.4f}")
    logger.info(f"  Detection success: {validation_metrics['drift_detection_success']['detection_success']}")

    return validation_metrics


def create_drift_scenarios(base_data: np.ndarray,
                           base_labels: np.ndarray,
                           scenario_configs: List[Dict]) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Create multiple drift scenarios from base data for comprehensive evaluation.

    Args:
        base_data: Original data without drift
        base_labels: Original labels
        scenario_configs: List of drift configurations to generate

    Returns:
        scenarios: List of (data, labels, config) tuples
    """
    scenarios = []

    logger.info(f"Creating {len(scenario_configs)} drift scenarios from base data")
    logger.info(f"Base data shape: {base_data.shape}")

    for i, config in enumerate(scenario_configs):
        scenario_name = config.get('name', f'Scenario_{i+1}')
        logger.info(f"Creating scenario {i+1}/{len(scenario_configs)}: {scenario_name}")

        try:
            # Create a copy of the config to avoid modifying original
            scenario_config = config.copy()

            # Add unique seed for each scenario to ensure reproducibility
            if 'seed' not in scenario_config:
                scenario_config['seed'] = 42 + i

            # Inject drift
            drifted_data, drifted_labels = inject_synthetic_drift(
                base_data.copy(), base_labels.copy(), scenario_config
            )

            # Validate drift injection
            validation_metrics = validate_drift_injection(base_data, drifted_data, scenario_config)

            # Add scenario metadata
            scenario_info = scenario_config.copy()
            scenario_info.update({
                'scenario_id': i,
                'scenario_name': scenario_name,
                'base_samples': len(base_data),
                'base_features': base_data.shape[1],
                'base_anomaly_ratio': float(base_labels.mean()),
                'final_anomaly_ratio': float(drifted_labels.mean()),
                'validation_metrics': validation_metrics,
                'drift_success': validation_metrics['drift_detection_success']['detection_success']
            })

            scenarios.append((drifted_data, drifted_labels, scenario_info))

            logger.info(f"  Scenario created successfully")
            logger.info(f"  Drift magnitude: {validation_metrics.get('overall_drift_magnitude', 0):.4f}")
            logger.info(f"  Final anomaly ratio: {scenario_info['final_anomaly_ratio']:.4f}")

        except Exception as e:
            logger.error(f"Failed to create scenario {i+1}: {e}")
            # Create a failed scenario entry
            failed_scenario_info = config.copy()
            failed_scenario_info.update({
                'scenario_id': i,
                'scenario_name': scenario_name,
                'error': str(e),
                'failed': True
            })
            scenarios.append((base_data.copy(), base_labels.copy(), failed_scenario_info))

    successful_scenarios = [s for s in scenarios if not s[2].get('failed', False)]
    logger.info(f"Successfully created {len(successful_scenarios)}/{len(scenario_configs)} scenarios")

    return scenarios


# Predefined drift scenario configurations for standard evaluation
STANDARD_DRIFT_SCENARIOS = [
    {
        'name': 'Gradual Drift - Low Magnitude',
        'type': 'gradual',
        'magnitude': 0.2,
        'start': 0.5,  # Start at 50% of data
        'num_affected_features': 0.3,  # 30% of features
        'drift_mode': 'mean',
        'description': 'Slow gradual drift affecting minority of features'
    },
    {
        'name': 'Gradual Drift - High Magnitude',
        'type': 'gradual',
        'magnitude': 1.0,
        'start': 0.5,
        'num_affected_features': 0.3,
        'drift_mode': 'mean',
        'description': 'Strong gradual drift with significant impact'
    },
    {
        'name': 'Sudden Drift - Medium Magnitude',
        'type': 'sudden',
        'magnitude': 0.5,
        'start': 0.5,
        'num_affected_features': 0.3,
        'drift_mode': 'mean',
        'description': 'Abrupt step change in feature distributions'
    },
    {
        'name': 'Recurring Drift - Sinusoidal',
        'type': 'recurring',
        'magnitude': 0.3,
        'start': 0.3,
        'frequency': 0.01,
        'num_affected_features': 0.2,
        'drift_mode': 'mean',
        'description': 'Cyclical drift pattern with periodic returns'
    },
    {
        'name': 'Variance Drift - Gradual',
        'type': 'gradual',
        'magnitude': 0.5,
        'start': 0.5,
        'num_affected_features': 0.3,
        'drift_mode': 'variance',
        'description': 'Gradual change in feature variance/spread'
    },
    {
        'name': 'Combined Drift - Mean and Variance',
        'type': 'gradual',
        'magnitude': 0.4,
        'start': 0.4,
        'num_affected_features': 0.4,
        'drift_mode': 'both',
        'description': 'Combined drift affecting both mean and variance'
    },
    {
        'name': 'Drift with Induced Anomalies',
        'type': 'gradual',
        'magnitude': 0.3,
        'start': 0.5,
        'num_affected_features': 0.3,
        'drift_mode': 'mean',
        'add_drift_anomalies': True,
        'drift_anomaly_ratio': 0.02,
        'drift_anomaly_magnitude': 2.0,
        'description': 'Drift with additional anomalies during transition'
    },
    {
        'name': 'Exponential Drift',
        'type': 'exponential',
        'magnitude': 0.6,
        'start': 0.3,
        'decay_rate': 0.02,
        'num_affected_features': 0.25,
        'drift_mode': 'mean',
        'description': 'Exponential drift with rapid initial change'
    },
    {
        'name': 'Rotation Drift',
        'type': 'gradual',
        'magnitude': 0.4,
        'start': 0.4,
        'num_affected_features': 0.4,
        'drift_mode': 'rotation',
        'description': 'Rotation of feature relationships'
    }
]


# Example usage and testing functions
def plot_drift_scenarios(scenarios: List[Tuple[np.ndarray, np.ndarray, Dict]],
                         feature_idx: int = 0,
                         save_path: Optional[str] = None):
    """
    Visualize multiple drift scenarios for a specific feature.

    Args:
        scenarios: List of (data, labels, config) tuples from create_drift_scenarios
        feature_idx: Index of feature to visualize
        save_path: Optional path to save the plot
    """
    n_scenarios = len(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_scenarios == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (data, labels, config) in enumerate(scenarios):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        if config.get('failed', False):
            ax.text(0.5, 0.5, f"Failed: {config.get('error', 'Unknown')}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config.get('scenario_name', f'Scenario {i+1}'))
            continue

        # Plot feature over time
        feature_data = data[:, feature_idx]
        time_steps = np.arange(len(feature_data))

        # Highlight drift start
        drift_start = config.get('start', 0.5)
        if isinstance(drift_start, float):
            drift_start = int(drift_start * len(data))

        # Plot normal periods and drift periods in different colors
        ax.plot(time_steps[:drift_start], feature_data[:drift_start],
                'b-', alpha=0.7, label='Pre-drift')
        ax.plot(time_steps[drift_start:], feature_data[drift_start:],
                'r-', alpha=0.7, label='Post-drift')

        # Mark anomalies
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, feature_data[anomaly_indices],
                       c='red', s=20, alpha=0.6, marker='x', label='Anomalies')

        # Add vertical line at drift start
        ax.axvline(x=drift_start, color='green', linestyle='--', alpha=0.5, label='Drift start')

        ax.set_title(f"{config.get('scenario_name', f'Scenario {i+1}')}\n"
                     f"Drift: {config.get('overall_drift_magnitude', 0):.3f}")
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'Feature {feature_idx}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_scenarios, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Drift scenarios plot saved to {save_path}")

    plt.show()


def test_drift_functions():
    """
    Test the drift injection functions with example data.
    """
    print("Testing drift injection functions...")

    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    base_data = np.random.randn(n_samples, n_features)
    base_labels = np.random.binomial(1, 0.05, n_samples)

    print(f"Created base data: {base_data.shape}")
    print(f"Base anomaly ratio: {base_labels.mean():.4f}")

    # Test 1: Single drift injection
    print("\n1. Testing single drift injection...")
    drift_config = {
        'type': 'gradual',
        'magnitude': 0.5,
        'start': 0.6,
        'num_affected_features': 0.3,
        'drift_mode': 'mean'
    }

    drifted_data, drifted_labels = inject_synthetic_drift(base_data, base_labels, drift_config)
    validation = validate_drift_injection(base_data, drifted_data, drift_config)

    print(f"Drift injection completed")
    print(f"Overall drift magnitude: {validation['overall_drift_magnitude']:.4f}")
    print(f"Detection success: {validation['drift_detection_success']['detection_success']}")

    # Test 2: Multiple scenarios
    print("\n2. Testing multiple drift scenarios...")
    scenarios = create_drift_scenarios(base_data, base_labels, STANDARD_DRIFT_SCENARIOS[:3])

    print(f"Created {len(scenarios)} scenarios:")
    for i, (data, labels, config) in enumerate(scenarios):
        if not config.get('failed', False):
            drift_mag = config['validation_metrics']['overall_drift_magnitude']
            anomaly_ratio = config['final_anomaly_ratio']
            print(f"  {config['scenario_name']}: drift={drift_mag:.4f}, anomalies={anomaly_ratio:.4f}")

    # Test 3: Visualization (optional)
    print("\n3. Testing visualization...")
    try:
        plot_drift_scenarios(scenarios[:3], feature_idx=0)
        print("Visualization test completed")
    except Exception as e:
        print(f"Visualization failed (expected if no display): {e}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_drift_functions()
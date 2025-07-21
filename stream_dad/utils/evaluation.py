"""
Evaluation Metrics and Protocols for Stream-DAD.

This module provides comprehensive evaluation metrics for anomaly detection
and concept drift adaptation, including standard AD metrics and novel
drift-specific measures.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
import logging
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AnomalyDetectionMetrics:
    """
    Comprehensive anomaly detection metrics calculator.

    Computes standard metrics like F1, AUROC, AUPRC and also provides
    detailed analysis of detection performance over time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.labels = []
        self.scores = []
        self.timestamps = []

    def update(self,
               pred: np.ndarray,
               label: np.ndarray,
               score: np.ndarray,
               timestamp: Optional[int] = None):
        """
        Update with new predictions and labels.

        Args:
            pred: Binary predictions [batch_size]
            label: True binary labels [batch_size]
            score: Anomaly scores [batch_size]
            timestamp: Optional timestamp for temporal analysis
        """
        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())
        self.scores.extend(score.flatten())

        if timestamp is not None:
            self.timestamps.extend([timestamp] * len(pred.flatten()))
        else:
            self.timestamps.extend(list(range(len(self.timestamps),
                                              len(self.timestamps) + len(pred.flatten()))))

    def compute_metrics(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute comprehensive anomaly detection metrics.

        Args:
            threshold: Decision threshold (if None, uses optimal threshold)

        Returns:
            metrics: Dictionary of computed metrics
        """
        if len(self.labels) == 0:
            return {}

        labels = np.array(self.labels)
        scores = np.array(self.scores)

        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(labels, scores)

        predictions = (scores >= threshold).astype(int)

        # Basic classification metrics
        metrics = {
            'threshold': threshold,
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'false_alarm_rate': self._compute_false_alarm_rate(labels, predictions),
            'detection_delay': self._compute_detection_delay(labels, predictions)
        }

        # ROC and PR metrics (only if both classes present)
        if len(np.unique(labels)) > 1:
            metrics.update({
                'auroc': roc_auc_score(labels, scores),
                'auprc': average_precision_score(labels, scores)
            })
        else:
            metrics.update({'auroc': 0.0, 'auprc': 0.0})

        # Confusion matrix components
        if len(np.unique(labels)) > 1 and len(np.unique(predictions)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            metrics.update({
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
            })

        return metrics

    def _find_optimal_threshold(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """Find optimal threshold maximizing F1 score."""
        if len(np.unique(labels)) == 1:
            return np.median(scores)

        # Try different thresholds
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_f1 = 0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def _compute_false_alarm_rate(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Compute false alarm rate (FPR)."""
        normal_mask = (labels == 0)
        if normal_mask.sum() == 0:
            return 0.0

        false_alarms = ((predictions == 1) & (labels == 0)).sum()
        return false_alarms / normal_mask.sum()

    def _compute_detection_delay(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Compute average detection delay for anomalies."""
        delays = []

        # Find anomaly segments
        anomaly_starts = []
        in_anomaly = False

        for i, label in enumerate(labels):
            if label == 1 and not in_anomaly:
                anomaly_starts.append(i)
                in_anomaly = True
            elif label == 0 and in_anomaly:
                in_anomaly = False

        # Compute delay for each anomaly segment
        for start in anomaly_starts:
            # Find first detection after anomaly start
            for i in range(start, len(predictions)):
                if predictions[i] == 1:
                    delays.append(i - start)
                    break
            else:
                # Anomaly was never detected
                delays.append(len(predictions) - start)

        return np.mean(delays) if delays else 0.0

    def compute_temporal_metrics(self, window_size: int = 100) -> Dict[str, List[float]]:
        """
        Compute metrics over sliding time windows.

        Args:
            window_size: Size of sliding window

        Returns:
            temporal_metrics: Time-series of metrics
        """
        if len(self.labels) < window_size:
            return {}

        labels = np.array(self.labels)
        scores = np.array(self.scores)

        temporal_metrics = defaultdict(list)

        for i in range(window_size, len(labels)):
            window_labels = labels[i-window_size:i]
            window_scores = scores[i-window_size:i]

            if len(np.unique(window_labels)) > 1:
                # Find optimal threshold for this window
                threshold = self._find_optimal_threshold(window_labels, window_scores)
                window_predictions = (window_scores >= threshold).astype(int)

                # Compute metrics for this window
                f1 = f1_score(window_labels, window_predictions, zero_division=0)
                precision = precision_score(window_labels, window_predictions, zero_division=0)
                recall = recall_score(window_labels, window_predictions, zero_division=0)
                far = self._compute_false_alarm_rate(window_labels, window_predictions)

                temporal_metrics['f1_score'].append(f1)
                temporal_metrics['precision'].append(precision)
                temporal_metrics['recall'].append(recall)
                temporal_metrics['false_alarm_rate'].append(far)
                temporal_metrics['timestamp'].append(self.timestamps[i] if self.timestamps else i)
            else:
                # No diversity in labels, append zeros
                temporal_metrics['f1_score'].append(0.0)
                temporal_metrics['precision'].append(0.0)
                temporal_metrics['recall'].append(0.0)
                temporal_metrics['false_alarm_rate'].append(0.0)
                temporal_metrics['timestamp'].append(self.timestamps[i] if self.timestamps else i)

        return dict(temporal_metrics)


class DriftAdaptationMetrics:
    """
    Metrics specific to concept drift adaptation performance.

    Measures how well the model adapts to distribution changes and
    maintains performance during drift periods.
    """

    def __init__(self, drift_detector=None):
        self.drift_detector = drift_detector
        self.reset()

    def reset(self):
        """Reset adaptation metrics."""
        self.drift_events = []
        self.adaptation_times = []
        self.performance_before_drift = []
        self.performance_after_drift = []
        self.forgetting_events = []

    def record_drift_event(self,
                           timestamp: int,
                           drift_magnitude: float,
                           performance_before: float,
                           adaptation_window: int = 100):
        """
        Record a drift event and track adaptation.

        Args:
            timestamp: When drift was detected
            drift_magnitude: Magnitude of the drift
            performance_before: Performance before drift
            adaptation_window: Window to measure adaptation
        """
        self.drift_events.append({
            'timestamp': timestamp,
            'magnitude': drift_magnitude,
            'performance_before': performance_before,
            'adaptation_window': adaptation_window
        })

    def record_adaptation_completion(self,
                                     timestamp: int,
                                     performance_after: float,
                                     adaptation_time: int):
        """
        Record when adaptation is complete.

        Args:
            timestamp: When adaptation completed
            performance_after: Performance after adaptation
            adaptation_time: Time taken to adapt
        """
        self.adaptation_times.append(adaptation_time)
        self.performance_after_drift.append(performance_after)

    def compute_adaptation_latency(self) -> Dict[str, float]:
        """Compute adaptation latency statistics."""
        if not self.adaptation_times:
            return {'mean_latency': 0.0, 'std_latency': 0.0, 'max_latency': 0.0}

        times = np.array(self.adaptation_times)
        return {
            'mean_latency': float(np.mean(times)),
            'std_latency': float(np.std(times)),
            'max_latency': float(np.max(times)),
            'min_latency': float(np.min(times)),
            'median_latency': float(np.median(times))
        }

    def compute_forgetting_ratio(self) -> float:
        """Compute catastrophic forgetting ratio."""
        if len(self.performance_before_drift) == 0 or len(self.performance_after_drift) == 0:
            return 0.0

        before = np.array(self.performance_before_drift)
        after = np.array(self.performance_after_drift)

        # Forgetting ratio: relative performance degradation
        min_len = min(len(before), len(after))
        if min_len == 0:
            return 0.0

        performance_drop = before[:min_len] - after[:min_len]
        forgetting_ratio = np.mean(np.maximum(performance_drop, 0) / (before[:min_len] + 1e-8))

        return float(forgetting_ratio)

    def compute_drift_detection_accuracy(self,
                                         true_drift_points: List[int],
                                         detected_drift_points: List[int],
                                         tolerance: int = 50) -> Dict[str, float]:
        """
        Compute accuracy of drift detection.

        Args:
            true_drift_points: Ground truth drift timestamps
            detected_drift_points: Detected drift timestamps
            tolerance: Tolerance window for matching detections

        Returns:
            detection_metrics: Precision, recall, F1 of drift detection
        """
        if not true_drift_points and not detected_drift_points:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}

        if not true_drift_points:
            return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0}

        if not detected_drift_points:
            return {'precision': 1.0, 'recall': 0.0, 'f1_score': 0.0}

        # Match detected drifts to true drifts within tolerance
        true_positives = 0
        matched_true = set()

        for detected in detected_drift_points:
            for i, true_point in enumerate(true_drift_points):
                if i not in matched_true and abs(detected - true_point) <= tolerance:
                    true_positives += 1
                    matched_true.add(i)
                    break

        false_positives = len(detected_drift_points) - true_positives
        false_negatives = len(true_drift_points) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


class FeatureSelectionMetrics:
    """
    Metrics for evaluating dynamic feature selection performance.

    Measures consistency, stability, and quality of feature selection
    decisions over time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset feature selection tracking."""
        self.selected_features_history = []
        self.feature_importance_history = []
        self.timestamps = []

    def update(self,
               selected_features: np.ndarray,
               feature_importance: Optional[np.ndarray] = None,
               timestamp: Optional[int] = None):
        """
        Update with new feature selection.

        Args:
            selected_features: Binary mask of selected features
            feature_importance: Importance scores (optional)
            timestamp: Current timestamp
        """
        self.selected_features_history.append(selected_features.copy())

        if feature_importance is not None:
            self.feature_importance_history.append(feature_importance.copy())

        if timestamp is not None:
            self.timestamps.append(timestamp)
        else:
            self.timestamps.append(len(self.timestamps))

    def compute_stability(self, window_size: int = 10) -> Dict[str, float]:
        """
        Compute feature selection stability metrics.

        Args:
            window_size: Window for computing stability

        Returns:
            stability_metrics: Dictionary of stability measures
        """
        if len(self.selected_features_history) < 2:
            return {'jaccard_similarity': 1.0, 'hamming_distance': 0.0, 'stability_score': 1.0}

        similarities = []
        hamming_distances = []

        # Compute pairwise similarities in sliding window
        for i in range(len(self.selected_features_history) - 1):
            current = self.selected_features_history[i]
            next_selection = self.selected_features_history[i + 1]

            # Jaccard similarity
            intersection = np.sum(current & next_selection)
            union = np.sum(current | next_selection)
            jaccard = intersection / union if union > 0 else 1.0
            similarities.append(jaccard)

            # Hamming distance
            hamming = np.sum(current != next_selection) / len(current)
            hamming_distances.append(hamming)

        return {
            'mean_jaccard_similarity': float(np.mean(similarities)),
            'std_jaccard_similarity': float(np.std(similarities)),
            'mean_hamming_distance': float(np.mean(hamming_distances)),
            'std_hamming_distance': float(np.std(hamming_distances)),
            'stability_score': float(np.mean(similarities))  # Overall stability
        }

    def compute_consistency(self, ground_truth_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute consistency with ground truth feature importance.

        Args:
            ground_truth_features: Known important features

        Returns:
            consistency_metrics: Agreement with ground truth
        """
        if ground_truth_features is None or len(self.selected_features_history) == 0:
            return {'agreement': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Compute agreement with ground truth over time
        agreements = []
        precisions = []
        recalls = []

        for selected in self.selected_features_history:
            # Convert to binary if needed
            if selected.dtype != bool:
                selected = selected > 0.5

            # Compute metrics
            intersection = np.sum(selected & ground_truth_features)
            selected_count = np.sum(selected)
            true_count = np.sum(ground_truth_features)

            precision = intersection / selected_count if selected_count > 0 else 0.0
            recall = intersection / true_count if true_count > 0 else 0.0
            agreement = intersection / np.sum(selected | ground_truth_features) if np.sum(selected | ground_truth_features) > 0 else 0.0

            agreements.append(agreement)
            precisions.append(precision)
            recalls.append(recall)

        return {
            'mean_agreement': float(np.mean(agreements)),
            'mean_precision': float(np.mean(precisions)),
            'mean_recall': float(np.mean(recalls)),
            'std_agreement': float(np.std(agreements))
        }

    def compute_sparsity_metrics(self) -> Dict[str, float]:
        """Compute sparsity-related metrics."""
        if len(self.selected_features_history) == 0:
            return {'mean_sparsity': 0.0, 'sparsity_trend': 0.0}

        sparsity_ratios = []

        for selected in self.selected_features_history:
            if selected.dtype != bool:
                selected = selected > 0.5
            sparsity = 1.0 - (np.sum(selected) / len(selected))
            sparsity_ratios.append(sparsity)

        # Compute trend (increasing/decreasing sparsity)
        if len(sparsity_ratios) > 1:
            trend = np.polyfit(range(len(sparsity_ratios)), sparsity_ratios, 1)[0]
        else:
            trend = 0.0

        return {
            'mean_sparsity': float(np.mean(sparsity_ratios)),
            'std_sparsity': float(np.std(sparsity_ratios)),
            'sparsity_trend': float(trend),
            'min_sparsity': float(np.min(sparsity_ratios)),
            'max_sparsity': float(np.max(sparsity_ratios))
        }


class ComputationalMetrics:
    """
    Metrics for computational efficiency and resource usage.

    Tracks training time, memory usage, inference latency, and energy consumption.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset computational tracking."""
        self.training_times = []
        self.inference_times = []
        self.memory_usage = []
        self.parameter_updates = []
        self.timestamps = []

    def record_training_step(self,
                             training_time: float,
                             memory_usage: Optional[float] = None,
                             parameter_updates: Optional[int] = None):
        """Record training step metrics."""
        self.training_times.append(training_time)

        if memory_usage is not None:
            self.memory_usage.append(memory_usage)

        if parameter_updates is not None:
            self.parameter_updates.append(parameter_updates)

        self.timestamps.append(len(self.timestamps))

    def record_inference_step(self, inference_time: float):
        """Record inference step metrics."""
        self.inference_times.append(inference_time)

    def compute_efficiency_metrics(self) -> Dict[str, float]:
        """Compute computational efficiency metrics."""
        metrics = {}

        if self.training_times:
            metrics.update({
                'mean_training_time': float(np.mean(self.training_times)),
                'std_training_time': float(np.std(self.training_times)),
                'total_training_time': float(np.sum(self.training_times)),
                'max_training_time': float(np.max(self.training_times))
            })

        if self.inference_times:
            metrics.update({
                'mean_inference_time': float(np.mean(self.inference_times)),
                'std_inference_time': float(np.std(self.inference_times)),
                'total_inference_time': float(np.sum(self.inference_times)),
                'max_inference_time': float(np.max(self.inference_times)),
                'throughput': float(len(self.inference_times) / np.sum(self.inference_times)) if np.sum(self.inference_times) > 0 else 0.0
            })

        if self.memory_usage:
            metrics.update({
                'mean_memory_usage': float(np.mean(self.memory_usage)),
                'max_memory_usage': float(np.max(self.memory_usage)),
                'memory_efficiency': float(np.std(self.memory_usage) / np.mean(self.memory_usage)) if np.mean(self.memory_usage) > 0 else 0.0
            })

        return metrics


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that combines all metrics for complete assessment.

    Provides unified interface for evaluating Stream-DAD performance across
    all dimensions: anomaly detection, drift adaptation, feature selection,
    and computational efficiency.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Initialize metric calculators
        self.ad_metrics = AnomalyDetectionMetrics()
        self.drift_metrics = DriftAdaptationMetrics()
        self.feature_metrics = FeatureSelectionMetrics()
        self.compute_metrics = ComputationalMetrics()

        # Evaluation history
        self.evaluation_history = []

    def update(self,
               predictions: np.ndarray,
               labels: np.ndarray,
               scores: np.ndarray,
               selected_features: Optional[np.ndarray] = None,
               drift_magnitude: Optional[float] = None,
               training_time: Optional[float] = None,
               inference_time: Optional[float] = None,
               timestamp: Optional[int] = None):
        """
        Update all metrics with new observations.

        Args:
            predictions: Binary predictions
            labels: True labels
            scores: Anomaly scores
            selected_features: Selected feature mask
            drift_magnitude: Current drift magnitude
            training_time: Time for training step
            inference_time: Time for inference
            timestamp: Current timestamp
        """
        # Update anomaly detection metrics
        self.ad_metrics.update(predictions, labels, scores, timestamp)

        # Update feature selection metrics
        if selected_features is not None:
            self.feature_metrics.update(selected_features, timestamp=timestamp)

        # Update computational metrics
        if training_time is not None:
            self.compute_metrics.record_training_step(training_time)

        if inference_time is not None:
            self.compute_metrics.record_inference_step(inference_time)

    def evaluate_comprehensive(self,
                               ground_truth_features: Optional[np.ndarray] = None,
                               true_drift_points: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation across all metrics.

        Args:
            ground_truth_features: Known important features
            true_drift_points: Ground truth drift timestamps

        Returns:
            comprehensive_results: Complete evaluation results
        """
        results = {
            'timestamp': len(self.evaluation_history),
            'anomaly_detection': self.ad_metrics.compute_metrics(),
            'feature_selection': {
                'stability': self.feature_metrics.compute_stability(),
                'sparsity': self.feature_metrics.compute_sparsity_metrics()
            },
            'computational': self.compute_metrics.compute_efficiency_metrics()
        }

        # Add consistency metrics if ground truth available
        if ground_truth_features is not None:
            results['feature_selection']['consistency'] = self.feature_metrics.compute_consistency(ground_truth_features)

        # Add drift adaptation metrics
        results['drift_adaptation'] = {
            'adaptation_latency': self.drift_metrics.compute_adaptation_latency(),
            'forgetting_ratio': self.drift_metrics.compute_forgetting_ratio()
        }

        # Add drift detection accuracy if available
        if true_drift_points is not None and hasattr(self.drift_metrics, 'detected_drift_points'):
            results['drift_adaptation']['detection_accuracy'] = self.drift_metrics.compute_drift_detection_accuracy(
                true_drift_points, self.drift_metrics.detected_drift_points
            )

        # Store evaluation
        self.evaluation_history.append(results)

        return results

    def get_temporal_analysis(self, window_size: int = 100) -> Dict[str, Any]:
        """Get temporal analysis of all metrics."""
        return {
            'anomaly_detection': self.ad_metrics.compute_temporal_metrics(window_size),
            'computational_trends': self._compute_computational_trends()
        }

    def _compute_computational_trends(self) -> Dict[str, List[float]]:
        """Compute trends in computational metrics."""
        if len(self.compute_metrics.training_times) < 2:
            return {}

        # Compute moving averages
        window_size = min(50, len(self.compute_metrics.training_times) // 4)

        if window_size < 2:
            return {}

        training_trend = []
        for i in range(window_size, len(self.compute_metrics.training_times)):
            window_avg = np.mean(self.compute_metrics.training_times[i-window_size:i])
            training_trend.append(window_avg)

        return {
            'training_time_trend': training_trend,
            'timestamps': list(range(window_size, len(self.compute_metrics.training_times)))
        }

    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_history:
            return "No evaluation data available."

        latest = self.evaluation_history[-1]

        report = "=== Stream-DAD Evaluation Report ===\n\n"

        # Anomaly Detection Performance
        ad_metrics = latest['anomaly_detection']
        report += "Anomaly Detection Performance:\n"
        report += f"  F1 Score: {ad_metrics.get('f1_score', 0):.4f}\n"
        report += f"  AUROC: {ad_metrics.get('auroc', 0):.4f}\n"
        report += f"  AUPRC: {ad_metrics.get('auprc', 0):.4f}\n"
        report += f"  False Alarm Rate: {ad_metrics.get('false_alarm_rate', 0):.4f}\n"
        report += f"  Detection Delay: {ad_metrics.get('detection_delay', 0):.2f}\n\n"

        # Feature Selection Performance
        if 'feature_selection' in latest:
            fs_metrics = latest['feature_selection']
            report += "Feature Selection Performance:\n"

            if 'stability' in fs_metrics:
                stability = fs_metrics['stability']
                report += f"  Stability Score: {stability.get('stability_score', 0):.4f}\n"
                report += f"  Mean Jaccard Similarity: {stability.get('mean_jaccard_similarity', 0):.4f}\n"

            if 'sparsity' in fs_metrics:
                sparsity = fs_metrics['sparsity']
                report += f"  Mean Sparsity: {sparsity.get('mean_sparsity', 0):.4f}\n"

            if 'consistency' in fs_metrics:
                consistency = fs_metrics['consistency']
                report += f"  Ground Truth Agreement: {consistency.get('mean_agreement', 0):.4f}\n"

        report += "\n"

        # Computational Performance
        if 'computational' in latest:
            comp_metrics = latest['computational']
            report += "Computational Performance:\n"
            report += f"  Mean Training Time: {comp_metrics.get('mean_training_time', 0):.4f}s\n"
            report += f"  Mean Inference Time: {comp_metrics.get('mean_inference_time', 0):.4f}s\n"
            report += f"  Throughput: {comp_metrics.get('throughput', 0):.2f} samples/s\n"
            if 'max_memory_usage' in comp_metrics:
                report += f"  Peak Memory Usage: {comp_metrics['max_memory_usage']:.2f} MB\n"

        report += "\n"

        # Drift Adaptation Performance
        if 'drift_adaptation' in latest:
            drift_metrics = latest['drift_adaptation']
            report += "Drift Adaptation Performance:\n"

            if 'adaptation_latency' in drift_metrics:
                latency = drift_metrics['adaptation_latency']
                if latency.get('mean_latency', 0) > 0:
                    report += f"  Mean Adaptation Latency: {latency['mean_latency']:.2f} steps\n"

            forgetting = drift_metrics.get('forgetting_ratio', 0)
            report += f"  Forgetting Ratio: {forgetting:.4f}\n"

        return report

    def reset_all(self):
        """Reset all metrics."""
        self.ad_metrics.reset()
        self.drift_metrics.reset()
        self.feature_metrics.reset()
        self.compute_metrics.reset()
        self.evaluation_history.clear()


def evaluate_anomaly_detection(model,
                               data_loader,
                               device: torch.device,
                               config: Dict = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation function for anomaly detection models.

    Args:
        model: Trained Stream-DAD model
        data_loader: Test data loader
        device: Device for computation
        config: Evaluation configuration

    Returns:
        results: Comprehensive evaluation results
    """
    config = config or {}
    evaluator = ComprehensiveEvaluator(config)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch['data'].to(device)
            labels = batch.get('labels', torch.zeros(data.shape[0], 1)).to(device)

            # Forward pass
            import time
            start_time = time.time()
            output = model(data, return_gates=True)
            inference_time = time.time() - start_time

            # Extract results
            anomaly_scores = output['anomaly_scores'].cpu().numpy()
            predictions = (anomaly_scores > np.percentile(anomaly_scores, 95)).astype(int)
            true_labels = labels.cpu().numpy().flatten()

            selected_features = None
            if 'gates' in output:
                gates = output['gates'].cpu().numpy()
                selected_features = (gates > 0.5).astype(int)

            # Update evaluator
            evaluator.update(
                predictions=predictions.flatten(),
                labels=true_labels,
                scores=anomaly_scores.flatten(),
                selected_features=selected_features[0] if selected_features is not None else None,
                inference_time=inference_time,
                timestamp=batch_idx
            )

    # Compute final results
    results = evaluator.evaluate_comprehensive()

    # Add temporal analysis
    results['temporal_analysis'] = evaluator.get_temporal_analysis()

    # Generate report
    results['report'] = evaluator.generate_report()

    return results
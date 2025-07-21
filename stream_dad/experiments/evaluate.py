"""
Evaluation Script for Stream-DAD.

This script provides comprehensive evaluation of trained Stream-DAD models
including performance metrics, drift analysis, and comparison with baselines.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

# Stream-DAD imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from stream_dad.core.models import StreamDAD, create_stream_dad_model
from stream_dad.utils.data_loading import create_data_loader, inject_synthetic_drift
from stream_dad.utils.evaluation import (
    ComprehensiveEvaluator,
    evaluate_anomaly_detection,
    AnomalyDetectionMetrics,
    DriftAdaptationMetrics,
    FeatureSelectionMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamDADEvaluator:
    """
    Comprehensive evaluator for Stream-DAD models.

    Provides detailed analysis including:
    - Performance metrics across different drift scenarios
    - Temporal analysis of adaptation behavior
    - Feature selection quality assessment
    - Computational efficiency analysis
    - Comparison with baseline methods
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Results storage
        self.results = {
            'model_performance': {},
            'temporal_analysis': {},
            'drift_analysis': {},
            'feature_analysis': {},
            'computational_analysis': {},
            'baseline_comparison': {}
        }

        # Evaluation metrics
        self.evaluator = ComprehensiveEvaluator(config.get('evaluation', {}))

    def load_model(self, checkpoint_path: str, input_dim: int) -> StreamDAD:
        """Load trained Stream-DAD model from checkpoint."""
        logger.info(f"Loading model from {checkpoint_path}")

        # Create model
        model = create_stream_dad_model(input_dim, self.config['model'])
        model.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load additional states
        if 'continual_learner_state' in checkpoint:
            model.continual_learner.load_state(checkpoint['continual_learner_state'])
        if 'drift_detector_state' in checkpoint:
            model.drift_detector.load_state(checkpoint['drift_detector_state'])

        model.eval()
        logger.info("Model loaded successfully")

        return model

    def evaluate_model_performance(self,
                                   model: StreamDAD,
                                   data_loader,
                                   dataset_name: str) -> Dict[str, Any]:
        """Evaluate basic model performance metrics."""
        logger.info(f"Evaluating model performance on {dataset_name}")

        model.eval()
        all_predictions = []
        all_labels = []
        all_scores = []
        all_gates = []
        inference_times = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                data = batch['data'].to(self.device)
                labels = batch.get('labels', torch.zeros(data.shape[0], 1)).to(self.device)

                # Measure inference time
                start_time = time.time()
                output = model(data, return_gates=True)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Extract results
                anomaly_scores = output['anomaly_scores'].cpu().numpy()
                gates = output['gates'].cpu().numpy()

                # Store results
                all_scores.extend(anomaly_scores.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_gates.append(gates)

                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx} batches")

        # Compute optimal threshold
        scores_array = np.array(all_scores)
        labels_array = np.array(all_labels)

        # Find optimal threshold using F1 score
        thresholds = np.linspace(scores_array.min(), scores_array.max(), 100)
        best_f1 = 0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            pred = (scores_array >= threshold).astype(int)
            tp = np.sum((pred == 1) & (labels_array == 1))
            fp = np.sum((pred == 1) & (labels_array == 0))
            fn = np.sum((pred == 0) & (labels_array == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Compute final predictions with optimal threshold
        final_predictions = (scores_array >= best_threshold).astype(int)

        # Compute comprehensive metrics

        performance = {
            'threshold': float(best_threshold),
            'f1_score': float(f1_score(labels_array, final_predictions, zero_division=0)),
            'precision': float(precision_score(labels_array, final_predictions, zero_division=0)),
            'recall': float(recall_score(labels_array, final_predictions, zero_division=0)),
            'false_alarm_rate': float(np.sum((final_predictions == 1) & (labels_array == 0)) / np.sum(labels_array == 0)) if np.sum(labels_array == 0) > 0 else 0.0,
            'mean_inference_time': float(np.mean(inference_times)),
            'std_inference_time': float(np.std(inference_times)),
            'throughput': float(len(all_scores) / np.sum(inference_times)) if np.sum(inference_times) > 0 else 0.0
        }

        # Add ROC metrics if both classes present
        if len(np.unique(labels_array)) > 1:
            performance.update({
                'auroc': float(roc_auc_score(labels_array, scores_array)),
                'auprc': float(average_precision_score(labels_array, scores_array))
            })

        # Confusion matrix
        if len(np.unique(labels_array)) > 1 and len(np.unique(final_predictions)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels_array, final_predictions).ravel()
            performance.update({
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            })

        self.results['model_performance'][dataset_name] = performance

        logger.info(f"Performance evaluation completed for {dataset_name}")
        logger.info(f"F1 Score: {performance['f1_score']:.4f}, "
                    f"AUROC: {performance.get('auroc', 0):.4f}, "
                    f"FAR: {performance['false_alarm_rate']:.4f}")

        return performance

    def evaluate_drift_adaptation(self,
                                  model: StreamDAD,
                                  data_loader,
                                  drift_points: Optional[List[int]] = None) -> Dict[str, Any]:
        """Evaluate drift adaptation capabilities."""
        logger.info("Evaluating drift adaptation performance")

        model.eval()
        drift_magnitudes = []
        performance_over_time = []
        adaptation_events = []

        # Sliding window for performance tracking
        window_size = self.config.get('evaluation', {}).get('adaptation_window', 100)
        recent_predictions = []
        recent_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                data = batch['data'].to(self.device)
                labels = batch.get('labels', torch.zeros(data.shape[0], 1)).to(self.device)

                # Forward pass
                output = model(data, return_gates=True)

                # Track drift magnitude
                drift_magnitude = model.drift_detector.get_current_drift_magnitude()
                drift_magnitudes.append(drift_magnitude)

                # Track predictions for sliding window performance
                anomaly_scores = output['anomaly_scores'].cpu().numpy()
                threshold = np.percentile(anomaly_scores, 95)  # Dynamic threshold
                predictions = (anomaly_scores > threshold).astype(int)

                recent_predictions.extend(predictions.flatten())
                recent_labels.extend(labels.cpu().numpy().flatten())

                # Maintain sliding window
                if len(recent_predictions) > window_size:
                    recent_predictions = recent_predictions[-window_size:]
                    recent_labels = recent_labels[-window_size:]

                # Compute performance in sliding window
                if len(recent_predictions) >= window_size and len(np.unique(recent_labels)) > 1:
                    window_f1 = f1_score(recent_labels, recent_predictions, zero_division=0)
                    performance_over_time.append({
                        'step': batch_idx,
                        'f1_score': float(window_f1),
                        'drift_magnitude': drift_magnitude
                    })

                # Detect adaptation events (high drift followed by recovery)
                adaptation_threshold = self.config.get('evaluation', {}).get('adaptation_threshold', 0.5)
                if drift_magnitude > adaptation_threshold:
                    adaptation_events.append({
                        'step': batch_idx,
                        'drift_magnitude': drift_magnitude,
                        'trigger': 'high_drift'
                    })

        # Analyze adaptation latency
        adaptation_analysis = self._analyze_adaptation_latency(
            performance_over_time, adaptation_events, drift_points
        )

        # Compute drift statistics
        drift_stats = {
            'mean_drift_magnitude': float(np.mean(drift_magnitudes)),
            'std_drift_magnitude': float(np.std(drift_magnitudes)),
            'max_drift_magnitude': float(np.max(drift_magnitudes)),
            'num_adaptation_events': len(adaptation_events),
            'adaptation_frequency': float(len(adaptation_events) / len(drift_magnitudes)) if len(drift_magnitudes) > 0 else 0.0
        }

        drift_results = {
            'drift_statistics': drift_stats,
            'adaptation_analysis': adaptation_analysis,
            'performance_over_time': performance_over_time,
            'adaptation_events': adaptation_events
        }

        self.results['drift_analysis'] = drift_results

        logger.info("Drift adaptation evaluation completed")
        logger.info(f"Mean drift magnitude: {drift_stats['mean_drift_magnitude']:.4f}")
        logger.info(f"Adaptation events: {drift_stats['num_adaptation_events']}")

        return drift_results

    def _analyze_adaptation_latency(self,
                                    performance_over_time: List[Dict],
                                    adaptation_events: List[Dict],
                                    drift_points: Optional[List[int]]) -> Dict[str, Any]:
        """Analyze adaptation latency after drift events."""
        if not performance_over_time or not adaptation_events:
            return {'adaptation_latencies': [], 'mean_latency': 0.0}

        adaptation_latencies = []
        recovery_threshold = 0.1  # Performance must recover within 10% of pre-drift level

        for event in adaptation_events:
            event_step = event['step']

            # Find pre-drift performance (look backward)
            pre_drift_performance = None
            for i in range(len(performance_over_time) - 1, -1, -1):
                if performance_over_time[i]['step'] < event_step - 50:  # At least 50 steps before
                    pre_drift_performance = performance_over_time[i]['f1_score']
                    break

            if pre_drift_performance is None:
                continue

            # Find recovery point (look forward)
            recovery_target = pre_drift_performance - recovery_threshold
            recovery_step = None

            for i, perf in enumerate(performance_over_time):
                if (perf['step'] > event_step and
                        perf['f1_score'] >= recovery_target):
                    recovery_step = perf['step']
                    break

            if recovery_step is not None:
                latency = recovery_step - event_step
                adaptation_latencies.append(latency)

        return {
            'adaptation_latencies': adaptation_latencies,
            'mean_latency': float(np.mean(adaptation_latencies)) if adaptation_latencies else 0.0,
            'std_latency': float(np.std(adaptation_latencies)) if adaptation_latencies else 0.0,
            'max_latency': float(np.max(adaptation_latencies)) if adaptation_latencies else 0.0
        }

    def evaluate_feature_selection(self,
                                   model: StreamDAD,
                                   data_loader,
                                   ground_truth_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate dynamic feature selection quality."""
        logger.info("Evaluating feature selection performance")

        model.eval()
        all_gates = []
        gate_evolution = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                data = batch['data'].to(self.device)

                # Forward pass
                output = model(data, return_gates=True)
                gates = output['gates'].cpu().numpy()

                # Store gates
                all_gates.append(gates)

                # Track evolution (sample every 10 steps)
                if batch_idx % 10 == 0:
                    gate_evolution.append({
                        'step': batch_idx,
                        'gates': gates[0].copy(),  # First sample in batch
                        'num_selected': int(np.sum(gates[0] > 0.5)),
                        'sparsity': float(1.0 - np.sum(gates[0] > 0.5) / len(gates[0]))
                    })

        # Concatenate all gates
        gates_array = np.vstack(all_gates)  # [num_samples, num_features]

        # Compute stability metrics
        stability_metrics = self._compute_gate_stability(gates_array)

        # Compute sparsity metrics
        sparsity_metrics = self._compute_sparsity_metrics(gates_array)

        # Compute consistency with ground truth if available
        consistency_metrics = {}
        if ground_truth_features is not None:
            consistency_metrics = self._compute_feature_consistency(
                gates_array, ground_truth_features
            )

        feature_results = {
            'stability': stability_metrics,
            'sparsity': sparsity_metrics,
            'consistency': consistency_metrics,
            'evolution': gate_evolution
        }

        self.results['feature_analysis'] = feature_results

        logger.info("Feature selection evaluation completed")
        logger.info(f"Mean sparsity: {sparsity_metrics['mean_sparsity']:.4f}")
        logger.info(f"Gate stability: {stability_metrics['mean_jaccard_similarity']:.4f}")

        return feature_results

    def _compute_gate_stability(self, gates_array: np.ndarray) -> Dict[str, float]:
        """Compute stability of gate selections over time."""
        binary_gates = (gates_array > 0.5).astype(int)

        # Compute pairwise Jaccard similarities
        similarities = []
        for i in range(len(binary_gates) - 1):
            gate1 = binary_gates[i]
            gate2 = binary_gates[i + 1]

            intersection = np.sum(gate1 & gate2)
            union = np.sum(gate1 | gate2)
            jaccard = intersection / union if union > 0 else 1.0
            similarities.append(jaccard)

        # Compute Hamming distances
        hamming_distances = []
        for i in range(len(binary_gates) - 1):
            hamming = np.sum(binary_gates[i] != binary_gates[i + 1]) / len(binary_gates[i])
            hamming_distances.append(hamming)

        return {
            'mean_jaccard_similarity': float(np.mean(similarities)),
            'std_jaccard_similarity': float(np.std(similarities)),
            'mean_hamming_distance': float(np.mean(hamming_distances)),
            'std_hamming_distance': float(np.std(hamming_distances))
        }

    def _compute_sparsity_metrics(self, gates_array: np.ndarray) -> Dict[str, float]:
        """Compute sparsity-related metrics."""
        binary_gates = (gates_array > 0.5).astype(int)
        sparsity_ratios = 1.0 - (np.sum(binary_gates, axis=1) / binary_gates.shape[1])

        # Compute trend
        if len(sparsity_ratios) > 1:
            trend = np.polyfit(range(len(sparsity_ratios)), sparsity_ratios, 1)[0]
        else:
            trend = 0.0

        return {
            'mean_sparsity': float(np.mean(sparsity_ratios)),
            'std_sparsity': float(np.std(sparsity_ratios)),
            'min_sparsity': float(np.min(sparsity_ratios)),
            'max_sparsity': float(np.max(sparsity_ratios)),
            'sparsity_trend': float(trend)
        }

    def _compute_feature_consistency(self,
                                     gates_array: np.ndarray,
                                     ground_truth_features: np.ndarray) -> Dict[str, float]:
        """Compute consistency with ground truth features."""
        binary_gates = (gates_array > 0.5).astype(int)
        ground_truth_binary = (ground_truth_features > 0.5).astype(int)

        # Compute metrics for each time step
        precisions = []
        recalls = []
        f1_scores = []

        for gates in binary_gates:
            tp = np.sum(gates & ground_truth_binary)
            fp = np.sum(gates & (~ground_truth_binary))
            fn = np.sum((~gates) & ground_truth_binary)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            'mean_precision': float(np.mean(precisions)),
            'mean_recall': float(np.mean(recalls)),
            'mean_f1_score': float(np.mean(f1_scores)),
            'std_precision': float(np.std(precisions)),
            'std_recall': float(np.std(recalls)),
            'std_f1_score': float(np.std(f1_scores))
        }

    def evaluate_computational_efficiency(self,
                                          model: StreamDAD,
                                          data_loader,
                                          num_runs: int = 100) -> Dict[str, Any]:
        """Evaluate computational efficiency metrics."""
        logger.info("Evaluating computational efficiency")

        model.eval()

        # Measure inference times
        inference_times = []
        memory_usage = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_runs:
                    break

                data = batch['data'].to(self.device)

                # Measure inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output = model(data, return_gates=True)
                    torch.cuda.synchronize()
                    end_time = time.time()
                else:
                    start_time = time.time()
                    output = model(data, return_gates=True)
                    end_time = time.time()

                inference_time = end_time - start_time
                inference_times.append(inference_time)

                # Measure memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    memory_usage.append(memory_mb)

        # Compute statistics
        efficiency_metrics = {
            'mean_inference_time': float(np.mean(inference_times)),
            'std_inference_time': float(np.std(inference_times)),
            'min_inference_time': float(np.min(inference_times)),
            'max_inference_time': float(np.max(inference_times)),
            'median_inference_time': float(np.median(inference_times)),
            'throughput': float(1.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0.0
        }

        if memory_usage:
            efficiency_metrics.update({
                'mean_memory_usage_mb': float(np.mean(memory_usage)),
                'max_memory_usage_mb': float(np.max(memory_usage)),
                'std_memory_usage_mb': float(np.std(memory_usage))
            })

        # Model size analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        efficiency_metrics.update({
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(total_params * 4 / (1024 * 1024))  # Assuming float32
        })

        self.results['computational_analysis'] = efficiency_metrics

        logger.info("Computational efficiency evaluation completed")
        logger.info(f"Mean inference time: {efficiency_metrics['mean_inference_time']:.4f}s")
        logger.info(f"Throughput: {efficiency_metrics['throughput']:.2f} samples/s")
        logger.info(f"Model size: {efficiency_metrics['model_size_mb']:.2f} MB")

        return efficiency_metrics

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        report_lines = ["=" * 60]
        report_lines.append("STREAM-DAD COMPREHENSIVE EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Model Performance Summary
        if 'model_performance' in self.results:
            report_lines.append("📊 MODEL PERFORMANCE SUMMARY")
            report_lines.append("-" * 40)

            for dataset, metrics in self.results['model_performance'].items():
                report_lines.append(f"\nDataset: {dataset}")
                report_lines.append(f"  F1 Score:           {metrics.get('f1_score', 0):.4f}")
                report_lines.append(f"  AUROC:              {metrics.get('auroc', 0):.4f}")
                report_lines.append(f"  AUPRC:              {metrics.get('auprc', 0):.4f}")
                report_lines.append(f"  False Alarm Rate:   {metrics.get('false_alarm_rate', 0):.4f}")
                report_lines.append(f"  Precision:          {metrics.get('precision', 0):.4f}")
                report_lines.append(f"  Recall:             {metrics.get('recall', 0):.4f}")
                report_lines.append(f"  Specificity:        {metrics.get('specificity', 0):.4f}")

        # Drift Adaptation Analysis
        if 'drift_analysis' in self.results:
            drift_stats = self.results['drift_analysis'].get('drift_statistics', {})
            adaptation_analysis = self.results['drift_analysis'].get('adaptation_analysis', {})

            report_lines.append("\n🌊 DRIFT ADAPTATION ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Mean Drift Magnitude:     {drift_stats.get('mean_drift_magnitude', 0):.4f}")
            report_lines.append(f"Max Drift Magnitude:      {drift_stats.get('max_drift_magnitude', 0):.4f}")
            report_lines.append(f"Adaptation Events:        {drift_stats.get('num_adaptation_events', 0)}")
            report_lines.append(f"Mean Adaptation Latency:  {adaptation_analysis.get('mean_latency', 0):.2f} steps")
            report_lines.append(f"Max Adaptation Latency:   {adaptation_analysis.get('max_latency', 0):.2f} steps")

        # Feature Selection Analysis
        if 'feature_analysis' in self.results:
            stability = self.results['feature_analysis'].get('stability', {})
            sparsity = self.results['feature_analysis'].get('sparsity', {})
            consistency = self.results['feature_analysis'].get('consistency', {})

            report_lines.append("\n🎯 FEATURE SELECTION ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Gate Stability (Jaccard):  {stability.get('mean_jaccard_similarity', 0):.4f}")
            report_lines.append(f"Mean Sparsity:             {sparsity.get('mean_sparsity', 0):.4f}")
            report_lines.append(f"Sparsity Std:              {sparsity.get('std_sparsity', 0):.4f}")

            if consistency:
                report_lines.append(f"GT Consistency (F1):      {consistency.get('mean_f1_score', 0):.4f}")
                report_lines.append(f"GT Precision:              {consistency.get('mean_precision', 0):.4f}")
                report_lines.append(f"GT Recall:                 {consistency.get('mean_recall', 0):.4f}")

        # Computational Efficiency
        if 'computational_analysis' in self.results:
            comp_metrics = self.results['computational_analysis']

            report_lines.append("\n⚡ COMPUTATIONAL EFFICIENCY")
            report_lines.append("-" * 40)
            report_lines.append(f"Mean Inference Time:      {comp_metrics.get('mean_inference_time', 0):.4f}s")
            report_lines.append(f"Throughput:               {comp_metrics.get('throughput', 0):.2f} samples/s")
            report_lines.append(f"Model Size:               {comp_metrics.get('model_size_mb', 0):.2f} MB")
            report_lines.append(f"Total Parameters:         {comp_metrics.get('total_parameters', 0):,}")

            if 'mean_memory_usage_mb' in comp_metrics:
                report_lines.append(f"Mean Memory Usage:        {comp_metrics['mean_memory_usage_mb']:.2f} MB")
                report_lines.append(f"Peak Memory Usage:        {comp_metrics['max_memory_usage_mb']:.2f} MB")

        report_lines.append("\n" + "=" * 60)

        report = "\n".join(report_lines)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")

        return report

    def save_results(self, save_path: str):
        """Save all evaluation results to JSON file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {save_path}")

    def plot_results(self, save_dir: str):
        """Generate and save visualization plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Performance over time
        if 'drift_analysis' in self.results and 'performance_over_time' in self.results['drift_analysis']:
            self._plot_performance_over_time(save_dir / 'performance_over_time.png')

        # Plot 2: Feature evolution
        if 'feature_analysis' in self.results and 'evolution' in self.results['feature_analysis']:
            self._plot_feature_evolution(save_dir / 'feature_evolution.png')

        # Plot 3: Drift magnitude over time
        if 'drift_analysis' in self.results:
            self._plot_drift_analysis(save_dir / 'drift_analysis.png')

        logger.info(f"Plots saved to {save_dir}")

    def _plot_performance_over_time(self, save_path: Path):
        """Plot performance metrics over time."""
        performance_data = self.results['drift_analysis']['performance_over_time']

        if not performance_data:
            return
"""
Ablation Studies for Stream-DAD.

This script systematically evaluates the contribution of each component
in Stream-DAD to understand their individual and combined effects on performance.
"""

import torch
import torch.nn as nn
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
from itertools import product

# Stream-DAD imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from stream_dad.core.models import StreamDAD, create_stream_dad_model
from stream_dad.utils.evaluation import ComprehensiveEvaluator
from stream_dad.experiments.train import StreamDADTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Comprehensive ablation study framework for Stream-DAD.

    Tests various component combinations to understand their individual
    and synergistic contributions to overall performance.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Ablation configurations
        self.ablation_configs = self._generate_ablation_configs()

        # Results storage
        self.results = {}

    def _generate_ablation_configs(self) -> Dict[str, Dict]:
        """Generate different ablation configurations."""

        base_config = self.config['model'].copy()

        ablations = {
            'full_model': {
                'name': 'Full Stream-DAD',
                'description': 'Complete model with all components',
                'config': base_config.copy()
            },

            'no_gating': {
                'name': 'No Dynamic Gating',
                'description': 'Static feature selection (all features used)',
                'config': {**base_config, 'disable_gating': True}
            },

            'no_ewc': {
                'name': 'No EWC Regularization',
                'description': 'No elastic weight consolidation',
                'config': {**base_config, 'lambda_ewc': 0.0}
            },

            'no_consistency': {
                'name': 'No Consistency Regularization',
                'description': 'No gate consistency penalty',
                'config': {**base_config, 'lambda_cons': 0.0}
            },

            'no_sparsity': {
                'name': 'No Sparsity Regularization',
                'description': 'No sparsity penalty on gates',
                'config': {**base_config, 'lambda_sparsity': 0.0, 'lambda_entropy': 0.0}
            },

            'no_drift_buffer': {
                'name': 'No Drift Buffer',
                'description': 'No memory buffer for high-drift samples',
                'config': {**base_config, 'continual': {**base_config['continual'], 'buffer': {'capacity': 0}}}
            },

            'no_continual_learning': {
                'name': 'No Continual Learning',
                'description': 'No EWC and no drift buffer',
                'config': {
                    **base_config,
                    'lambda_ewc': 0.0,
                    'continual': {**base_config['continual'], 'buffer': {'capacity': 0}}
                }
            },

            'static_baseline': {
                'name': 'Static Baseline',
                'description': 'No gating, no continual learning, no drift adaptation',
                'config': {
                    **base_config,
                    'disable_gating': True,
                    'lambda_ewc': 0.0,
                    'lambda_cons': 0.0,
                    'continual': {**base_config['continual'], 'buffer': {'capacity': 0}}
                }
            },

            'gating_only': {
                'name': 'Gating Only',
                'description': 'Only dynamic gating, no continual learning',
                'config': {
                    **base_config,
                    'lambda_ewc': 0.0,
                    'continual': {**base_config['continual'], 'buffer': {'capacity': 0}}
                }
            },

            'continual_only': {
                'name': 'Continual Learning Only',
                'description': 'Only continual learning, no dynamic gating',
                'config': {**base_config, 'disable_gating': True}
            }
        }

        return ablations

    def run_ablation_study(self,
                           train_loader,
                           val_loader,
                           dataset_name: str) -> Dict[str, Any]:
        """Run complete ablation study."""
        logger.info(f"Starting ablation study on {dataset_name}")
        logger.info(f"Testing {len(self.ablation_configs)} configurations")

        all_results = {}

        for ablation_name, ablation_info in self.ablation_configs.items():
            logger.info(f"Running ablation: {ablation_info['name']}")
            logger.info(f"Description: {ablation_info['description']}")

            # Update config for this ablation
            modified_config = self.config.copy()
            modified_config['model'] = ablation_info['config']

            # Run training and evaluation for this configuration
            try:
                result = self._run_single_ablation(
                    modified_config, train_loader, val_loader, ablation_name
                )
                all_results[ablation_name] = {
                    'info': ablation_info,
                    'results': result
                }

                # Log key metrics
                if 'final_metrics' in result:
                    metrics = result['final_metrics']
                    logger.info(f"Results - F1: {metrics.get('f1_score', 0):.4f}, "
                                f"AUROC: {metrics.get('auroc', 0):.4f}, "
                                f"FAR: {metrics.get('false_alarm_rate', 0):.4f}")

            except Exception as e:
                logger.error(f"Ablation {ablation_name} failed: {e}")
                all_results[ablation_name] = {
                    'info': ablation_info,
                    'results': {'error': str(e)}
                }

        self.results = all_results
        logger.info("Ablation study completed")

        return all_results

    def _run_single_ablation(self,
                             config: Dict,
                             train_loader,
                             val_loader,
                             ablation_name: str) -> Dict[str, Any]:
        """Run training and evaluation for a single ablation configuration."""

        # Create modified model based on ablation config
        input_dim = config['data'].get('input_dim', 50)  # Default or from data
        model = self._create_ablated_model(input_dim, config['model'])
        model.to(self.device)

        # Setup trainer with modified config
        trainer_config = config.copy()
        trainer_config['experiment'] = {'save_dir': f'./ablation_temp/{ablation_name}'}
        trainer_config['training']['max_steps'] = config.get('ablation', {}).get('max_steps', 1000)
        trainer_config['training']['eval_interval'] = config.get('ablation', {}).get('eval_interval', 200)

        trainer = StreamDADTrainer(trainer_config)
        trainer.model = model
        trainer.setup_model(input_dim)  # This will setup optimizer etc.

        # Train model
        start_time = time.time()
        final_results = trainer.train_streaming(train_loader, val_loader)
        training_time = time.time() - start_time

        # Additional evaluation metrics
        evaluator = ComprehensiveEvaluator(config.get('evaluation', {}))

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 100:  # Limit for ablation study
                    break

                data = batch['data'].to(self.device)
                labels = batch.get('labels', torch.zeros(data.shape[0], 1)).to(self.device)

                output = model(data, return_gates=True)

                # Extract results
                anomaly_scores = output['anomaly_scores'].cpu().numpy()
                threshold = np.percentile(anomaly_scores, 95)
                predictions = (anomaly_scores > threshold).astype(int)

                selected_features = None
                if 'gates' in output:
                    gates = output['gates'].cpu().numpy()
                    selected_features = (gates > 0.5).astype(int)

                evaluator.update(
                    predictions=predictions.flatten(),
                    labels=labels.cpu().numpy().flatten(),
                    scores=anomaly_scores.flatten(),
                    selected_features=selected_features[0] if selected_features is not None else None,
                    timestamp=batch_idx
                )

        comprehensive_results = evaluator.evaluate_comprehensive()

        return {
            'final_metrics': final_results.get('anomaly_detection', {}) if final_results else {},
            'comprehensive_results': comprehensive_results,
            'training_time': training_time,
            'training_steps': trainer.current_step
        }

    def _create_ablated_model(self, input_dim: int, model_config: Dict) -> StreamDAD:
        """Create a model with specific components disabled based on ablation config."""

        # Create base model
        model = create_stream_dad_model(input_dim, model_config)

        # Apply ablations by modifying the model
        if model_config.get('disable_gating', False):
            # Replace gating network with identity (all gates = 1)
            model.gating_network = IdentityGating(input_dim)

        return model

    def analyze_component_contributions(self) -> Dict[str, Any]:
        """Analyze the contribution of each component."""
        if not self.results:
            logger.warning("No results available for analysis")
            return {}

        # Extract key metrics from all ablations
        metrics_data = []

        for ablation_name, ablation_data in self.results.items():
            if 'error' in ablation_data['results']:
                continue

            results = ablation_data['results']
            final_metrics = results.get('final_metrics', {})

            metrics_data.append({
                'ablation': ablation_name,
                'name': ablation_data['info']['name'],
                'f1_score': final_metrics.get('f1_score', 0),
                'auroc': final_metrics.get('auroc', 0),
                'false_alarm_rate': final_metrics.get('false_alarm_rate', 1),
                'precision': final_metrics.get('precision', 0),
                'recall': final_metrics.get('recall', 0),
                'training_time': results.get('training_time', 0)
            })

        df = pd.DataFrame(metrics_data)

        if df.empty:
            return {'error': 'No valid results for analysis'}

        # Calculate relative performance compared to full model
        full_model_metrics = df[df['ablation'] == 'full_model']

        if full_model_metrics.empty:
            logger.warning("Full model results not found")
            baseline_f1 = df['f1_score'].max()
            baseline_auroc = df['auroc'].max()
        else:
            baseline_f1 = full_model_metrics['f1_score'].iloc[0]
            baseline_auroc = full_model_metrics['auroc'].iloc[0]

        # Compute relative performance
        df['f1_relative'] = df['f1_score'] / baseline_f1 if baseline_f1 > 0 else 0
        df['auroc_relative'] = df['auroc'] / baseline_auroc if baseline_auroc > 0 else 0

        # Rank ablations by performance
        df = df.sort_values('f1_score', ascending=False)

        # Component contribution analysis
        component_contributions = self._calculate_component_contributions(df)

        analysis = {
            'metrics_summary': df.to_dict('records'),
            'component_contributions': component_contributions,
            'performance_ranking': df[['ablation', 'name', 'f1_score', 'auroc', 'false_alarm_rate']].to_dict('records'),
            'best_ablation': df.iloc[0].to_dict(),
            'worst_ablation': df.iloc[-1].to_dict()
        }

        return analysis

    def _calculate_component_contributions(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate individual component contributions to performance."""
        contributions = {}

        # Get baseline (full model) performance
        full_model = df[df['ablation'] == 'full_model']
        if full_model.empty:
            return contributions

        baseline_f1 = full_model['f1_score'].iloc[0]

        # Calculate contribution of each component
        component_ablations = {
            'dynamic_gating': 'no_gating',
            'ewc_regularization': 'no_ewc',
            'consistency_regularization': 'no_consistency',
            'sparsity_regularization': 'no_sparsity',
            'drift_buffer': 'no_drift_buffer'
        }

        for component, ablation_key in component_ablations.items():
            ablation_result = df[df['ablation'] == ablation_key]

            if not ablation_result.empty:
                ablation_f1 = ablation_result['f1_score'].iloc[0]
                # Contribution = performance drop when component is removed
                contribution = baseline_f1 - ablation_f1
                contribution_percent = (contribution / baseline_f1 * 100) if baseline_f1 > 0 else 0

                contributions[component] = {
                    'absolute_contribution': float(contribution),
                    'relative_contribution_percent': float(contribution_percent),
                    'performance_with_component': float(baseline_f1),
                    'performance_without_component': float(ablation_f1)
                }

        return contributions

    def generate_ablation_report(self) -> str:
        """Generate comprehensive ablation study report."""
        if not self.results:
            return "No ablation results available."

        analysis = self.analyze_component_contributions()

        if 'error' in analysis:
            return f"Analysis failed: {analysis['error']}"

        report_lines = ["=" * 70]
        report_lines.append("STREAM-DAD ABLATION STUDY REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Performance ranking
        report_lines.append("📊 PERFORMANCE RANKING")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Rank':<4} {'Configuration':<25} {'F1 Score':<10} {'AUROC':<8} {'FAR':<8}")
        report_lines.append("-" * 50)

        for i, config in enumerate(analysis['performance_ranking'], 1):
            report_lines.append(
                f"{i:<4} {config['name'][:24]:<25} "
                f"{config['f1_score']:<10.4f} {config['auroc']:<8.4f} "
                f"{config['false_alarm_rate']:<8.4f}"
            )

        # Component contributions
        report_lines.append("\n🔧 COMPONENT CONTRIBUTIONS")
        report_lines.append("-" * 50)

        contributions = analysis['component_contributions']

        # Sort components by contribution
        sorted_components = sorted(
            contributions.items(),
            key=lambda x: x[1]['relative_contribution_percent'],
            reverse=True
        )

        for component, contrib_data in sorted_components:
            component_name = component.replace('_', ' ').title()
            report_lines.append(
                f"{component_name:<25}: "
                f"{contrib_data['relative_contribution_percent']:>6.1f}% "
                f"({contrib_data['absolute_contribution']:>+.4f})"
            )

        # Best vs Worst
        best = analysis['best_ablation']
        worst = analysis['worst_ablation']

        report_lines.append("\n📈 BEST vs WORST CONFIGURATION")
        report_lines.append("-" * 50)
        report_lines.append(f"Best:  {best['name']}")
        report_lines.append(f"       F1={best['f1_score']:.4f}, AUROC={best['auroc']:.4f}")
        report_lines.append(f"Worst: {worst['name']}")
        report_lines.append(f"       F1={worst['f1_score']:.4f}, AUROC={worst['auroc']:.4f}")
        report_lines.append(f"Performance Gap: {(best['f1_score'] - worst['f1_score']):.4f} F1 points")

        # Key insights
        report_lines.append("\n💡 KEY INSIGHTS")
        report_lines.append("-" * 50)

        # Find most important component
        # if sorted_components:
        #     most_important = sorted_components[0]
        #     report_lines.append(
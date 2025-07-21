"""
Stream-DAD: Continual Self-Supervised Learning via Dynamic Spatio-Temporal
Feature Selection for Concept Drift-Aware Multivariate Anomaly Detection

This package implements the Stream-DAD algorithm for anomaly detection in
non-stationary multivariate time series data.
"""

__version__ = "1.0.0"
__author__ = "Stream-DAD Authors"

from .core.models import StreamDAD, GRUEncoder, GRUDecoder
from .core.gating import DynamicGatingNetwork
from .core.continual import ContinualLearner
from .core.drift_detection import DriftDetector
from .utils.data_loading import StreamingDataLoader
from .utils.evaluation import evaluate_anomaly_detection

__all__ = [
    'StreamDAD',
    'GRUEncoder',
    'GRUDecoder',
    'DynamicGatingNetwork',
    'ContinualLearner',
    'DriftDetector',
    'StreamingDataLoader',
    'evaluate_anomaly_detection'
]
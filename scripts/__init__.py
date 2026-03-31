"""
Scripts package for hybrid sarcasm detection.

Contains training, evaluation, and XAI analysis scripts.
"""

from .train import train_model
from .evaluate import evaluate_model
from .xai import run_xai_analysis

__all__ = ['train_model', 'evaluate_model', 'run_xai_analysis']

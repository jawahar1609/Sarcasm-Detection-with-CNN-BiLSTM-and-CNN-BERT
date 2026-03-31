import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import models
import models.cnn_bilstm as cnn_bilstm
import models.cnn_bert as cnn_bert
import models.cnn as cnn
import models.bert as bert
import models.lstm as lstm

from preprocessing import SarcasmDataset, prepare_data

# Reuse utilities from train.py
from scripts.train import MODEL_MAP, _get_device


def _load_checkpoint(model_path, device):
    """Load model checkpoint and extract configuration."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]
    model_type_name = config["model_type"]
    params = config["params"]
    
    return vocab, model_type_name, params, checkpoint["model_state"]


def _build_model(model_type_name, vocab, params, device):
    """Build model from configuration."""
    # Handle both class names and lowercase names
    model_class = None
    for key, cls in MODEL_MAP.items():
        if key == model_type_name or cls.__name__ == model_type_name:
            model_class = cls
            break
    
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type_name}")
    
    vocab_size = len(vocab)
    pad_idx = vocab["<pad>"]
    
    model = model_class(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        **params
    ).to(device)
    
    return model


def _evaluate_loop(model, loader, device, model_type_name):
    """Run evaluation loop and collect predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids, attn_mask, labels = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass - handle BERT-based models
            if 'bert' in model_type_name.lower():
                logits = model(input_ids, attention_mask=attn_mask)
            else:
                logits = model(input_ids)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = None  # In case only one class is present
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'auc_roc': float(auc_roc) if auc_roc is not None else None,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'class_0': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1': float(f1_per_class[0]),
                'support': int(support[0]),
            },
            'class_1': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1': float(f1_per_class[1]),
                'support': int(support[1]),
            }
        }
    }
    
    return metrics


def print_metrics(metrics, model_name, dataset_name):
    """Print formatted evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"Evaluation Results: {model_name} on {dataset_name}")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"  Precision (wtd):    {metrics['precision_weighted']:.4f}")
    print(f"  Recall (wtd):       {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (wtd):     {metrics['f1_weighted']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Class 0 (Non-Sarcastic):")
    print(f"    Precision: {metrics['per_class_metrics']['class_0']['precision']:.4f}")
    print(f"    Recall:    {metrics['per_class_metrics']['class_0']['recall']:.4f}")
    print(f"    F1-Score:  {metrics['per_class_metrics']['class_0']['f1']:.4f}")
    print(f"    Support:   {metrics['per_class_metrics']['class_0']['support']}")
    
    print(f"  Class 1 (Sarcastic):")
    print(f"    Precision: {metrics['per_class_metrics']['class_1']['precision']:.4f}")
    print(f"    Recall:    {metrics['per_class_metrics']['class_1']['recall']:.4f}")
    print(f"    F1-Score:  {metrics['per_class_metrics']['class_1']['f1']:.4f}")
    print(f"    Support:   {metrics['per_class_metrics']['class_1']['support']}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                Predicted")
    print(f"                0      1")
    print(f"  Actual  0  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"          1  {cm[1][0]:5d}  {cm[1][1]:5d}")
    print(f"{'='*70}\n")


def save_metrics(metrics, model_name, dataset_name, output_dir):
    """Save evaluation metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'metrics': metrics
    }
    
    filename = f"{output_dir}/{model_name}_{dataset_name}_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def evaluate_model(
    model_path,
    dataset_name,
    test_df,
    output_dir="results/metrics",
    batch_size=64,
    max_len=40,
    print_results=True
):
    # Get device
    device = _get_device()
    print(f"Using device: {device}")
    
    # Load checkpoint
    vocab, model_type_name, params, model_state = _load_checkpoint(model_path, device)
    
    # Build model
    model = _build_model(model_type_name, vocab, params, device)
    model.load_state_dict(model_state)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_type_name} with {num_params:,} parameters")
    
    # Create dataset and loader
    test_ds = SarcasmDataset(test_df, vocab, max_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    print(f"\nEvaluating {model_type_name}...")
    y_pred, y_prob, y_true = _evaluate_loop(model, test_loader, device, model_type_name)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Print results
    if print_results:
        print_metrics(metrics, model_type_name, dataset_name)
    
    # Save results
    save_metrics(metrics, model_type_name, dataset_name, output_dir)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate sarcasm detection models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--dataset', type=str, default='sarcasm_news',
                        help='Dataset name (sarcasm_news or sarc)')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Path to raw data. If provided, uses prepare_data()')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/metrics',
                        help='Directory to save results')
    parser.add_argument('--max_len', type=int, default=40,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Load test data
    if args.raw_data_path:
        print(f"\nPreparing data from raw source: {args.raw_data_path}")
        _, _, test_df, _ = prepare_data(
            dataset=args.dataset,
            raw_data_path=args.raw_data_path,
            save_to_disk=False
        )
        print(f"Test set size: {len(test_df)}")
    else:
        # Fallback to CSV loading
        data_dir = f"processed_data/{args.dataset}"
        test_path = f"{data_dir}/test.csv"
        print(f"\nLoading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"Test set size: {len(test_df)}")
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model_path,
        dataset_name=args.dataset,
        test_df=test_df,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        print_results=True
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

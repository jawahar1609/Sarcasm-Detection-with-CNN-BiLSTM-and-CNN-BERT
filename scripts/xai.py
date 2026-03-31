import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from preprocessing import SarcasmDataset

# Reuse utilities from other scripts
from scripts.train import _get_device
from scripts.evaluate import _load_checkpoint, _build_model


def _tokenize_text(text, vocab, max_len=40):
    tokens = text.lower().strip().split()[:max_len]
    pad_idx = vocab["<pad>"]
    unk_idx = vocab["<unk>"]
    
    ids = [vocab.get(tok, unk_idx) for tok in tokens]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    
    attention_mask = [1 if id != pad_idx else 0 for id in ids]
    
    return tokens, ids, attention_mask


def _compute_token_importance(model, vocab, text, device, model_type_name, max_len=40):
    tokens, ids, attention_mask = _tokenize_text(text, vocab, max_len)
    pad_idx = vocab["<pad>"]
    
    # Get baseline prediction
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    attn_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    with torch.no_grad():
        if 'bert' in model_type_name.lower():
            baseline_logit = model(input_ids, attention_mask=attn_mask)
        else:
            baseline_logit = model(input_ids)
        
        baseline_prob = torch.sigmoid(baseline_logit).item()
    
    # Compute importance by masking each token
    importances = []
    
    for i in range(len(tokens)):
        # Create masked version
        masked_ids = ids.copy()
        masked_ids[i] = pad_idx  # Mask this token
        
        masked_attention = attention_mask.copy()
        masked_attention[i] = 0
        
        input_ids_masked = torch.tensor([masked_ids], dtype=torch.long).to(device)
        attn_mask_masked = torch.tensor([masked_attention], dtype=torch.long).to(device)
        
        with torch.no_grad():
            if 'bert' in model_type_name.lower():
                masked_logit = model(input_ids_masked, attention_mask=attn_mask_masked)
            else:
                masked_logit = model(input_ids_masked)
            
            masked_prob = torch.sigmoid(masked_logit).item()
        
        # Importance = change in prediction when token is masked
        importance = abs(baseline_prob - masked_prob)
        importances.append(importance)
    
    return tokens, importances, baseline_prob


def _visualize_importance(tokens, importances, prediction, true_label, title, save_path, text=None):
    """Create and save token importance visualization."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Normalize importances for color scale
    max_importance = max(importances) if importances else 1
    norm_importances = [imp / max_importance for imp in importances]
    
    # Create color map
    colors = plt.cm.RdYlGn_r(norm_importances)
    
    # Plot bars
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, importances, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Importance Score (Change in Probability)', fontsize=12)
    
    pred_label = "Sarcastic" if prediction >= 0.5 else "Non-Sarcastic"
    true_label_text = "Sarcastic" if true_label == 1 else "Non-Sarcastic"
    ax.set_title(f'{title}\nPrediction: {prediction:.3f} ({pred_label}), True: ({true_label_text})', fontsize=14)
    ax.invert_yaxis()
    
    # Add original text if provided
    if text:
        import textwrap
        wrapped_text = textwrap.fill(text, width=100)
        plt.figtext(0.5, 0.02, f"Original Text:\n{wrapped_text}", ha="center", fontsize=10, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        plt.subplots_adjust(bottom=0.2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def _compare_models(model1, model2, vocab, text, device, model1_name, model2_name, 
                    max_len, output_dir, plot_idx=0):
    """Compare token importance between two models."""
    # Analyze both models
    tokens_1, imp_1, pred_1 = _compute_token_importance(
        model1, vocab, text, device, model1_name, max_len
    )
    
    tokens_2, imp_2, pred_2 = _compute_token_importance(
        model2, vocab, text, device, model2_name, max_len
    )
    
    # Ensure same tokens
    assert tokens_1 == tokens_2, "Token mismatch between models"
    
    # Compute difference in importance
    importance_diff = [abs(a - b) for a, b in zip(imp_1, imp_2)]
    
    # Visualize comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    y_pos = np.arange(len(tokens_1))
    
    # Model 1 importance
    ax1.barh(y_pos, imp_1, color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tokens_1)
    ax1.set_xlabel('Importance Score')
    ax1.set_title(f'{model1_name} Token Importance (Pred: {pred_1:.3f})')
    ax1.invert_yaxis()
    
    # Model 2 importance
    ax2.barh(y_pos, imp_2, color='coral', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tokens_2)
    ax2.set_xlabel('Importance Score')
    ax2.set_title(f'{model2_name} Token Importance (Pred: {pred_2:.3f})')
    ax2.invert_yaxis()
    
    # Difference
    colors = ['green' if imp_1[i] > imp_2[i] else 'red' for i in range(len(importance_diff))]
    ax3.barh(y_pos, importance_diff, color=colors, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(tokens_2)
    ax3.set_xlabel('Absolute Difference in Importance')
    ax3.set_title(f'Difference (Green: {model1_name} > {model2_name}, Red: {model2_name} > {model1_name})')
    ax3.invert_yaxis()
    
    # Add original text
    import textwrap
    wrapped_text = textwrap.fill(text, width=100)
    plt.figtext(0.5, 0.02, f"Original Text:\n{wrapped_text}", ha="center", fontsize=10, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    save_path = f"{output_dir}/sample_{plot_idx+1}_comparison_{model1_name}_vs_{model2_name}.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to {save_path}")
    return save_path


def run_xai_analysis(
    model_path,
    dataset_name,
    test_df,
    output_dir="results/xai",
    num_samples=10,
    max_len=40,
    model2_path=None
):
    # Get device
    device = _get_device()
    print(f"Using device: {device}")
    
    # Load checkpoint and build model
    vocab, model_type_name, params, model_state = _load_checkpoint(model_path, device)
    model = _build_model(model_type_name, vocab, params, device)
    model.load_state_dict(model_state)
    
    print(f"Loaded {model_type_name}")
    
    # Create dataset
    dataset = SarcasmDataset(test_df, vocab, max_len, return_extras=True)
    
    # Create output directory
    model_output_dir = f"{output_dir}/{model_type_name}_{dataset_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    visualization_paths = []
    
    # Select balanced samples (half sarcastic, half non-sarcastic)
    sarcastic_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    non_sarcastic_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    num_per_class = num_samples // 2
    selected_indices = (
        np.random.choice(sarcastic_indices, min(num_per_class, len(sarcastic_indices)), replace=False).tolist() +
        np.random.choice(non_sarcastic_indices, min(num_per_class, len(non_sarcastic_indices)), replace=False).tolist()
    )
    
    print(f"\nAnalyzing {len(selected_indices)} samples with {model_type_name}...")
    
    for idx, sample_idx in enumerate(selected_indices):
        text = dataset.texts[sample_idx]
        true_label = dataset.labels[sample_idx]
        
        print(f"\nSample {idx + 1}/{len(selected_indices)}")
        print(f"Text: {text[:100]}...")
        print(f"True label: {'Sarcastic' if true_label == 1 else 'Non-Sarcastic'}")
        
        tokens, importances, prediction = _compute_token_importance(
            model, vocab, text, device, model_type_name, max_len
        )
        
        print(f"Prediction: {prediction:.3f} ({'Sarcastic' if prediction >= 0.5 else 'Non-Sarcastic'})")
        print(f"Top 3 important tokens: {[tokens[i] for i in np.argsort(importances)[-3:][::-1]]}")
        
        save_path = f"{model_output_dir}/sample_{idx+1}_{model_type_name}_{'sarc' if true_label == 1 else 'nonsarc'}.png"
        viz_path = _visualize_importance(
            tokens, importances, prediction, true_label,
            f"Sample {idx + 1}: {model_type_name}", save_path, text
        )
        visualization_paths.append(viz_path)
        print(f"Saved visualization to {viz_path}")
    
    # Optional: Compare with second model if provided
    if model2_path:
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS")
        print("="*70)
        
        # Load second model
        vocab2, model2_type_name, params2, model2_state = _load_checkpoint(model2_path, device)
        model2 = _build_model(model2_type_name, vocab2, params2, device)
        model2.load_state_dict(model2_state)
        
        print(f"Loaded second model: {model2_type_name}")
        
        comparison_dir = f"{output_dir}/comparison_{model_type_name}_vs_{model2_type_name}"
        
        # Compare on a few samples
        comparison_samples = np.random.choice(selected_indices, min(5, len(selected_indices)), replace=False)
        
        for plot_idx, sample_idx in enumerate(comparison_samples):
            text = dataset.texts[sample_idx]
            print(f"\nComparing sample {plot_idx+1}/5: {text[:80]}...")
            
            viz_path = _compare_models(
                model, model2, vocab, text, device,
                model_type_name, model2_type_name, max_len,
                comparison_dir, plot_idx
            )
            visualization_paths.append(viz_path)
    
    print(f"\n{'='*70}")
    print(f"XAI Analysis complete! Generated {len(visualization_paths)} visualizations")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return visualization_paths


def main():
    parser = argparse.ArgumentParser(description='XAI analysis for sarcasm detection')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model2_path', type=str, default=None,
                        help='Path to second model for comparison (optional)')
    parser.add_argument('--dataset', type=str, default='sarcasm_news',
                        help='Dataset name (sarcasm_news or sarc)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='results/xai',
                        help='Output directory for visualizations')
    parser.add_argument('--max_len', type=int, default=40,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Load test data from CSV
    data_dir = f"processed_data/{args.dataset}"
    test_path = f"{data_dir}/test.csv"
    
    print(f"\nLoading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test set size: {len(test_df)}")
    
    # Run XAI analysis
    visualization_paths = run_xai_analysis(
        model_path=args.model_path,
        dataset_name=args.dataset,
        test_df=test_df,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_len=args.max_len,
        model2_path=args.model2_path
    )
    
    print(f"\nGenerated {len(visualization_paths)} visualizations")


if __name__ == "__main__":
    main()

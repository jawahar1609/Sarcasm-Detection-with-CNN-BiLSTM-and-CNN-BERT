import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import models.cnn_bilstm as cnn_bilstm
import models.cnn_bert as cnn_bert
import models.cnn as cnn 
import models.bert as bert
import models.lstm as lstm
import os

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from preprocessing import SarcasmDataset, prepare_data


# Constants - Map model names to classes
MODEL_MAP = {
    'cnn': cnn.CNN,
    'lstm': lstm.LSTM,
    'bert': bert.Bert,
    'cnn_bilstm': cnn_bilstm.HybridCNNBiLSTM,
    'cnn_bert': cnn_bert.HybridCNNBert,
}


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _get_model_params(model_name, embed_dim=128, lstm_hidden=128, 
                      bert_num_layers=12, bert_attn_heads=16, bert_intermediate_size=3072):
    model_type = MODEL_MAP[model_name]
    
    if model_type == cnn_bilstm.HybridCNNBiLSTM:
        return {
            "embed_dim": embed_dim,
            "lstm_hidden_dim": lstm_hidden,
        }
    elif model_type == cnn_bert.HybridCNNBert:
        return {
            "embed_dim": embed_dim,
            "bert_num_layers": bert_num_layers,
            "bert_attn_heads": bert_attn_heads,
            "bert_intermediate_size": bert_intermediate_size,
        }
    elif model_type == cnn.CNN:
        return {
            "embed_dim": embed_dim,
        }
    elif model_type == bert.Bert:
        return {
            "embed_dim": embed_dim,
            "bert_num_layers": bert_num_layers,
            "bert_attn_heads": bert_attn_heads,
            "bert_intermediate_size": bert_intermediate_size,
        }
    elif model_type == lstm.LSTM:
        return {
            "embed_dim": embed_dim,
            "lstm_hidden_dim": lstm_hidden,
        }
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def _setup_optimizer(model, model_name, lr=1e-3, bert_lr=2e-5):
    model_type = MODEL_MAP[model_name]
    
    if model_type is cnn_bert.HybridCNNBert:
        # Separate learning rates for BERT and other components
        bert_params = list(model.bert.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
        optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": bert_lr},
            {"params": other_params, "lr": lr},
        ])
    elif model_type is bert.Bert:
        optimizer = torch.optim.AdamW(model.parameters(), lr=bert_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    return optimizer


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids, attn_mask, labels = batch
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        if isinstance(model, cnn_bert.HybridCNNBert) or isinstance(model, bert.Bert):
            logits = model(input_ids, attention_mask=attn_mask)
        else:
            logits = model(input_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            input_ids, attn_mask, labels = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device).unsqueeze(1)

            if isinstance(model, cnn_bert.HybridCNNBert) or isinstance(model, bert.Bert):
                logits = model(input_ids, attention_mask=attn_mask)
            else:
                logits = model(input_ids)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_model(
    model_name,
    dataset_name,
    train_df,
    val_df,
    test_df,
    output_dir="results/models",
    epochs=10,
    batch_size=64,
    lr=1e-3,
    bert_lr=2e-5,
    early_stop_patience=3,
    max_len=40,
    min_freq=2,
    embed_dim=128,
    lstm_hidden=128,
    bert_num_layers=12,
    bert_attn_heads=16,
    bert_intermediate_size=3072,
    seed=42
):
    model_type = MODEL_MAP[model_name]
    print(f"Training model: {model_type.__name__}")
    
    # Set random seed
    set_seed(seed)
    
    # Get device
    device = _get_device()
    print(f"Using device: {device}")
    
    # Build vocab from training data
    from preprocessing.utils import build_vocab
    all_texts = train_df['text'].tolist() + val_df['text'].tolist() + test_df['text'].tolist()
    vocab, _ = build_vocab(all_texts, min_freq=min_freq)
    vocab_size = len(vocab)
    pad_idx = vocab["<pad>"]
    print(f"vocab size: {vocab_size}")
    
    # Create datasets and dataloaders
    train_ds = SarcasmDataset(train_df, vocab, max_len)
    val_ds = SarcasmDataset(val_df, vocab, max_len)
    test_ds = SarcasmDataset(test_df, vocab, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Get model-specific parameters
    params = _get_model_params(
        model_name, embed_dim, lstm_hidden, 
        bert_num_layers, bert_attn_heads, bert_intermediate_size
    )
    
    # Initialize model
    model = model_type(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        **params
    ).to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/{model_name}_{dataset_name}_model.pt"
    
    # Set up optimizer
    optimizer = _setup_optimizer(model, model_name, lr, bert_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nepoch {epoch}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        
        print(
            f"train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "model_type": model_name,
                        "params": params,
                    },
                },
                model_path,
            )
            print(f"  new best model saved (val acc = {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    print(f"\ntraining done. best val acc: {best_val_acc:.4f}")
    
    # Evaluate best checkpoint on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    print(f"test loss: {test_loss:.4f}, acc: {test_acc:.4f}")
    
    return model_path


def main():
    """Main function for CLI usage."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train sarcasm detection models')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['cnn', 'lstm', 'bert', 'cnn_bilstm', 'cnn_bert'],
                        help='Model architecture to train')
    parser.add_argument('--dataset', type=str, default='sarcasm_news',
                        help='Dataset name (sarcasm_news or sarc)')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Optional: Path to raw/processed data. If not provided, auto-detects or downloads')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for non-BERT components')
    parser.add_argument('--bert_lr', type=float, default=2e-5,
                        help='Learning rate for BERT components')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Early stopping patience (epochs)')
    args = parser.parse_args()
    
    set_seed(42)
    
    # Load data using preprocessing pipeline
    print("Loading data and building vocab...")
    train_df, val_df, test_df, _ = prepare_data(
        dataset=args.dataset,
        raw_data_path=args.raw_data_path,
        save_to_disk=False,
        min_freq=2
    )
    
    print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Train model
    model_path = train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir="results/models",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bert_lr=args.bert_lr,
        early_stop_patience=args.early_stop_patience,
    )
    
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()

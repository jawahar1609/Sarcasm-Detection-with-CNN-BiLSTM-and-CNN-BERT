from pathlib import Path
import pandas as pd
from .preprocess import preprocess_headlines, preprocess_sarc
from .utils import build_vocab, tokenize
from .download import auto_download_dataset
from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR, DATASETS
from collections import Counter


def _load_processed_csvs(data_path, dataset_name=None):
    """Load train/val/test CSVs from a processed data directory."""
    data_path = Path(data_path)
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    print(f"Loaded from processed CSVs:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def prepare_data(
        dataset='sarcasm_news',
        raw_data_path=None,
        save_to_disk=False,
        output_dir='processed_data',
        min_freq=2,
        max_vocab=None,
        auto_download=True,
        **kwargs):
    raw_data_path = Path(raw_data_path)
    
    # Set output directory based on dataset
    out_dir = f"{output_dir}/{dataset}" if save_to_disk else None
    
    # Process based on dataset type
    if dataset == 'sarcasm_news':
        if raw_data_path.is_file() and raw_data_path.suffix == '.json':
            # Raw JSON file - preprocess it
            print(f"Processing Sarcasm Headlines dataset...")
            train_df, val_df, test_df = preprocess_headlines(
                json_path=str(raw_data_path),
                out_dir=out_dir or DATASETS['sarcasm_news']['processed_subdir'],
                save_to_disk=save_to_disk,
                **kwargs
            )
            text_column = 'text'
            
        elif raw_data_path.is_dir() and (raw_data_path / "train.csv").exists():
            # Processed directory - load CSVs
            train_df, val_df, test_df = _load_processed_csvs(raw_data_path, 'sarcasm_news')
            text_column = 'text'
        else:
            raise FileNotFoundError(
                f"Invalid path for sarcasm_news: {raw_data_path}\n"
                f"Expected either:\n"
                f"  - JSON file: {DATASETS['sarcasm_news']['raw_filename']}\n"
                f"  - Processed directory with train/val/test.csv"
            )
        
    elif dataset == 'sarc':
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Data path not found: {raw_data_path}")
        
        # Check if this is a raw data directory or processed directory
        raw_csv = raw_data_path / DATASETS['sarc']['raw_filename']
        processed_train = raw_data_path / "train.csv"
        
        if raw_csv.exists():
            # Raw data directory - preprocess it
            print(f"Processing SARC (Reddit) dataset from raw data...")
            train_df, val_df, test_df = preprocess_sarc(
                data_dir=str(raw_data_path),
                out_dir=out_dir or DATASETS['sarc']['processed_subdir'],
                save_to_disk=save_to_disk,
                **kwargs
            )
            text_column = 'final_text'
            
        elif processed_train.exists():
            # Processed data directory - load CSVs
            train_df, val_df, test_df = _load_processed_csvs(raw_data_path, 'sarc')
            
            # Detect column name
            if 'final_text' in train_df.columns:
                text_column = 'final_text'
            elif 'text' in train_df.columns:
                text_column = 'text'
            else:
                raise ValueError(f"Could not find text column in {raw_data_path}/train.csv")
        else:
            raise FileNotFoundError(
                f"SARC data not found in {raw_data_path}\n"
                f"Expected either:\n"
                f"  - Raw: {raw_csv}\n"
                f"  - Processed: {processed_train}\n"
                f"\nPlease run 'python download_data.py' or preprocess your data first."
            )
        
        # Rename to 'text' for consistency if needed
        if text_column != 'text':
            train_df = train_df.rename(columns={text_column: 'text'})
            val_df = val_df.rename(columns={text_column: 'text'})
            test_df = test_df.rename(columns={text_column: 'text'})
            text_column = 'text'
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'sarcasm_news' or 'sarc'")
    
    # Build vocabulary from training data only
    print(f"\nBuilding vocabulary (min_freq={min_freq})...")
    vocab_stoi, vocab_itos = build_vocab(
        train_df[text_column].tolist(),
        min_freq=min_freq
    )
    
    # Apply max_vocab limit if specified
    if max_vocab is not None and len(vocab_stoi) > max_vocab:
        print(f"Limiting vocabulary from {len(vocab_stoi)} to {max_vocab} tokens")
        
        word_counts = Counter()
        for text in train_df[text_column].tolist():
            word_counts.update(tokenize(text))
        
        # Keep top max_vocab - 2 words (excluding special tokens)
        top_words = [word for word, _ in word_counts.most_common(max_vocab - 2)]
        
        # Rebuild vocab with limit
        vocab_stoi = {'<pad>': 0, '<unk>': 1}
        for word in top_words:
            if word not in vocab_stoi:
                vocab_stoi[word] = len(vocab_stoi)
        
        vocab_itos = {i: s for s, i in vocab_stoi.items()}
    
    print(f"Vocabulary size: {len(vocab_stoi)}")
    print(f"\nData preparation complete!")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df, vocab_stoi

# Unified data pipeline
from .data_pipeline import prepare_data

# Auto-download functionality
from .download import auto_download_dataset

# Data preprocessing functions
from .preprocess import preprocess_headlines, preprocess_sarc
from .dataset import SarcasmDataset

# Cleaning utilities
from .utils import (
    clean_text,
    clean_text_basic,
    remove_html,
    normalize_whitespace,
    tokenize,
    build_vocab,
    split_train_val_test,
    print_split_summary
)

# Configuration
from .config import RANDOM_SEED, TEST_SIZE, VAL_SIZE, ensure_dir, format_file_size, DATASETS

# Build __all__ dynamically
__all__ = [
    # Pipeline
    "prepare_data",
    "auto_download_dataset",
    # Preprocessing
    "preprocess_headlines",
    "preprocess_sarc",
    # Tokenization
    "tokenize",
    "build_vocab",
    # Utilities
    "clean_text",
    "clean_text_basic",
    "remove_html",
    "normalize_whitespace",
    "split_train_val_test",
    "print_split_summary",
    # Config
    "RANDOM_SEED",
    "TEST_SIZE",
    "VAL_SIZE",
    "DATASETS",
    "ensure_dir",
    "format_file_size",
]

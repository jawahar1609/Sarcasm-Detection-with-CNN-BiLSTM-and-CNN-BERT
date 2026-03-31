from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.20  # 20% of data for testing
VAL_SIZE = 0.10   # 10% of remaining data for validation

RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"

DATASETS = {
    'sarcasm_news': {
        'name': 'Sarcasm Headlines Dataset v2',
        'kaggle_path': 'rmisra/news-headlines-dataset-for-sarcasm-detection',
        'raw_subdir': RAW_DATA_DIR,
        'processed_subdir': f'{PROCESSED_DATA_DIR}/sarcasm_news',
        'raw_filename': 'Sarcasm_Headlines_Dataset_v2.json',
        'file_pattern': '**/Sarcasm_Headlines_Dataset*.json',
        'fallback_pattern': '**/*.json',
    },
    'sarc': {
        'name': 'SARC Dataset (Reddit Sarcasm)',
        'kaggle_path': 'danofer/sarcasm',
        'raw_subdir': f'{RAW_DATA_DIR}/sarc',
        'processed_subdir': f'{PROCESSED_DATA_DIR}/sarc',
        'raw_filename': 'train-balanced-sarcasm.csv',
        'file_pattern': '**/train-balanced*.csv',
        'fallback_pattern': '**/*.csv',
    }
}

DEFAULT_MIN_FREQ = 2
DEFAULT_MAX_VOCAB = None  # None means no limit

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_file_size(size_bytes):
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"

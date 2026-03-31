from pathlib import Path
import shutil
import kagglehub
from .config import ensure_dir, format_file_size, DATASETS


def _download_dataset(dataset_key, raw_data_dir=None, return_dir=False):
    config = DATASETS[dataset_key]
    data_dir = Path(raw_data_dir or config['raw_subdir'])
    ensure_dir(data_dir)
    
    target_file = data_dir / config['raw_filename']
    
    if target_file.exists():
        return data_dir if return_dir else target_file
    
    print(f"\nDownloading {config['name']} from Kaggle...")
    path = kagglehub.dataset_download(config['kaggle_path'])
    downloaded_path = Path(path)
    
    files = list(downloaded_path.glob(config['file_pattern']))
    if not files:
        files = list(downloaded_path.glob(config['fallback_pattern']))
    
    if not files:
        raise FileNotFoundError(
            f"Could not find file in downloaded dataset at {path}\n"
            f"Please check the dataset structure or download manually from:\n"
            f"https://www.kaggle.com/datasets/{config['kaggle_path']}"
        )
    
    shutil.copy(files[0], target_file)
    print(f"Downloaded ({format_file_size(target_file.stat().st_size)})")
    return data_dir if return_dir else target_file


def download_sarcasm_headlines(raw_data_dir=None):
    return _download_dataset('sarcasm_news', raw_data_dir, return_dir=False)


def download_sarc(raw_data_dir=None):
    return _download_dataset('sarc', raw_data_dir, return_dir=True)


def auto_download_dataset(dataset='sarcasm_news', raw_data_dir=None):
    if dataset == 'sarcasm_news':
        return download_sarcasm_headlines(raw_data_dir)
    elif dataset == 'sarc':
        return download_sarc(raw_data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'sarcasm_news' or 'sarc'")
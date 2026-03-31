import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split

# Suppress BeautifulSoup warnings for URL-like strings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def remove_html(text):
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text()


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text_basic(text):
    if not isinstance(text, str):
        return ""
    
    text = remove_html(text)
    text = normalize_whitespace(text)
    
    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = remove_html(text)
    
    # Remove URLs (http/https and www)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tokenize(text: str):
    return text.lower().strip().split()


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    # special tokens
    stoi = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            stoi.setdefault(word, len(stoi))

    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def split_train_val_test(data, test_size=0.20, val_size=0.10, random_seed=42, stratify_column='label'):
    # Train/Test split
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_seed,
        stratify=data[stratify_column]
    )

    # Train/Val split
    train_data, val_data = train_test_split(
        train_data,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_data[stratify_column]
    )

    return train_data, val_data, test_data


def print_split_summary(train_df, val_df, test_df, title="Final splits"):
    print(f"\n{title}:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)}")

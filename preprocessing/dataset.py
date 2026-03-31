import torch
from torch.utils.data import Dataset
import pandas as pd
from .utils import tokenize

class SarcasmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab, max_len: int, return_extras: bool = False):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab["<pad>"]
        self.unk_idx = vocab["<unk>"]
        self.return_extras = return_extras

    def __len__(self):
        return len(self.texts)

    def encode(self, text: str):
        tokens = tokenize(text)
        ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return ids, tokens[:self.max_len]

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        ids, tokens = self.encode(text)
        attention_mask = [1 if id != self.pad_idx else 0 for id in ids]
        
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        if self.return_extras:
            return ids_tensor, attention_mask_tensor, label_tensor, tokens, text
        else:
            return ids_tensor, attention_mask_tensor, label_tensor

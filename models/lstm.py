import torch
import torch.nn as nn
import torch.nn.functional as f

class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()

        # word embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # bilstm branch
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        lstm_feat_dim = 2 * lstm_hidden_dim 
        combined_dim =  lstm_feat_dim

        self.dropout = nn.Dropout(dropout)
        # one logit for binary classification
        self.fc = nn.Linear(combined_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids is a batch of token ids
        x = self.embedding(input_ids)

        # bilstm branch
        lstm_out, (h_n, c_n) = self.bilstm(x)
        # use last layer forward + backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        lstm_feat = torch.cat([h_forward, h_backward], dim=1)

        # combine cnn + lstm features
        lstm_feat = self.dropout(lstm_feat)
        logits = self.fc(lstm_feat)

        return logits
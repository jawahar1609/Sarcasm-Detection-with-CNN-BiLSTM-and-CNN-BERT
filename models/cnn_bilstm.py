import torch
import torch.nn as nn
import torch.nn.functional as f

class HybridCNNBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        cnn_num_filters: int = 100,
        cnn_filter_sizes=(3, 4, 5),
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

        # cnn branch with multiple filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=cnn_num_filters,
                kernel_size=fs,
            )
            for fs in cnn_filter_sizes
        ])

        # bilstm branch
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        cnn_feat_dim = cnn_num_filters * len(cnn_filter_sizes) 
        lstm_feat_dim = 2 * lstm_hidden_dim 
        combined_dim = cnn_feat_dim + lstm_feat_dim

        self.dropout = nn.Dropout(dropout)
        # one logit for binary classification
        self.fc = nn.Linear(combined_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids is a batch of token ids
        x = self.embedding(input_ids)

        # cnn branch
        x_cnn = x.permute(0, 2, 1)
        cnn_feats = []
        for conv in self.convs:
            c = f.relu(conv(x_cnn))
            c = f.max_pool1d(c, c.shape[2])
            c = c.squeeze(2)
            cnn_feats.append(c)
        cnn_out = torch.cat(cnn_feats, dim=1)

        # bilstm branch
        lstm_out, (h_n, c_n) = self.bilstm(x)
        # use last layer forward + backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        lstm_feat = torch.cat([h_forward, h_backward], dim=1)

        # combine cnn + lstm features
        combined = torch.cat([cnn_out, lstm_feat], dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)

        return logits
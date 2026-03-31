import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
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

        cnn_feat_dim = cnn_num_filters * len(cnn_filter_sizes) 

        self.dropout = nn.Dropout(dropout)
        # one logit for binary classification
        self.fc = nn.Linear(cnn_feat_dim, 1)

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

        dropout = self.dropout(cnn_out)
        logits = self.fc(dropout)

        return logits
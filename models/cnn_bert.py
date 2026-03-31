import torch
import torch.nn as nn
import torch.nn.functional as f

class HybridCNNBert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        bert_num_layers: int = 12,
        bert_attn_heads: int = 16,
        bert_intermediate_size: int = 2048,
        cnn_num_filters: int = 100,
        cnn_filter_sizes=(3, 4, 5),
        pad_idx: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()

        bert_hidden_dim = embed_dim  # ensure embedding dim matches BERT hidden size
        if embed_dim % bert_attn_heads != 0:
            raise ValueError("embed_dim must be divisible by bert_attn_heads")

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

        # BERT branch

        from transformers import BertConfig

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=bert_hidden_dim,
            num_hidden_layers=bert_num_layers,
            num_attention_heads=bert_attn_heads,
            intermediate_size=bert_intermediate_size,

            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,

            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,

            # training / misc
            pad_token_id=0,
        )
        from transformers import BertModel
        self.bert = BertModel(config)

        cnn_feat_dim = cnn_num_filters * len(cnn_filter_sizes) 
        bert_feat_dim = config.hidden_size
        combined_dim = cnn_feat_dim + bert_feat_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(combined_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
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

        # BERT branch - use shared embeddings with CNN
        # This ensures both CNN and BERT use the same embedding layer
        bert_outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)

        if attention_mask is not None:
            hidden_states = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            
            masked_hidden = hidden_states * mask_expanded  # Zero out padding
            sum_hidden = masked_hidden.sum(dim=1)          # [batch, hidden] - sum per sequence

            # add and divide to get the mean
            count_tokens = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # Prevent division by zero
            
            bert_feat = sum_hidden / count_tokens  # [batch, hidden] - average per sequence
        else:
            bert_feat = bert_outputs.last_hidden_state.mean(dim=1)

        # combine cnn + bert features
        combined = torch.cat([cnn_out, bert_feat], dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)

        return logits
# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        """
        Bahdanau Additive Attention with Xavier init, temperature scaling,
        and padding mask support to prevent flat uniform attention weights.
        """
        super(BahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_historical = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_current    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_score      = nn.Linear(hidden_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W_historical.weight)
        nn.init.xavier_uniform_(self.W_current.weight)
        nn.init.xavier_uniform_(self.V_score.weight)

    def forward(self, lstm_outputs: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Args:
            lstm_outputs : [Batch, SeqLen, Hidden]
            padding_mask : [Batch, SeqLen] — True where token == 0 (padding)
        Returns:
            context_vector   : [Batch, Hidden]
            attention_weights: [Batch, SeqLen, 1]
        """
        current_query = lstm_outputs[:, -1, :].unsqueeze(1)          # [B, 1, H]
        score_energy  = torch.tanh(
            self.W_historical(lstm_outputs) + self.W_current(current_query)
        )
        raw_scores    = self.V_score(score_energy)                    # [B, SeqLen, 1]
        # Lower temperature (0.1 * sqrt(d)) sharpens softmax distribution
        scaled_scores = raw_scores / (0.1 * math.sqrt(self.hidden_dim))

        # Mask padding positions to -inf so softmax assigns them ~0 weight
        if padding_mask is not None:
            scaled_scores = scaled_scores.masked_fill(
                padding_mask.unsqueeze(-1), -1e9
            )

        attention_weights = F.softmax(scaled_scores, dim=1)           # [B, SeqLen, 1]
        context_vector    = torch.sum(lstm_outputs * attention_weights, dim=1)  # [B, H]
        return context_vector, attention_weights


class GRU4Rec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super(GRU4Rec, self).__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru        = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(item_seq)
        gru_out, _ = self.gru(embedded)
        return self.output_head(gru_out[:, -1, :])


class StackedLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super(StackedLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout
        )
        self.attention   = BahdanauAttention(hidden_dim=hidden_dim)
        self.layer_norm  = nn.LayerNorm(hidden_dim)   # normalises hidden states → sharper attention
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def load_pretrained_embeddings(self, weight_matrix: np.ndarray):
        self.embedding.weight.data.copy_(torch.from_numpy(weight_matrix).float())
        print("[+] Word2Vec embedding weight matrix loaded into PyTorch layer successfully.")

    def forward(self, item_seq: torch.Tensor):
        padding_mask = (item_seq == 0)                                # [B, SeqLen]
        embedded     = self.embedding(item_seq)
        lstm_out, _  = self.lstm(embedded)
        lstm_out     = self.layer_norm(lstm_out)   # normalise before attention
        context_vector, attention_weights = self.attention(lstm_out, padding_mask)
        logits = self.output_head(context_vector)
        return logits, attention_weights
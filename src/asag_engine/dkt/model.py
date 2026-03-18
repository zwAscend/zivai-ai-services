from __future__ import annotations

import mindspore.nn as nn


class _DKTCore(nn.Cell):
    def __init__(
        self,
        num_skills: int,
        embed_dim: int = 96,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        vocab_size = (num_skills * 2) + 1
        self.emb = nn.Embedding(vocab_size=vocab_size, embedding_size=embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            has_bias=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Dense(hidden_dim, num_skills)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return self.sigmoid(logits)


class DKTNetLSTM(nn.Cell):
    def __init__(
        self,
        num_skills: int,
        embed_dim: int = 96,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # The checkpoint keys are prefixed with `net.`, so keep a wrapper cell
        # with that attribute name to load the published artifact cleanly.
        self.net = _DKTCore(
            num_skills=num_skills,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def construct(self, x):
        return self.net(x)

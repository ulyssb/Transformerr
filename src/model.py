import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.tokenizer import PAD_ID

DEFAULT_CHECKPOINT_PATH = Path("artifacts/model.pt")


def get_device() -> torch.device:
    """cuda > mps (Apple Silicon) > cpu, whichever is actually available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """True above the diagonal, i.e. future positions are masked."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_length: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length).unsqueeze(1)  # (max_length, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (batch, seq_len, d_model), mask: bool, True = masked out

        batch, seq_len, _ = x.shape

        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (batch, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ V  # (batch, n_heads, seq_len, d_head)
        # merge heads: (batch, n_heads, seq_len, d_head) -> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise MLP with GELU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Small GPT-style decoder-only LM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=PAD_ID
        )
        self.position_encoding = PositionalEncoding(
            config.d_model, config.max_length, config.dropout
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying (Press & Wolf, 2016)
        self.lm_head.weight = self.token_embedding.weight

        # N(0, 1) would give logits with std sqrt(d_model) at init; with std 1/sqrt(d_model)
        # they start at ~1, and forward() scales the input side back up by sqrt(d_model)
        nn.init.normal_(self.token_embedding.weight, std=config.d_model**-0.5)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # input_ids, attention_mask: (batch, seq_len) -> logits: (batch, seq_len, vocab)
        _, seq_len = input_ids.shape
        device = input_ids.device

        # scaled so tokens are on the same scale as the sinusoidal PE (Vaswani et al., 2017)
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.position_encoding(x)

        # packed blocks can attend across stories; a per-story document mask is a possible experiment
        mask = causal_mask(seq_len, device)[None, None, :, :]
        if attention_mask is not None:
            # don't attend to padding keys
            pad_mask = (attention_mask == 0)[:, None, None, :]
            mask = mask | pad_mask

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.lm_head(x)


def save_checkpoint(model: TransformerLM, path: Path | str = DEFAULT_CHECKPOINT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_checkpoint(
    path: Path | str = DEFAULT_CHECKPOINT_PATH, device: torch.device | None = None
) -> TransformerLM:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device or "cpu")
    model = TransformerLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device) if device is not None else model

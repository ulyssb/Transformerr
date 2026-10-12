from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    max_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
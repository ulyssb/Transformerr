import torch

from src.config import ModelConfig
from src.model import TransformerLM, causal_mask


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50, max_length=16, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0
    )


def test_causal_mask_shape_and_values() -> None:
    mask = causal_mask(4, torch.device("cpu"))

    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    assert not mask.diagonal().any()
    assert mask[0, 1] and mask[0, 3]
    assert not mask[3, 0]


def test_forward_shape_and_backward() -> None:
    model = TransformerLM(tiny_config())
    input_ids = torch.randint(0, tiny_config().vocab_size, (2, 10))

    logits = model(input_ids)

    assert logits.shape == (2, 10, tiny_config().vocab_size)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None


def test_causal_masking_blocks_future_tokens() -> None:
    # changing the last token shouldn't affect earlier positions
    torch.manual_seed(0)
    model = TransformerLM(tiny_config())
    model.eval()

    ids_a = torch.tensor([[1, 2, 3, 4, 5]])
    ids_b = ids_a.clone()
    ids_b[0, -1] = 6  # change only the last (future, relative to earlier positions) token

    with torch.no_grad():
        out_a = model(ids_a)
        out_b = model(ids_b)

    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1])


def test_padding_mask_ignored() -> None:
    torch.manual_seed(0)
    model = TransformerLM(tiny_config())
    model.eval()

    input_ids_a = torch.tensor([[1, 2, 3, 0, 0]])  # id 0 == PAD_ID
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    input_ids_b = input_ids_a.clone()
    input_ids_b[0, -1] = 7  # change content of a padded (masked-out) slot

    with torch.no_grad():
        out_a = model(input_ids_a, attention_mask)
        out_b = model(input_ids_b, attention_mask)

    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)

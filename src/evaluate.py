import math

import torch
import torch.nn.functional as F

from src.model import DEFAULT_CHECKPOINT_PATH, TransformerLM, get_device, load_checkpoint
from src.tokenizer import DEFAULT_SAVE_PATH, load_tokenizer, pack_texts


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()

    blocks = pack_texts(tokenizer, texts, model.config.max_length)  # (n_blocks, max_length)
    total_loss = 0.0

    for start in range(0, len(blocks), batch_size):
        input_ids = blocks[start : start + batch_size].to(device)

        logits = model(input_ids)  # (batch, seq_len, vocab)

        # sum, then divide by total token count at the end
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()

    avg_loss = total_loss / blocks.numel()
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


if __name__ == "__main__":
    from src.data import load_story_splits

    _train_stories, _validation_stories, test_stories = load_story_splits()

    device = get_device()
    tokenizer = load_tokenizer(DEFAULT_SAVE_PATH)
    model = load_checkpoint(DEFAULT_CHECKPOINT_PATH, device=device)

    metrics = evaluate(model, tokenizer, test_stories, device=device)
    print(f"test loss: {metrics['loss']:.4f}  test perplexity: {metrics['perplexity']:.2f}")

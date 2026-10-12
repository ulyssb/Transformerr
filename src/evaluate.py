import math

import torch
import torch.nn.functional as F

from src.model import DEFAULT_CHECKPOINT_PATH, TransformerLM, get_device, load_checkpoint
from src.tokenizer import DEFAULT_SAVE_PATH, IGNORE_INDEX, encode_batch, load_tokenizer


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

    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = encode_batch(tokenizer, texts[start : start + batch_size])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)  # (batch, seq_len, vocab)

        # sum, then divide by total token count at the end
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_real_tokens = (labels != IGNORE_INDEX).sum().item()

        total_loss += loss.item()
        total_tokens += n_real_tokens

    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


if __name__ == "__main__":
    from src.data import load_story_splits

    _train_stories, _validation_stories, test_stories = load_story_splits()

    device = get_device()
    tokenizer = load_tokenizer(DEFAULT_SAVE_PATH, max_length=128)
    model = load_checkpoint(DEFAULT_CHECKPOINT_PATH, device=device)

    metrics = evaluate(model, tokenizer, test_stories, device=device)
    print(f"test loss: {metrics['loss']:.4f}  test perplexity: {metrics['perplexity']:.2f}")

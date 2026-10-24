from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ModelConfig
from src.data import load_story_splits
from src.evaluate import evaluate
from src.model import (
    DEFAULT_CHECKPOINT_PATH,
    TransformerLM,
    get_device,
    save_checkpoint,
)
from src.tokenizer import DEFAULT_SAVE_PATH, load_tokenizer, pack_texts


def train(
    model: TransformerLM,
    tokenizer,
    train_texts: list[str],
    validation_texts: list[str],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: torch.device | None = None,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
) -> TransformerLM:
    device = device or get_device()
    model = model.to(device)

    # a tensor works as a dataset: indexing gives one block, batches get stacked
    train_blocks = pack_texts(tokenizer, train_texts, model.config.max_length)
    train_loader = DataLoader(train_blocks, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in progress:
            input_ids = batch.to(device)  # (batch, max_length), no padding

            logits = model(input_ids)
            # position t predicts token t+1: drop the last prediction and the first target
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=running_loss / n_batches)

        val_metrics = evaluate(model, tokenizer, validation_texts, batch_size, device)
        print(
            f"epoch {epoch}: train_loss={running_loss / n_batches:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_ppl={val_metrics['perplexity']:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, checkpoint_path)
            print(f"  saved new best checkpoint to {checkpoint_path}")

    return model


if __name__ == "__main__":
    train_stories, validation_stories, _test_stories = load_story_splits()

    tokenizer = load_tokenizer(DEFAULT_SAVE_PATH)
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = TransformerLM(config)

    train(model, tokenizer, train_stories, validation_stories)

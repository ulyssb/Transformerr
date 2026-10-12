from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
from src.tokenizer import DEFAULT_SAVE_PATH, IGNORE_INDEX, encode_batch, load_tokenizer


class StoryDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def make_collate_fn(tokenizer):
    def collate(batch: list[str]) -> dict[str, torch.Tensor]:
        return encode_batch(tokenizer, batch)

    return collate


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

    train_loader = DataLoader(
        StoryDataset(train_texts),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
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

    tokenizer = load_tokenizer(DEFAULT_SAVE_PATH, max_length=128)
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), max_length=128)
    model = TransformerLM(config)

    train(model, tokenizer, train_stories, validation_stories)

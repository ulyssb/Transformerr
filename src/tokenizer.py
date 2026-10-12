from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from src.data import load_story_splits

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
]

# special tokens get the first ids, in this order
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(len(SPECIAL_TOKENS))

IGNORE_INDEX = -100

DEFAULT_SAVE_PATH = Path("artifacts/tokenizer.json")


def train_tokenizer(
    texts: list[str],
    vocab_size: int = 8_000,
    save_path: Path | str = DEFAULT_SAVE_PATH,
) -> Tokenizer:
    save_path = Path(save_path)

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))

    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))

    return tokenizer


def load_tokenizer(
    path: Path | str = DEFAULT_SAVE_PATH,
    max_length: int | None = None,
) -> Tokenizer:
    """If max_length is set, pad/truncate every encoding to that length."""
    tokenizer = Tokenizer.from_file(str(path))

    if max_length is not None:
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding(
            pad_id=PAD_ID,
            pad_token=PAD_TOKEN,
            length=max_length,
        )

    return tokenizer


def encode_batch(tokenizer: Tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    """Needs padding enabled (load_tokenizer with max_length). Pad positions
    in labels are set to IGNORE_INDEX."""
    encodings = tokenizer.encode_batch(texts)

    input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
    attention_mask = torch.tensor(
        [e.attention_mask for e in encodings], dtype=torch.long
    )
    labels = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    train, validation, test = load_story_splits()

    tokenizer = train_tokenizer(train)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"pad/bos/eos/unk ids: {[tokenizer.token_to_id(t) for t in SPECIAL_TOKENS]}")

    example = train[0][:80]
    encoding = tokenizer.encode(example)
    print()
    print("Example text:", example)
    print("Tokens:", encoding.tokens[:8], "...")
    print("Ids:", encoding.ids[:8], "...")
    print("Decoded:", tokenizer.decode(encoding.ids))

    reloaded = load_tokenizer(max_length=32)
    batch = encode_batch(reloaded, train[:4])
    print()
    print("Batch input_ids shape:", tuple(batch["input_ids"].shape))
    print("Batch attention_mask shape:", tuple(batch["attention_mask"].shape))
    print("Batch labels shape:", tuple(batch["labels"].shape))
    print("First row labels (pad -> -100):", batch["labels"][0].tolist())

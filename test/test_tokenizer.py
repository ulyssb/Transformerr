from pathlib import Path

import pytest
import torch

from src.tokenizer import (
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    PAD_ID,
    encode_batch,
    load_tokenizer,
    pack_texts,
    train_tokenizer,
)

CORPUS = [
    "Once upon a time, there was a little girl named Lily.",
    "She loved to play in the park with her friends.",
    "One day, she found a big red ball under a tree.",
    "The little girl was very happy and smiled.",
]


def test_train_tokenizer_round_trip(tmp_path: Path) -> None:
    save_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(CORPUS, vocab_size=300, save_path=save_path)

    assert save_path.exists()
    assert tokenizer.get_vocab_size() >= 260

    text = CORPUS[0]
    encoding = tokenizer.encode(text)

    assert encoding.ids[0] == BOS_ID
    assert encoding.ids[-1] == EOS_ID
    # add_prefix_space adds a leading space
    assert tokenizer.decode(encoding.ids).lstrip() == text


def test_special_token_ids_are_fixed(tmp_path: Path) -> None:
    tokenizer = train_tokenizer(CORPUS, vocab_size=300, save_path=tmp_path / "t.json")

    assert tokenizer.token_to_id("<pad>") == PAD_ID == 0
    assert tokenizer.token_to_id("<bos>") == BOS_ID == 1
    assert tokenizer.token_to_id("<eos>") == EOS_ID == 2


def test_load_tokenizer_applies_padding(tmp_path: Path) -> None:
    save_path = tmp_path / "tokenizer.json"
    train_tokenizer(CORPUS, vocab_size=300, save_path=save_path)

    # tiny vocab -> long encodings, so max_length needs to be big enough to pad
    tokenizer = load_tokenizer(save_path, max_length=64)
    encoding = tokenizer.encode(CORPUS[1])

    assert len(encoding.ids) == 64
    assert encoding.ids[-1] == PAD_ID
    assert encoding.attention_mask[-1] == 0


def test_encode_batch_shapes_and_labels(tmp_path: Path) -> None:
    save_path = tmp_path / "tokenizer.json"
    train_tokenizer(CORPUS, vocab_size=300, save_path=save_path)
    tokenizer = load_tokenizer(save_path, max_length=16)

    batch = encode_batch(tokenizer, CORPUS)

    assert batch["input_ids"].shape == (4, 16)
    assert batch["attention_mask"].shape == (4, 16)
    assert batch["labels"].shape == (4, 16)
    assert batch["input_ids"].dtype == torch.long

    is_pad = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][is_pad] == IGNORE_INDEX)
    assert torch.all(batch["labels"][~is_pad] == batch["input_ids"][~is_pad])


def test_pack_texts_concatenates_and_cuts_into_blocks(tmp_path: Path) -> None:
    save_path = tmp_path / "tokenizer.json"
    train_tokenizer(CORPUS, vocab_size=300, save_path=save_path)
    tokenizer = load_tokenizer(save_path)

    stream = [i for text in CORPUS for i in tokenizer.encode(text).ids]
    blocks = pack_texts(tokenizer, CORPUS, block_size=16)

    assert blocks.shape == (len(stream) // 16, 16)
    assert blocks.dtype == torch.long
    # blocks are the stream, in order, with the short tail dropped
    assert blocks.view(-1).tolist() == stream[: blocks.numel()]
    # stories are separated by <eos> <bos>, never padded
    assert PAD_ID not in blocks


def test_pack_texts_rejects_padding_tokenizer(tmp_path: Path) -> None:
    save_path = tmp_path / "tokenizer.json"
    train_tokenizer(CORPUS, vocab_size=300, save_path=save_path)
    tokenizer = load_tokenizer(save_path, max_length=16)

    with pytest.raises(ValueError):
        pack_texts(tokenizer, CORPUS, block_size=16)

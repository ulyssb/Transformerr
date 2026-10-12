# Transformerr

Small GPT-style transformer LM written from scratch in PyTorch, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

Work in progress: attention isn't implemented yet, so nothing has been trained.

## Files

- `src/data.py`: loads TinyStories (10k train, 1k val, 1k test)
- `src/tokenizer.py`: byte-level BPE tokenizer, 8k vocab
- `src/baseline.py`: bigram model, for a baseline perplexity
- `src/model.py`: the transformer (4 layers, d_model 256, 4 heads, context 128)
- `src/train.py`, `src/evaluate.py`, `src/generate.py`
- `test/`: pytest tests

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

From the repo root:

```bash
python -m src.tokenizer
python -m src.baseline
python -m src.train
python -m src.evaluate
python -m src.generate
python -m pytest
```

Uses CUDA or MPS if available. TinyStories gets downloaded on the first run.

## TODO

- implement `MultiHeadSelfAttention.forward`
- shift labels by one in train/evaluate (right now position t is scored on token t, not t+1)
- train and compare against the bigram baseline
- tests for `data.py`
- maybe: lr warmup/decay, dynamic padding, packing stories instead of truncating

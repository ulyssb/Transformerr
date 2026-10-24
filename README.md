# Transformerr

Small GPT-style transformer LM written from scratch in PyTorch, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

It's a deliberately tiny model (5.2M parameters, ~16 min of training on an Apple Silicon laptop), so it writes fluent, story-shaped English but loses track of characters and logic over longer stretches.

## Results

Trained for 3 epochs on 100k stories (~21M tokens), context 256:

| model | test perplexity |
|---|---|
| bigram baseline (`baseline.py`), same data | 59.85 |
| transformer | **8.28** |

Sample (`generate.py`, prompt "A very, very long time ago", temperature 0.8, top-k 50):

> A very, very long time ago. The sun was shining and it was so peaceful and peaceful. Little Red knew that it meant to look like new people, so he carefully put on his belt and ran outside. He was glad that the sun could shine brightly.
>
> But then, he heard a loud noise. It was a little voice. It was her dad! Dad came outside and hugged him. He said he had done something special. The little yellow bird was sorry.
>
> Little Red was relieved. He promised that his dad would help him feel better soon. He said that if he wanted a shower, he could keep his pocket.
>
> So Red decided to take off his jacket and go outside to play. He was having so much fun!
>
> When sunset was over, Jimmy was very sleepy. He yawned and said goodnight to the stars, feeling happy and warm again.

The story arc and grammar hold up; who's who doesn't ("her dad", a bird out of nowhere, Red turning into Jimmy).

## Pipeline

```
data.py -> tokenizer.py -> model.py -> train.py -> evaluate.py
                                          \
                                           -> generate.py
```

`baseline.py` uses the same data and tokenizer, for comparison.

## Files

- `src/data.py`: loads TinyStories (100k train, 1k val, 1k test)
- `src/tokenizer.py`: byte-level BPE tokenizer, 8k vocab, plus `pack_texts` (stories wrapped in `<bos>`/`<eos>`, joined into one stream, cut into fixed-size blocks)
- `src/baseline.py`: bigram model, for a baseline perplexity
- `src/model.py`: the transformer (pre-norm decoder, 4 layers, d_model 256, 4 heads, context 256, sinusoidal positions, tied embeddings)
- `src/train.py`: AdamW, next-token cross-entropy, keeps the checkpoint with the best val loss
- `src/evaluate.py`, `src/generate.py` (temperature + top-k sampling)
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
python -m src.tokenizer   # only needed once, writes artifacts/tokenizer.json
python -m src.baseline
python -m src.train       # writes artifacts/model.pt
python -m src.evaluate
python -m src.generate
python -m pytest
```

Uses CUDA or MPS if available. TinyStories gets downloaded on the first run.

## TODO

- more data / epochs: val loss was still falling after epoch 3
- top-p sampling, KV caching in `generate.py`
- maybe: lr warmup/decay, a per-story document mask

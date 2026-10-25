# Transformerr

Small GPT-style transformer LM written from scratch in PyTorch, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

It's a deliberately tiny model (5.2M parameters, ~16 min of training on an Apple Silicon laptop), so it writes fluent, story-shaped English but loses track of characters and logic over longer stretches.

## Results

Trained for 3 epochs on 100k stories (~21M tokens), context 256:

| model | test perplexity |
|---|---|
| bigram baseline (`baseline.py`), same data | 59.85 |
| transformer | **8.28** |

Sample `generate.py`, prompt "The sun was out and it was shining.", temperature 0.8, top-k 50:

> The sun was out and it was shining. Little Tommy loved the sun and the birds. One day, he saw a little bird in the sky. The bird was sad and asked Tommy to come closer.
>
> "Let's go see the bird," said Tommy.
>
> The bird was scared and said, "No, it doesn't stay here. We need to be kind and help, and we can watch the bird and the birds. Then we can watch it and see what happens."
>
> Tommy nodded and walked towards the bird. The bird flapped its wings and flew away. Little Tommy watched the bird until it flew away.
>
> Inside, the bird found a big yellow apple in a tree. Tommy and the bird flew around the tree and saw the red apple in the branch. They wanted to eat the apple, but they could not.
>
> A man came to the park and saw Tommy playing with the apples and the blue bird. He said, "Hey, you! That is not a good apple. You should not throw it away. You can't throw it away."
>
> Tommy and the man looked at the apple and thought, "Okay, sir. Let's go and buy the apples."
>
> Tommy took the red apple and went to the man. He reached out his hand and handed it to the man. He said, "Here you go, but be careful. The apple is not for eating. It is good for your snack." Tommy smiled and said, "Thank you, man. You're welcome, mister. I love you."

Grammar and characters hold up (Tommy stays Tommy); the physical logic and who's speaking don't (Tommy flies, the yellow apple turns red, Tommy says both "thank you" and "you're welcome").

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

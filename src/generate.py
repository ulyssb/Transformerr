import torch
import torch.nn.functional as F

from src.model import DEFAULT_CHECKPOINT_PATH, TransformerLM, get_device, load_checkpoint
from src.tokenizer import DEFAULT_SAVE_PATH, EOS_ID, load_tokenizer



@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 400,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    """Returns Generated text."""  
    # Currently uses top k sampling - consider changing to top nucleus.
    # Also implement KV caching.
    
    model.eval()
    device = next(model.parameters()).device

    # drop the <eos> added by the tokenizer
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids[:-1]], dtype=torch.long, device=device
    )

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.max_length :]
        logits = model(context)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1] / temperature  # (vocab_size)

        if top_k is not None:
            top_values, top_indices = torch.topk(next_logits, top_k)
            probs = torch.zeros_like(next_logits)
            probs.scatter_(0, top_indices, F.softmax(top_values, dim=-1))
        else:
            probs = F.softmax(next_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)  # Does the sampling
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

        if next_id.item() == EOS_ID:
            break

    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    device = get_device()
    tokenizer = load_tokenizer(DEFAULT_SAVE_PATH)  # no fixed padding needed here
    model = load_checkpoint(DEFAULT_CHECKPOINT_PATH, device=device)

    story = generate(model, tokenizer, prompt="A very, very long time ago")
    print(story)

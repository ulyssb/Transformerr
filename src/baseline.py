import math
from collections import Counter, defaultdict

from src.data import load_story_splits
from src.tokenizer import DEFAULT_SAVE_PATH, load_tokenizer


class BigramLanguageModel:
    """Bigram LM with add-one smoothing, used as a baseline."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.bigram_counts: dict[int, Counter] = defaultdict(Counter)
        self.unigram_totals: Counter = Counter()

    def fit(self, token_id_sequences: list[list[int]]) -> None:
        for ids in token_id_sequences:
            for prev, nxt in zip(ids, ids[1:]):
                self.bigram_counts[prev][nxt] += 1
                self.unigram_totals[prev] += 1

    def next_token_prob(self, prev_id: int, next_id: int) -> float:
        count = self.bigram_counts[prev_id][next_id]
        total = self.unigram_totals[prev_id]
        return (count + 1) / (total + self.vocab_size)

    def perplexity(self, token_id_sequences: list[list[int]]) -> float:
        log_prob_sum = 0.0
        n_tokens = 0
        for ids in token_id_sequences:
            for prev, nxt in zip(ids, ids[1:]):
                log_prob_sum += math.log(self.next_token_prob(prev, nxt))
                n_tokens += 1
        return math.exp(-log_prob_sum / n_tokens)


if __name__ == "__main__":
    train_stories, _validation_stories, test_stories = load_story_splits()
    tokenizer = load_tokenizer(DEFAULT_SAVE_PATH)

    train_ids = [tokenizer.encode(text).ids for text in train_stories]
    test_ids = [tokenizer.encode(text).ids for text in test_stories]

    model = BigramLanguageModel(vocab_size=tokenizer.get_vocab_size())
    model.fit(train_ids)

    print(f"Bigram baseline test perplexity: {model.perplexity(test_ids):.2f}")

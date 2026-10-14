from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Some TinyStories texts have punctuation that was UTF-8 encoded, decoded as
# Windows-1252 and saved again, e.g. “ -> â€œ. ” lost its last byte on the way
# (0x9d isn't a cp1252 character), so it's a bare â€ and can't be decoded back.
# Everything maps to ASCII, which is what the clean stories use.
# Order matters: the bare â€ has to come last.
MOJIBAKE_FIXES = [
    ("â€œ", '"'),  # “
    ("â€™", "'"),  # ’
    ("â€˜", "'"),  # ‘
    ("â€“", "-"),  # –
    ("â€”", "-"),  # —
    ("â€¦", "..."),  # …
    ("â€Š", ""),  # hair space
    ("â€", '"'),  # ”
]


def fix_mojibake(text: str) -> str:
    for broken, fixed in MOJIBAKE_FIXES:
        text = text.replace(broken, fixed)
    return text


def clean_story(text: str) -> str:
    return fix_mojibake(text).strip()


def load_story_splits(
    train_size: int = 10_000,
    evaluation_size: int = 2_000,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Returns (train, validation, test) lists of stories."""

    source_name = "roneneldan/TinyStories"

    train_dataset = load_dataset(
        source_name,
        split=f"train[:{train_size}]",
    )

    evaluation_dataset = load_dataset(
        source_name,
        split=f"validation[:{evaluation_size}]",
    )

    train_stories = [
        clean_story(example["text"])
        for example in train_dataset
        if clean_story(example["text"])
    ]

    evaluation_stories = [
        clean_story(example["text"])
        for example in evaluation_dataset
        if clean_story(example["text"])
    ]

    validation_stories, test_stories = train_test_split(
        evaluation_stories,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )

    return train_stories, validation_stories, test_stories


if __name__ == "__main__":
    train, validation, test = load_story_splits()

    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")
    print()
    print("Example training story:")
    print(train[0][:500])
from datasets import load_dataset
from sklearn.model_selection import train_test_split


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
        example["text"].strip()
        for example in train_dataset
        if example["text"].strip()
    ]

    evaluation_stories = [
        example["text"].strip()
        for example in evaluation_dataset
        if example["text"].strip()
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
from src.data import clean_story, fix_mojibake


def test_fix_mojibake_quotes() -> None:
    # real example from the train split
    broken = "shouted at her from behind, â€œBe careful not to touch it!â€ Annie"
    expected = 'shouted at her from behind, "Be careful not to touch it!" Annie'

    assert fix_mojibake(broken) == expected


def test_fix_mojibake_apostrophes_dashes_ellipsis() -> None:
    assert fix_mojibake("Iâ€™m here") == "I'm here"
    assert fix_mojibake("â€˜hiâ€™") == "'hi'"
    assert fix_mojibake("a â€“ b â€” c") == "a - b - c"
    assert fix_mojibake("waitâ€¦") == "wait..."
    assert fix_mojibake("onâ€Š/â€Šoff") == "on/off"


def test_fix_mojibake_leaves_clean_text_alone() -> None:
    text = 'Tom said, "Let\'s go - now..." and café stayed café.'

    assert fix_mojibake(text) == text


def test_clean_story_strips_after_fixing() -> None:
    assert clean_story("  â€œHi!â€\n") == '"Hi!"'

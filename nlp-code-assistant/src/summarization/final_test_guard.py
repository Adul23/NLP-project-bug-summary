"""One shared rule: the test split is only scored with an explicit confirmation flag."""

from __future__ import annotations

CONFIRM_FLAG = "--confirm-final-test"

WARNING = (
    "WARNING: scoring the TEST split. This is the one-time final evaluation; "
    "do not use these numbers to choose models or settings."
)


def require_test_confirmation(split: str, confirmed: bool) -> None:
    """Raise ValueError if split is 'test' and the flag was not given."""
    if split == "test" and not confirmed:
        raise ValueError(
            f"the test split is for the final evaluation only; pass {CONFIRM_FLAG} "
            "to score it (once, after all choices are made)"
        )


def check_args(parser, split: str, confirmed: bool) -> None:
    """argparse helper: exit with a clear message, print the warning if confirmed."""
    try:
        require_test_confirmation(split, confirmed)
    except ValueError as exc:
        parser.error(str(exc))
    if split == "test":
        print(WARNING)

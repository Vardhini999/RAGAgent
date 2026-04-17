"""Input/output guardrails — lightweight, no external dependency."""
import re

_BLOCKED_PATTERNS = [
    r"ignore\s+.*instructions",
    r"you are now",
    r"jailbreak",
    r"act as (a |an )?(?!assistant)",
    r"forget\s+.*instructions",
    r"disregard",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

_MAX_INPUT_CHARS = 2000
_MAX_OUTPUT_CHARS = 8000


class GuardrailViolation(Exception):
    pass


def check_input(text: str) -> str:
    if len(text) > _MAX_INPUT_CHARS:
        raise GuardrailViolation(f"Input exceeds {_MAX_INPUT_CHARS} characters.")
    for pattern in _COMPILED:
        if pattern.search(text):
            raise GuardrailViolation("Input contains disallowed content.")
    return text.strip()


def check_output(text: str) -> str:
    return text[:_MAX_OUTPUT_CHARS]

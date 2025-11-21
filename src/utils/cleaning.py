import re

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, collapse whitespace."""
    if text is None:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()
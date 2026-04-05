"""Sentence boundary detection for streaming LLM token output."""

import re

_ABBREV = re.compile(
    r"\b(Dr|Mr|Mrs|Prof|Nr|St|bzw|ca|etc|evtl|ggf|sog|z\.B|d\.h|u\.a|o\.ä)\.$",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"\d\.$")


def is_sentence_end(buffer):
    """Return True if the buffer ends at a sentence boundary."""
    text = buffer.rstrip()
    if not text:
        return False
    if text[-1] in (".", "!", "?"):
        if _ABBREV.search(text):
            return False
        if _DECIMAL.search(text):
            return False
        return True
    return False


def split_sentences(token_stream):
    """Accumulate tokens from a generator, yield complete sentences."""
    buffer = ""
    for token in token_stream:
        buffer += token
        if is_sentence_end(buffer):
            sentence = buffer.strip()
            if sentence:
                yield sentence
            buffer = ""
    remainder = buffer.strip()
    if remainder:
        yield remainder

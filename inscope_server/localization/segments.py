"""Deterministic text units with exact Unicode character offsets."""
from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int


def split_spans(text, granularity='paragraph'):
    if granularity not in ('paragraph', 'line', 'sentence'):
        raise ValueError('granularity must be paragraph, line or sentence')
    wrapper = re.fullmatch(r'<tool_response>\r?\n([\s\S]*?)\r?\n</tool_response>', text)
    offset, body = (wrapper.start(1), wrapper[1]) if wrapper else (0, text)
    if granularity == 'paragraph':
        cuts = [(m.start(), m.end()) for m in re.finditer(r'\r?\n[ \t]*\r?\n', body)]
    elif granularity == 'line':
        cuts = [(m.start(), m.end()) for m in re.finditer(r'\r?\n', body)]
    else:
        # ASCII sentence punctuation requires whitespace/end; Chinese terminal
        # punctuation does not. This is a documented rule, not an NLP model.
        cuts = [(m.end(), m.end()) for m in re.finditer(r'[。！？]+|[.!?]+(?=\s|$)', body)]
    spans, start = [], 0
    for end, following in cuts + [(len(body), len(body))]:
        part = body[start:end]
        if part.strip():
            spans.append(Span(offset + start + len(part) - len(part.lstrip()),
                              offset + end - len(part) + len(part.rstrip())))
        start = following
    return spans


def coalition_text(text, spans, mask):
    pieces, cursor = [], 0
    for i, span in enumerate(spans):
        pieces.append(text[cursor:span.start])
        if mask & (1 << i):
            pieces.append(text[span.start:span.end])
        cursor = span.end
    pieces.append(text[cursor:])
    return ''.join(pieces)

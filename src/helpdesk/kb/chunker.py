"""Markdown document chunker for knowledge base indexing.

Splits markdown documents into chunks suitable for vector embedding.
Each chunk corresponds to an H2 section, preserving the heading as
context for the embedding model.
"""

import re


def chunk_markdown(text: str, source: str = "") -> list[dict[str, str]]:
    """Split a markdown document into chunks on H2 (``##``) boundaries.

    Each chunk contains the full text of one section, including its
    heading. Content before the first H2 (or the entire document if
    there are no H2 headings) becomes a single chunk.

    Args:
        text: The markdown text to split.
        source: Label for the source document (e.g. filename),
            attached as metadata on each chunk.

    Returns:
        A list of dicts, each with ``"text"`` and ``"source"`` keys.
    """
    if not text.strip():
        return []

    # Split on lines that start with "## " (H2 headers).
    # The pattern captures the header so it stays with its section.
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    chunks = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        chunks.append({"text": stripped, "source": source})

    return chunks

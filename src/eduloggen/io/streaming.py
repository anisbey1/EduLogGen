"""Chunk iterators for processing large sources in bounded memory (FR-I.8)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

__all__ = ["chunked"]

T = TypeVar("T")


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive lists of at most ``size`` items.

    Args:
        items: Any iterable, e.g. the rows of :meth:`BaseReader.read`.
        size: Maximum chunk length; must be positive.

    Yields:
        Lists of items in their original order; only the last may be shorter.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError("size must be positive")
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk

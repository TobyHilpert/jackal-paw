from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    key: tuple[Any, ...]
    results: dict[str, Any]


class ResultsCache:
    def __init__(self) -> None:
        self._entry: CacheEntry | None = None

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        if self._entry is None or self._entry.key != key:
            return None
        return self._entry.results

    def set(self, key: tuple[Any, ...], results: dict[str, Any]) -> None:
        self._entry = CacheEntry(key=key, results=results)

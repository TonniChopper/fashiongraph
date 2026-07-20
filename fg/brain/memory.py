"""Lightweight memory store.

Holds session / brand / user facts as a flat dict with optional JSON
persistence. Enough for the Bootstrapper and Personal Stylist to carry brand
DNA and user preferences across turns; a richer store can replace it later
behind the same interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fg.config import settings

logger: logging.Logger = logging.getLogger(__name__)


class Memory:
    """A simple namespaced key-value memory with optional persistence.

    Attributes:
        namespace: Logical scope (e.g. a brand or user id).
        store: The in-memory dict.
    """

    def __init__(self, namespace: str = "session", persist: bool = False) -> None:
        """Initializes memory, loading from disk if persistence is on.

        Args:
            namespace: Scope name; also the filename when persisting.
            persist: If ``True``, load/save under ``<data_dir>/memory``.
        """
        self.namespace = namespace
        self._persist = persist
        self._path = settings.data_dir / "memory" / f"{namespace}.json"
        self.store: dict[str, Any] = {}
        if persist and self._path.exists():
            try:
                self.store = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load memory %s (%s).", self._path, exc)

    def remember(self, key: str, value: Any) -> None:
        """Stores a fact (and persists if enabled).

        Args:
            key: Fact name.
            value: JSON-serialisable value.
        """
        self.store[key] = value
        if self._persist:
            self._save()

    def update(self, facts: dict[str, Any]) -> None:
        """Merges multiple facts at once.

        Args:
            facts: Mapping of key → value.
        """
        self.store.update(facts)
        if self._persist:
            self._save()

    def recall(self, key: str, default: Any = None) -> Any:
        """Returns a stored fact or *default*."""
        return self.store.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        """Returns a shallow copy of all facts."""
        return dict(self.store)

    def _save(self) -> None:
        """Writes the store to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.store, indent=2), encoding="utf-8")

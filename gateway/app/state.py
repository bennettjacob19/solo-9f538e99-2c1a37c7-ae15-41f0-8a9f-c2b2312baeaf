"""Durable scheduler state that must survive a gateway restart.

Only the bits that decide whether a slot may be used are persisted: the
quarantine (active fault) record, the last dispatch timestamp and the last
batch id.  Everything else (the queue, caller futures) is in-memory by nature.

The file is tiny and rewritten atomically (``os.replace``) whenever the state
changes, so a crash cannot leave a half-written record behind.  Losing the
directory is equivalent to starting with an empty state; a corrupt file is
treated as "slot not confirmed" (fail closed) if it cannot be parsed at all,
because silently forgetting an unconfirmed batch reintroduces the bug.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger("embedding_gateway.state")


class StateStore:
    """Atomic JSON persistence for the scheduler's slot-authority state."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def load(self) -> dict[str, Any] | None:
        """Return the persisted state, ``None`` if absent.

        Raises :class:`StateStoreError` for corruption so the caller can decide
        to fail closed rather than treating a broken file as "no fault".
        """
        if self.path is None:
            return None
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StateStoreError(f"cannot read state file {self.path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise StateStoreError(
                f"state file {self.path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise StateStoreError(f"state file {self.path} is not a JSON object")
        return data

    def save(self, data: dict[str, Any]) -> None:
        if self.path is None:
            return
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
            # fsync the directory so the rename itself is durable.
            with suppress(OSError):
                dir_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp_name)
            raise


class StateStoreError(RuntimeError):
    """The persisted state could not be read."""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

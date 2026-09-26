"""Small SQLite database abstraction used throughout EDGE HUNTER."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
from typing import Any, Iterator


class Database:
    """SQLite-backed database abstraction.

    The connection is shared inside one application process because the local
    FastAPI application can execute requests from multiple threads. A re-entrant
    lock serializes low-level operations that are not already transactionally
    guarded by SQLite.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    def connect(self) -> sqlite3.Connection:
        """Open the database connection and create its parent directory."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.path,
                    timeout=30,
                    check_same_thread=False,
                )
                self._connection.row_factory = sqlite3.Row
                self._connection.execute("PRAGMA foreign_keys = ON")
            return self._connection

    @property
    def connection(self) -> sqlite3.Connection:
        """Return an active connection."""
        if self._connection is None:
            return self.connect()
        return self._connection

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute one SQL statement through the abstraction."""
        with self._lock:
            return self.connection.execute(sql, parameters)

    def commit(self) -> None:
        """Commit the current transaction."""
        with self._lock:
            self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run an atomic transaction under the database lock."""
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

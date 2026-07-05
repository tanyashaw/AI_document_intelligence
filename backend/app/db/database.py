"""
Lightweight SQLite persistence for user accounts.

Kept deliberately simple (stdlib sqlite3, no ORM) to match the rest of the
project's storage style (see app/memory/chat_memory.py, which is JSON-file
based). This gives us real, durable, uniquely-constrained user records
without adding a database server dependency.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

_DB_FILE = Path(__file__).parent / "app.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


# Run once at import time so the table exists before the first request.
init_db()
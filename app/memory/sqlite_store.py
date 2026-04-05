"""SQLite storage for personal facts with deduplication."""

import logging
import os
import sqlite3
from datetime import datetime

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    source_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
"""


class MemorySQLite:

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def add(self, user_id: str, subject: str, fact: str,
            category: str = "general", source_message: str = None) -> int | None:
        """Insert a memory. Returns id, or None if duplicate."""
        if self._deduplicate(user_id, subject, fact):
            log.debug("Duplicate memory skipped: %s — %s", subject, fact)
            return None
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO memories (user_id, subject, fact, category, source_message, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, subject, fact, category, source_message, now, now),
        )
        self._conn.commit()
        log.info("Stored memory #%d: %s — %s", cur.lastrowid, subject, fact)
        return cur.lastrowid

    def search_by_subject(self, subject: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE LOWER(subject) = LOWER(?)", (subject,)
        ).fetchall()
        log.debug("SQLite search_by_subject('%s'): %d results", subject, len(rows))
        return [dict(r) for r in rows]

    def search_by_text(self, query: str) -> list[dict]:
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE fact LIKE ? OR subject LIKE ?",
            (pattern, pattern),
        ).fetchall()
        log.debug("SQLite search_by_text('%s'): %d results", query[:60], len(rows))
        return [dict(r) for r in rows]

    def get_all_for_user(self, user_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, memory_id: int):
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def delete_all_for_user(self, user_id: str) -> int:
        cur = self._conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        self._conn.commit()
        log.info("Deleted %d memories for user %s", cur.rowcount, user_id)
        return cur.rowcount

    def update(self, memory_id: int, fact: str):
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE memories SET fact = ?, updated_at = ? WHERE id = ?",
            (fact, now, memory_id),
        )
        self._conn.commit()

    def _deduplicate(self, user_id: str, subject: str, fact: str) -> bool:
        """Return True if a matching fact already exists for this user+subject."""
        existing = self._conn.execute(
            "SELECT fact FROM memories WHERE user_id = ? AND LOWER(subject) = LOWER(?)",
            (user_id, subject),
        ).fetchall()
        fact_lower = fact.lower().strip()
        for row in existing:
            if row["fact"].lower().strip() == fact_lower:
                return True
        return False

    def close(self):
        self._conn.close()

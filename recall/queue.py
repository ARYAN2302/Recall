"""
recall.queue — correction persistence.

A correction is one (input, target) pair the user wants the model to
learn. We persist them so we can:
  - Re-probe PPL on prior corrections (for AVR VERIFY)
  - Re-train if the adapter gets corrupted
  - Audit what the model has actually been taught

SQLite is overkill for v1 but it's the right shape: atomic writes,
zero deps, survives crashes. JSONL would also work.
"""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    input TEXT NOT NULL,
    target TEXT NOT NULL,
    metadata TEXT,              -- JSON blob
    eval_pairs TEXT,            -- JSON list of [prompt, answer] probes
    status TEXT DEFAULT 'queued',  -- queued | trained | failed
    created_at REAL,
    trained_at REAL
);
CREATE TABLE IF NOT EXISTS snapshots (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,         -- path to safetensors file
    created_at REAL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS avr_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at REAL,
    after_correction TEXT,
    repair_steps INTEGER,
    converged INTEGER,
    drift_report TEXT           -- JSON
);
"""


class CorrectionQueue:
    """SQLite-backed correction queue.

    Thread-safe via SQLite's own locking. One connection per instance.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add(self, input: str, target: str,
            metadata: Optional[Dict] = None,
            eval_pairs: Optional[List[List[str]]] = None) -> str:
        """Add a correction. Returns the new correction id."""
        cid = f"c{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO corrections (id, input, target, metadata, eval_pairs, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, 'queued', ?)",
            (cid, input, target,
             json.dumps(metadata or {}),
             json.dumps(eval_pairs or [[input, target]]),
             time.time()),
        )
        self._conn.commit()
        return cid

    def get(self, cid: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM corrections WHERE id = ?", (cid,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_all(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM corrections ORDER BY created_at ASC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_trained(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM corrections WHERE status = 'trained' "
            "ORDER BY created_at ASC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_trained(self, cid: str) -> None:
        self._conn.execute(
            "UPDATE corrections SET status = 'trained', trained_at = ? "
            "WHERE id = ?", (time.time(), cid))
        self._conn.commit()

    def mark_failed(self, cid: str) -> None:
        self._conn.execute(
            "UPDATE corrections SET status = 'failed' WHERE id = ?", (cid,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM corrections").fetchone()[0]

    def count_trained(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM corrections WHERE status = 'trained'"
        ).fetchone()[0]

    def add_snapshot(self, path: str, note: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO snapshots (path, created_at, note) VALUES (?, ?, ?)",
            (path, time.time(), note))
        self._conn.commit()
        return cur.lastrowid

    def latest_snapshot(self) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM snapshots ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def log_avr(self, after_correction: str, repair_steps: int,
                converged: bool, drift_report: Dict) -> None:
        self._conn.execute(
            "INSERT INTO avr_log (triggered_at, after_correction, "
            "repair_steps, converged, drift_report) VALUES (?, ?, ?, ?, ?)",
            (time.time(), after_correction, repair_steps, int(converged),
             json.dumps(drift_report)))
        self._conn.commit()

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return {
            "id": row["id"],
            "input": row["input"],
            "target": row["target"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "eval_pairs": json.loads(row["eval_pairs"] or "[]"),
            "status": row["status"],
            "created_at": row["created_at"],
            "trained_at": row["trained_at"],
        }

    def close(self):
        self._conn.close()

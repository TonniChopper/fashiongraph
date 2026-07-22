"""SQLite-backed triple store for the fashion knowledge graph.

SQLite (stdlib, zero-dep, file-based) is the store; a NetworkX view can be
built over it later for graph algorithms. This keeps the KG free and local on
the M4, and — because it's pure stdlib — fully testable in-memory.

Node identity is the *canonical* key (lowercased); a display form is kept for
presentation.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from fg.config import settings
from fg.kg.schema import Triple, canonical_entity

logger: logging.Logger = logging.getLogger(__name__)


def _default_db_path() -> Path:
    """Default KG database path."""
    return settings.data_dir / "kg" / "fashion_kg.sqlite"


class KnowledgeGraph:
    """A persistent triple store with neighbour/query helpers.

    Attributes:
        conn: The SQLite connection.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Opens (and initialises) the KG database.

        Args:
            db_path: File path, or ``":memory:"`` for an ephemeral store;
                defaults to ``data/kg/fashion_kg.sqlite``.
        """
        if db_path == ":memory:":
            self.conn = sqlite3.connect(":memory:")
        else:
            p = Path(db_path) if db_path else _default_db_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(p))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Creates the triples table + indices if absent."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS triples (
                subject_key   TEXT NOT NULL,
                subject       TEXT NOT NULL,
                subject_type  TEXT,
                relation      TEXT NOT NULL,
                object_key    TEXT NOT NULL,
                object        TEXT NOT NULL,
                object_type   TEXT,
                source        TEXT,
                UNIQUE(subject_key, relation, object_key)
            );
            CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject_key);
            CREATE INDEX IF NOT EXISTS idx_object  ON triples(object_key);
            CREATE INDEX IF NOT EXISTS idx_relation ON triples(relation);
            """
        )
        self.conn.commit()

    def add_triples(self, triples: Iterable[Triple]) -> int:
        """Inserts valid triples (idempotent on the canonical key).

        Args:
            triples: Triples to add; invalid ones are skipped.

        Returns:
            Number of new rows inserted.
        """
        rows = [
            (t.subject_key, t.subject.strip(), t.subject_type, t.relation,
             t.object_key, t.object.strip(), t.object_type, t.source)
            for t in triples if t.is_valid()
        ]
        before = self.count()
        self.conn.executemany(
            """INSERT OR IGNORE INTO triples
               (subject_key, subject, subject_type, relation,
                object_key, object, object_type, source)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        return self.count() - before

    def neighbors(self, entity: str, limit: int = 50) -> list[dict]:
        """Returns facts where *entity* is subject or object.

        Args:
            entity: Entity surface form (matched on canonical key).
            limit: Max facts to return.

        Returns:
            Fact dicts with subject/relation/object/source.
        """
        key = canonical_entity(entity)
        cur = self.conn.execute(
            """SELECT subject, relation, object, subject_type, object_type, source
               FROM triples WHERE subject_key = ? OR object_key = ? LIMIT ?""",
            (key, key, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def facts_as_text(self, entity: str, limit: int = 25) -> list[str]:
        """Renders an entity's facts as readable ``subject relation object`` lines."""
        out: list[str] = []
        for f in self.neighbors(entity, limit=limit):
            rel = f["relation"].replace("_", " ")
            out.append(f"{f['subject']} {rel} {f['object']}")
        return out

    def edges_of(self, key: str) -> list[dict]:
        """Returns raw edges touching a canonical *key* (for traversal).

        Args:
            key: A canonical entity key (already normalised).

        Returns:
            Edge dicts with subject_key/object_key/subject/object/relation.
        """
        cur = self.conn.execute(
            """SELECT subject_key, object_key, subject, object, relation
               FROM triples WHERE subject_key = ? OR object_key = ?""",
            (key, key),
        )
        return [dict(r) for r in cur.fetchall()]

    def subjects_with(self, relation: str, object_name: str) -> list[str]:
        """Returns subjects related to *object_name* by *relation*.

        e.g. ``subjects_with("based_in", "Milan")`` → brands based in Milan.

        Args:
            relation: A relation type.
            object_name: Object entity (matched on canonical key).

        Returns:
            Distinct subject display names.
        """
        cur = self.conn.execute(
            """SELECT DISTINCT subject FROM triples
               WHERE relation = ? AND object_key = ?""",
            (relation, canonical_entity(object_name)),
        )
        return [r[0] for r in cur.fetchall()]

    def outgoing(self, entity: str) -> list[dict]:
        """Returns facts where *entity* is the subject (its own attributes).

        Args:
            entity: Entity surface form (matched on canonical key).

        Returns:
            Fact dicts with relation/object.
        """
        key = canonical_entity(entity)
        cur = self.conn.execute(
            "SELECT relation, object, object_type FROM triples WHERE subject_key = ?",
            (key,),
        )
        return [dict(r) for r in cur.fetchall()]

    def top_subjects(self, n: int = 10) -> list[tuple[str, str, int]]:
        """Returns the entities with the most outgoing facts.

        Args:
            n: How many to return.

        Returns:
            ``(display_name, canonical_key, fact_count)`` tuples, richest first.
        """
        cur = self.conn.execute(
            """SELECT subject_key, MIN(subject) AS disp, COUNT(*) AS c
               FROM triples GROUP BY subject_key ORDER BY c DESC LIMIT ?""",
            (n,),
        )
        return [(r["disp"], r["subject_key"], r["c"]) for r in cur.fetchall()]

    def entities(self) -> set[str]:
        """Returns the set of all canonical entity keys in the graph."""
        cur = self.conn.execute(
            "SELECT subject_key FROM triples UNION SELECT object_key FROM triples"
        )
        return {r[0] for r in cur.fetchall()}

    def count(self) -> int:
        """Returns the number of triples."""
        return self.conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def stats(self) -> dict:
        """Returns summary counts (triples, entities, relations breakdown)."""
        rel = self.conn.execute(
            "SELECT relation, COUNT(*) c FROM triples GROUP BY relation ORDER BY c DESC"
        ).fetchall()
        return {
            "triples": self.count(),
            "entities": len(self.entities()),
            "relations": {r["relation"]: r["c"] for r in rel},
        }

    def close(self) -> None:
        """Closes the connection."""
        self.conn.close()

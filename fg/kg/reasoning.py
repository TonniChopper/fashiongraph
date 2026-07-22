"""Graph reasoning over the fashion KG — the thing flat RAG cannot do.

Vector retrieval finds passages *similar* to a query. It cannot traverse
relationships: "what connects Margiela to Galliano?", "which designers
influenced houses based in Milan?". Those are graph operations — path-finding
and multi-hop joins — and they are where a knowledge graph structurally beats a
vector store.

Pure SQLite + BFS (no hard NetworkX dependency); an optional NetworkX view is
offered for algorithms like community detection.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from fg.kg.schema import canonical_entity
from fg.kg.store import KnowledgeGraph

logger: logging.Logger = logging.getLogger(__name__)


class GraphReasoner:
    """Multi-hop queries and path-finding over a :class:`KnowledgeGraph`.

    Attributes:
        kg: The backing knowledge graph.
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        """Wraps a knowledge graph with reasoning helpers.

        Args:
            kg: The knowledge graph to reason over.
        """
        self.kg = kg

    def subjects_with(self, relation: str, object_name: str) -> list[str]:
        """Subjects related to *object_name* by *relation* (one-hop filter)."""
        return self.kg.subjects_with(relation, object_name)

    def objects_of(self, entity: str, relation: str) -> list[str]:
        """Objects of *entity* under *relation* (e.g. what X is known_for)."""
        return [f["object"] for f in self.kg.outgoing(entity) if f["relation"] == relation]

    def paths(self, src: str, dst: str, max_hops: int = 3) -> list[list[dict]]:
        """Finds relationship paths from *src* to *dst* (BFS, ≤ max_hops).

        Edges are traversed in both directions (the graph is a web of facts,
        not a strict hierarchy). Returns the shortest paths found.

        Args:
            src: Start entity.
            dst: Target entity.
            max_hops: Maximum path length in edges.

        Returns:
            A list of paths; each path is a list of oriented step dicts
            (``from``/``relation``/``to``). Empty if unreachable.
        """
        src_k, dst_k = canonical_entity(src), canonical_entity(dst)
        if src_k == dst_k:
            return []
        frontier: deque[tuple[str, list[dict]]] = deque([(src_k, [])])
        visited: set[str] = {src_k}
        found: list[list[dict]] = []
        best_len: int | None = None

        while frontier:
            node, path = frontier.popleft()
            if len(path) >= max_hops:
                continue
            for edge in self.kg.edges_of(node):
                if edge["subject_key"] == node:
                    to_key, step = edge["object_key"], {
                        "from": edge["subject"], "relation": edge["relation"],
                        "to": edge["object"]}
                else:
                    to_key, step = edge["subject_key"], {
                        "from": edge["object"], "relation": edge["relation"],
                        "to": edge["subject"]}
                new_path = path + [step]
                if to_key == dst_k:
                    if best_len is None:
                        best_len = len(new_path)
                    if len(new_path) == best_len:
                        found.append(new_path)
                    continue
                if to_key not in visited and (best_len is None or len(new_path) < best_len):
                    visited.add(to_key)
                    frontier.append((to_key, new_path))
        return found

    def two_hop(self, entity: str, limit: int = 40) -> list[dict]:
        """Returns facts one and two hops out from *entity* (neighbourhood)."""
        first = self.kg.neighbors(entity, limit=limit)
        keys = {canonical_entity(f["object"]) for f in first} | {
            canonical_entity(f["subject"]) for f in first
        }
        keys.discard(canonical_entity(entity))
        second: list[dict] = []
        for k in list(keys)[:limit]:
            second.extend(self.kg.neighbors(k, limit=5))
        return first + second

    def to_networkx(self) -> Any:
        """Builds a NetworkX MultiDiGraph view (for algorithms).

        Returns:
            A ``networkx.MultiDiGraph``.

        Raises:
            RuntimeError: If NetworkX is not installed.
        """
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install networkx") from exc
        g = nx.MultiDiGraph()
        cur = self.kg.conn.execute(
            "SELECT subject, relation, object FROM triples"
        )
        for s, rel, o in cur.fetchall():
            g.add_edge(s, o, relation=rel)
        return g


def format_path(path: list[dict]) -> str:
    """Renders a path as ``A —relation→ B —relation→ C``.

    Args:
        path: Oriented step dicts (``from``/``relation``/``to``).

    Returns:
        A readable one-line path (empty string for an empty path).
    """
    if not path:
        return ""
    parts = [path[0]["from"]]
    for step in path:
        parts.append(f"—{step['relation'].replace('_', ' ')}→ {step['to']}")
    return " ".join(parts)

"""FastAPI serving layer — exposes the FashionGraph brain over HTTP.

Thin wrappers over the pieces the CLI already uses, so the web front end (React +
tldraw canvas) and any client share one backend:

* ``GET  /health``   — liveness + which components loaded.
* ``POST /chat``     — the autonomous ReAct agent (KG + live web search); answers
                       with citations and (optionally) grows the KG/RAG from what
                       it read.
* ``POST /analyze``  — Personal-Stylist look review from an uploaded photo.

Heavy components (LLM, KG, RAG, vision stack) are lazily initialised on first use
and cached, so importing/booting the app is cheap and a missing component degrades
gracefully instead of crashing startup.

Run::

    pip install fastapi uvicorn python-multipart
    uvicorn fg.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

#: Lazy singleton cache for expensive components.
_state: dict[str, Any] = {}


def _llm() -> Any:
    if "llm" not in _state:
        from fg.llm import get_llm
        _state["llm"] = get_llm()
    return _state["llm"]


def _kg() -> Any:
    if "kg" not in _state:
        try:
            from fg.kg.store import KnowledgeGraph
            _state["kg"] = KnowledgeGraph()
        except Exception as exc:  # noqa: BLE001
            logger.warning("KG unavailable (%s).", exc)
            _state["kg"] = None
    return _state["kg"]


def _indexer() -> Any:
    if "indexer" not in _state:
        try:
            from fg.rag.indexer import FashionKnowledgeIndexer
            _state["indexer"] = FashionKnowledgeIndexer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG indexer unavailable (%s).", exc)
            _state["indexer"] = None
    return _state["indexer"]


def _context() -> Any:
    """Builds a ContextBuilder (KG + RAG) for grounding the capability tools."""
    if "context" not in _state:
        from fg.brain.context_builder import ContextBuilder
        _state["context"] = ContextBuilder(_indexer(), kg=_kg())
    return _state["context"]


def _reviewer() -> Any:
    """Builds the look-review capability once (mirrors the CLI composition root)."""
    if "reviewer" not in _state:
        from fg.brain.context_builder import ContextBuilder
        from fg.capabilities.personal_stylist.look_review import LookReview
        from fg.llm import get_llm
        from fg.vision.perception import build_perception_stack

        try:
            llm = get_llm(vision=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision LLM unavailable (%s) — text fallback.", exc)
            llm = _llm()
        stack = build_perception_stack(on_note=lambda m: logger.info("perception: %s", m))
        ctx = ContextBuilder(_indexer(), kg=stack.kg or _kg())
        _state["reviewer"] = LookReview(
            llm, embedder=stack.embedder, segmenter=stack.segmenter,
            visual_index=stack.visual_index, aesthetic_scorer=stack.aesthetic_scorer,
            movement_matcher=stack.movement_matcher, kg_linker=stack.kg_linker,
            runway_linker=stack.runway_linker, kg=stack.kg,
            context_builder=ctx, vision=True,
        )
    return _state["reviewer"]


def create_app() -> Any:
    """Builds and returns the FastAPI application."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="FashionGraph API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    class ChatIn(BaseModel):
        message: str
        max_steps: int = 4
        learn: bool = True

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "kg": _kg() is not None,
            "rag": _indexer() is not None,
        }

    def _run_agent(body: "ChatIn") -> dict:
        """Shared agent run for /agent and /chat."""
        from fg.brain.agent import ReActAgent

        agent = ReActAgent(
            _llm(), kg=_kg(),
            indexer=_indexer() if body.learn else None,
            context_builder=_context(),
            max_steps=body.max_steps,
        )
        try:
            res = agent.run(body.message)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Agent failed: {exc}") from exc
        learned = agent.ingest_collected() if body.learn else {}
        # Minimal canvas payload: an answer card + a card per source.
        cards = [{"type": "answer", "text": res.answer}]
        cards += [{"type": "source", "url": u} for u in dict.fromkeys(res.sources)]
        return {
            "answer": res.answer,
            "sources": list(dict.fromkeys(res.sources)),
            "trace": res.trace,
            "learned": learned,
            "cards": cards,
        }

    @app.post("/agent")
    def agent(body: ChatIn) -> dict:
        """The one router: styling, brand DNA, trends, lineage, live facts — the
        agent picks the tool. Returns an answer + canvas cards + sources."""
        return _run_agent(body)

    @app.post("/chat")
    def chat(body: ChatIn) -> dict:
        """Alias of /agent (free-form chat)."""
        return _run_agent(body)

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), occasion: str = Form("")) -> dict:
        """Reviews an uploaded outfit photo (garments, styling, lineage, sources)."""
        data = await file.read()
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        tmp = Path(tempfile.gettempdir()) / f"fg_upload{suffix}"
        tmp.write_bytes(data)
        try:
            result = _reviewer().run({"image_path": str(tmp), "occasion": occasion})
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Analyze failed: {exc}") from exc
        return {
            "review": result.text,
            "garments": result.data.get("garments", []),
            "occasion": occasion,
            "sources": result.sources,
        }

    return app


app = create_app()

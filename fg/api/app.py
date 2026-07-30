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

# NOTE: no ``from __future__ import annotations`` here — it breaks FastAPI's
# multipart (File/Form) schema generation for /analyze on pydantic v2.

import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger: logging.Logger = logging.getLogger(__name__)


class SelItem(BaseModel):
    """One canvas object the user selected (the referent for 'this')."""
    type: str = "card"
    text: str = ""
    data: dict = Field(default_factory=dict)


class ChatIn(BaseModel):
    """Request body for /agent and /chat."""
    message: str
    selection: list[SelItem] = Field(default_factory=list)
    max_steps: int = 4
    learn: bool = True


def _render_selection(items: list[SelItem]) -> str:
    """Turns selected canvas objects into a compact referent string."""
    lines = []
    for it in items:
        body = it.text or (", ".join(f"{k}: {v}" for k, v in it.data.items()) if it.data else "")
        lines.append(f"- ({it.type}) {body}".rstrip())
    return "\n".join(lines)

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


def _retriever() -> Any:
    """RAG *read* side — a FashionRetriever (the indexer is write-only)."""
    if "retriever" not in _state:
        try:
            from fg.rag.retriever import FashionRetriever
            _state["retriever"] = FashionRetriever()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG retriever unavailable (%s).", exc)
            _state["retriever"] = None
    return _state["retriever"]


def _context() -> Any:
    """Builds a ContextBuilder (KG + RAG) for grounding the capability tools."""
    if "context" not in _state:
        from fg.brain.context_builder import ContextBuilder
        _state["context"] = ContextBuilder(_retriever(), kg=_kg())
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
        ctx = ContextBuilder(_retriever(), kg=stack.kg or _kg())
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

    app = FastAPI(title="FashionGraph API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

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
            res = agent.run(body.message, context=_render_selection(body.selection))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Agent failed: {exc}") from exc
        learned = agent.ingest_collected() if body.learn else {}
        # Canvas payload: the typed cards the capabilities produced (style / brand_dna
        # / trend / lineage), then source cards. The prose answer rides alongside for
        # the chat panel.
        cards = list(res.cards)
        cards += [{"type": "source", "url": u} for u in dict.fromkeys(res.sources)]
        if not res.cards:                       # plain conversational reply → one answer card
            cards.insert(0, {"type": "answer", "text": res.answer})
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

    @app.post("/agent/stream")
    def agent_stream(body: ChatIn):
        """Same router, streamed (SSE): live `step` + `card` events, then `final`.
        Lets the canvas show thinking and place cards incrementally."""
        import json
        import queue
        import threading
        from fastapi.responses import StreamingResponse
        from fg.brain.agent import ReActAgent

        q: "queue.Queue" = queue.Queue()
        emit = lambda kind, **d: q.put({"type": kind, **d})

        def work() -> None:
            agent = ReActAgent(
                _llm(), kg=_kg(), indexer=_indexer() if body.learn else None,
                context_builder=_context(), max_steps=body.max_steps,
                on_step=lambda m: emit("step", text=m),
                on_card=lambda c: emit("card", card=c),
            )
            try:
                res = agent.run(body.message, context=_render_selection(body.selection))
                learned = agent.ingest_collected() if body.learn else {}
                emit("final", answer=res.answer,
                     sources=list(dict.fromkeys(res.sources)), learned=learned)
            except Exception as exc:  # noqa: BLE001
                emit("error", message=str(exc))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def gen():
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- boards: server-side save / load ----
    _boards_dir = Path(_state.get("boards_dir") or "data/boards")

    class Board(BaseModel):
        id: str = ""
        name: str = "Untitled board"
        state: dict = Field(default_factory=dict)

    @app.get("/boards")
    def list_boards() -> list:
        _boards_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for p in sorted(_boards_dir.glob("*.json")):
            try:
                d = __import__("json").loads(p.read_text())
                out.append({"id": p.stem, "name": d.get("name", p.stem),
                            "saved": p.stat().st_mtime})
            except Exception:  # noqa: BLE001
                continue
        return out

    @app.get("/boards/{board_id}")
    def get_board(board_id: str) -> dict:
        import json as _j
        p = _boards_dir / f"{board_id}.json"
        if not p.exists():
            raise HTTPException(404, "No such board.")
        return _j.loads(p.read_text())

    @app.post("/boards")
    def save_board(board: Board) -> dict:
        import json as _j
        import re
        import time
        _boards_dir.mkdir(parents=True, exist_ok=True)
        bid = board.id or (re.sub(r"[^a-z0-9]+", "-", board.name.lower()).strip("-") or "board")
        data = {"id": bid, "name": board.name, "state": board.state, "saved": time.time()}
        (_boards_dir / f"{bid}.json").write_text(_j.dumps(data))
        return {"id": bid, "name": board.name}

    @app.delete("/boards/{board_id}")
    def delete_board(board_id: str) -> dict:
        (_boards_dir / f"{board_id}.json").unlink(missing_ok=True)
        return {"deleted": board_id}

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
        from fg.brain.output_contract import Depth, Format, OutputContract
        try:
            result = _reviewer().run({"image_path": str(tmp), "occasion": occasion},
                                     OutputContract(Depth.SURFACE, Format.CHAT))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Analyze failed: {exc}") from exc
        from fg.brain.cards import build_look_card

        garments = result.data.get("garments", [])
        return {
            "review": result.text,
            "garments": garments,
            "occasion": occasion,
            "sources": result.sources,
            "card": build_look_card(result.text, garments, result.sources),
        }

    @app.post("/compose")
    async def compose(files: list[UploadFile] = File(default=[]), note: str = Form("")) -> dict:
        """Reviews several garments/looks together as one outfit (one vision call)."""
        from fg.llm import get_llm
        from fg.llm.base import Message, encode_image

        imgs = []
        for f in files:
            try:
                imgs.append(encode_image(await f.read()))
            except Exception:  # noqa: BLE001
                continue
        if not imgs and not note:
            raise HTTPException(400, "Nothing to compose.")
        try:
            llm = get_llm(vision=True)
        except Exception:  # noqa: BLE001
            llm = _llm()
        system = ("You are an expert stylist composing a set of pieces into ONE outfit. "
                  "Look at all the images together. In a tight paragraph: does this group "
                  "work as a single look, what's the read, and 2–3 concrete moves to make "
                  "it cohere. Keep it short unless there's genuinely more to say.")
        user = Message("user", (note or "Compose and review these pieces as one outfit."), images=imgs)
        try:
            text = llm.chat([Message("system", system), user], max_tokens=380)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Compose failed: {exc}") from exc
        return {"answer": text, "card": {"type": "look", "review": text, "garments": [],
                                         "text": text[:180]}}

    # Serve the built front end (if present) so the whole app is one process on one
    # port — no separate dev server, no CORS. Registered LAST so API routes win.
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="app")
        logger.info("Serving front end from %s", dist)

    return app


app = create_app()

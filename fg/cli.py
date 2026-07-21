"""FashionGraph command-line entry point.

Installed command is ``fgraph`` (``fg`` is a reserved shell builtin). Or run as
a module: ``python -m fg.cli ...``.

    fgraph info                                   # resolved config
    fgraph data list                              # available ingest sources
    fgraph data build --source fashion_products,wikipedia [--limit N]
    fgraph data smoke "quiet luxury tailoring"    # retrieval smoke test
    fgraph bootstrap                              # interactive brand builder
    fgraph bootstrap --answers brand.json --out brand.md
    fgraph route "help me start a quiet-luxury knitwear label"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fg import __version__
from fg.config import settings


def _cmd_info() -> None:
    """Prints the resolved configuration."""
    print(f"FashionGraph {__version__}")
    print(f"  LLM backend:      {settings.llm_backend} ({settings.ollama_model})")
    print(f"  Fashion embedder: {settings.fashion_embed_model}")
    print(f"  Chroma dir:       {settings.chroma_dir}")
    print(f"  RAG collection:   {settings.chroma_collection}")


def _cmd_data_list() -> None:
    """Lists registered ingest sources."""
    from fg.data.sources import SOURCES

    print("Available ingest sources:\n")
    for spec in SOURCES.values():
        exists = "✓" if spec.default_root().exists() else "·"
        print(f"  [{exists}] {spec.name:18s} {spec.description}")
        print(f"       → {spec.default_root()}")


def _cmd_data_build(sources: str, limit: int | None) -> None:
    """Builds the knowledge index from the given comma-separated sources."""
    from fg.data.ingest import build

    names = [s.strip() for s in sources.split(",") if s.strip()]
    stats = build(names, limit=limit)
    print("\nIngest summary:")
    for k, v in stats.as_dict().items():
        print(f"  {k}: {v}")


def _cmd_data_smoke(query: str, n: int) -> None:
    """Runs a retrieval smoke test and prints results."""
    from fg.data.ingest import smoke

    results = smoke(query, n_results=n)
    if not results:
        print("No results — is the index built? Try `fg data build` first.")
        return
    print(f"\nTop {len(results)} for {query!r}:\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        src = meta.get("source", "?")
        title = meta.get("title", "")
        snippet = r["document"][:140].replace("\n", " ")
        print(f"  #{i} [{src}] {title}\n      {snippet}…  (dist={r['distance']:.3f})")


def _build_llm_and_context(backend: str | None):
    """Builds the LLM and a (RAG-grounded, if possible) context builder once."""
    from fg.brain.context_builder import ContextBuilder
    from fg.llm import get_llm

    llm = get_llm(backend) if backend else get_llm()
    retriever = None
    try:
        from fg.rag.retriever import FashionRetriever

        retriever = FashionRetriever()
    except Exception as exc:  # noqa: BLE001
        print(f"(note: running ungrounded — retriever unavailable: {exc})")
    return llm, ContextBuilder(retriever)


def _build_bootstrapper(backend: str | None):
    """Constructs a grounded BrandBootstrapper."""
    from fg.capabilities.strategize.bootstrapper import BrandBootstrapper

    llm, ctx = _build_llm_and_context(backend)
    return BrandBootstrapper(llm, ctx)


def _build_analyzer(backend: str | None):
    """Constructs a grounded TrendAnalyzer."""
    from fg.capabilities.understand.trend_analysis import TrendAnalyzer

    llm, ctx = _build_llm_and_context(backend)
    return TrendAnalyzer(llm, ctx)


def _build_router(backend: str | None):
    """Builds a router with both capabilities registered (shared LLM+context)."""
    from fg.brain.router import FashionRouter
    from fg.capabilities.strategize.bootstrapper import BrandBootstrapper
    from fg.capabilities.understand.trend_analysis import TrendAnalyzer

    llm, ctx = _build_llm_and_context(backend)
    router = FashionRouter(llm=llm)
    router.register(BrandBootstrapper(llm, ctx))
    router.register(TrendAnalyzer(llm, ctx))
    return router


def _cmd_analyze(topic, out, backend, depth, fmt) -> None:
    """Runs trend analysis on a topic."""
    from fg.brain.output_contract import OutputContract

    analyzer = _build_analyzer(backend)
    print(f"\nAnalysing “{topic}”…\n")
    result = analyzer.run(topic, OutputContract.from_strings(depth, fmt))
    if out:
        Path(out).write_text(result.text, encoding="utf-8")
        print(f"Saved → {out}")
    else:
        print(result.text)
    if result.sources:
        print("\nSources: " + ", ".join(result.sources))


def _cmd_bootstrap(answers_path, out, backend, depth, fmt) -> None:
    """Runs the brand bootstrapper (interactive or from an answers file)."""
    from fg.brain.output_contract import OutputContract

    bs = _build_bootstrapper(backend)

    if answers_path:
        answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    else:
        print("Answer these ~10 questions (Enter to skip any):")
        answers = {}
        for q in bs.questions():
            hint = f"  [{q.hint}]" if q.hint else ""
            answers[q.id] = input(f"\n{q.prompt}{hint}\n> ").strip()

    contract = OutputContract.from_strings(depth, fmt)
    print("\nGenerating brand…\n")
    result = bs.run(answers, contract)

    if out:
        Path(out).write_text(result.text, encoding="utf-8")
        print(f"Saved → {out}")
    else:
        print(result.text)
    if result.sources:
        print("\nSources: " + ", ".join(result.sources))


def _cmd_route(query, backend) -> None:
    """Demonstrates the router: classify + dispatch across capabilities."""
    router = _build_router(backend)
    print(f"Intent: {router.classify(query).value}\n")
    result = router.route(query)
    print(result.text)
    if result.sources:
        print("\nSources: " + ", ".join(result.sources))


def main() -> None:
    """Parses args and dispatches subcommands."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="fgraph", description="FashionGraph CLI")
    parser.add_argument("--version", action="version", version=f"fg {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show resolved configuration")

    data = sub.add_parser("data", help="Data ingest pipeline").add_subparsers(
        dest="data_command"
    )
    data.add_parser("list", help="List available sources")
    build_p = data.add_parser("build", help="Build the knowledge index")
    build_p.add_argument("--source", required=True, help="Comma-separated source names")
    build_p.add_argument("--limit", type=int, default=None, help="Per-source doc cap")
    smoke_p = data.add_parser("smoke", help="Retrieval smoke test")
    smoke_p.add_argument("query", help="Query text")
    smoke_p.add_argument("-n", type=int, default=5, help="Number of results")

    boot_p = sub.add_parser("bootstrap", help="Brand Bootstrapper (AI co-founder)")
    boot_p.add_argument("--answers", default=None, help="JSON file of answers (else interactive)")
    boot_p.add_argument("--out", default=None, help="Write the brand doc to this file")
    boot_p.add_argument("--backend", default=None, help="LLM backend: ollama|openai")
    boot_p.add_argument("--depth", default="detailed", help="surface|detailed|expert")
    boot_p.add_argument("--format", default="report", help="chat|report|visual")

    an_p = sub.add_parser("analyze", help="Trend analysis on a topic")
    an_p.add_argument("topic", help="Trend / era / aesthetic, e.g. 'quiet luxury'")
    an_p.add_argument("--out", default=None, help="Write the analysis to this file")
    an_p.add_argument("--backend", default=None, help="LLM backend: ollama|openai")
    an_p.add_argument("--depth", default="detailed", help="surface|detailed|expert")
    an_p.add_argument("--format", default="report", help="chat|report|visual")

    route_p = sub.add_parser("route", help="Classify + dispatch a request")
    route_p.add_argument("query", help="Natural-language request")
    route_p.add_argument("--backend", default=None, help="LLM backend: ollama|openai")

    args = parser.parse_args()

    if args.command == "bootstrap":
        _cmd_bootstrap(args.answers, args.out, args.backend, args.depth, args.format)
    elif args.command == "analyze":
        _cmd_analyze(args.topic, args.out, args.backend, args.depth, args.format)
    elif args.command == "route":
        _cmd_route(args.query, args.backend)
    elif args.command == "data":
        if args.data_command == "list":
            _cmd_data_list()
        elif args.data_command == "build":
            _cmd_data_build(args.source, args.limit)
        elif args.data_command == "smoke":
            _cmd_data_smoke(args.query, args.n)
        else:
            parser.parse_args(["data", "--help"])
    else:
        _cmd_info()


if __name__ == "__main__":
    main()

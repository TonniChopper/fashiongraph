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
    """Builds the LLM and a (RAG + KG grounded, if possible) context builder."""
    from fg.brain.context_builder import ContextBuilder
    from fg.llm import get_llm

    llm = get_llm(backend) if backend else get_llm()
    retriever = None
    try:
        from fg.rag.retriever import FashionRetriever

        retriever = FashionRetriever()
    except Exception as exc:  # noqa: BLE001
        print(f"(note: running ungrounded — retriever unavailable: {exc})")
    kg = None
    try:
        from pathlib import Path as _P

        from fg.kg.store import KnowledgeGraph, _default_db_path

        if _P(_default_db_path()).exists():
            kg = KnowledgeGraph()
    except Exception as exc:  # noqa: BLE001
        print(f"(note: KG unavailable: {exc})")
    return llm, ContextBuilder(retriever, kg=kg)


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


def _build_look_review(backend: str | None):
    """Constructs a LookReview with whatever vision components are available.

    Uses a vision-language model so the stylist actually sees the photo. The
    perception components are assembled by ``build_perception_stack`` (one
    tested composition root); the KG is loaded once and shared with the context
    builder.
    """
    from fg.brain.context_builder import ContextBuilder
    from fg.capabilities.personal_stylist.look_review import LookReview
    from fg.llm import get_llm
    from fg.vision.perception import build_perception_stack

    try:
        llm = get_llm(backend, vision=True)   # VLM: the stylist sees the image
    except Exception as exc:  # noqa: BLE001
        print(f"(note: vision LLM unavailable, falling back to text: {exc})")
        llm = get_llm(backend)

    stack = build_perception_stack(on_note=lambda m: print(f"(note: {m})"))

    retriever = None
    try:
        from fg.rag.retriever import FashionRetriever

        retriever = FashionRetriever()
    except Exception as exc:  # noqa: BLE001
        print(f"(note: retriever unavailable: {exc})")

    ctx = ContextBuilder(retriever, kg=stack.kg)
    return LookReview(
        llm, embedder=stack.embedder, segmenter=stack.segmenter,
        visual_index=stack.visual_index, aesthetic_scorer=stack.aesthetic_scorer,
        movement_matcher=stack.movement_matcher, kg_linker=stack.kg_linker,
        runway_linker=stack.runway_linker, kg=stack.kg,
        context_builder=ctx, vision=True,
    )


def _cmd_look(image, occasion, out, backend, depth, fmt) -> None:
    """Runs a Personal Stylist look review on an image."""
    from fg.brain.output_contract import OutputContract

    reviewer = _build_look_review(backend)
    print(f"\nReviewing {image}…\n")
    result = reviewer.run(
        {"image_path": image, "occasion": occasion or ""},
        OutputContract.from_strings(depth, fmt),
    )
    if out:
        Path(out).write_text(result.text, encoding="utf-8")
        print(f"Saved → {out}")
    else:
        print(result.text)
    if result.data.get("garments"):
        print("\nDetected: " + ", ".join(result.data["garments"]))
    if result.sources:
        print("Sources: " + ", ".join(result.sources))


def _cmd_vision_build(limit) -> None:
    """Builds the product visual index (heavy — runs on the GPU)."""
    from fg.vision.embedder import FashionEmbedder
    from fg.vision.index import build_product_index

    print("Loading fashion embedder (Marqo-FashionSigLIP)…")
    embedder = FashionEmbedder()
    print(f"Building visual index{f' (limit {limit})' if limit else ''}…")
    path = build_product_index(embedder, limit=limit)
    print(f"Saved visual index → {path}")


def _cmd_vision_build_runway(limit) -> None:
    """Builds the runway visual index from labeled Vogue imagery."""
    from fg.vision.embedder import FashionEmbedder
    from fg.vision.runway import build_runway_index

    print("Loading fashion embedder (Marqo-FashionSigLIP)…")
    embedder = FashionEmbedder()
    print(f"Building runway index{f' (limit {limit})' if limit else ''}…")
    path = build_runway_index(embedder, limit=limit)
    print(f"Saved runway index → {path}")


def _cmd_vision_extract_runway(per_collection, limit, backend) -> None:
    """Runs the VLM over runway looks → captions + image-grounded KG edges."""
    from fg.kg.store import KnowledgeGraph
    from fg.llm import get_llm
    from fg.vision.vlm_extract import extract_runway_kg

    llm = get_llm(backend, vision=True) if backend else get_llm(vision=True)
    kg = KnowledgeGraph()
    print("Running VLM over sampled runway looks (image-grounded extraction)…")
    stats = extract_runway_kg(llm, kg, per_collection=per_collection, limit=limit,
                              on_note=lambda m: print(f"  {m}"))
    print("\nVLM extraction summary:")
    for k, v in stats.as_dict().items():
        print(f"  {k}: {v}")


def _cmd_vision_build_textures(directory, limit) -> None:
    """Builds the fabric-texture visual index from a folder-per-fabric dataset."""
    from fg.vision.embedder import FashionEmbedder
    from fg.vision.fabric_texture import build_texture_index

    print("Loading fashion embedder (Marqo-FashionSigLIP)…")
    embedder = FashionEmbedder()
    print(f"Building fabric-texture index from {directory}…")
    path = build_texture_index(embedder, directory, limit=limit)
    print(f"Saved fabric-texture index → {path}")


def _cmd_vision_eval_runway(holdout, neighbors) -> None:
    """Runs held-out designer top-k accuracy on the runway index (no model)."""
    from fg.vision.index import VisualIndex
    from fg.vision.runway import _default_runway_index_path
    from fg.vision.runway_eval import evaluate_designer_topk

    index = VisualIndex.load(_default_runway_index_path())
    for split in ("image", "collection"):
        res = evaluate_designer_topk(index, holdout_frac=holdout,
                                     neighbors=neighbors, split_by=split)
        label = "by-image (leaky)" if split == "image" else "by-collection (honest)"
        print(f"\nRunway grounding [{label}] — {res['n_test']} held-out looks, "
              f"{res['n_designers']} designers:")
        print(f"  top-1: {res['top1']:.3f}   top-3: {res['top3']:.3f}   "
              f"top-5: {res['top5']:.3f}   (random top-1: {res['random_top1']:.3f})")


def _cmd_kg_build(source, limit, backend) -> None:
    """Builds the knowledge graph from a corpus source."""
    from fg.kg.build import build_kg
    from fg.llm import get_llm

    llm = get_llm(backend) if backend else get_llm()
    stats = build_kg(llm, source=source, limit=limit)
    print("\nKG build summary:")
    for k, v in stats.as_dict().items():
        print(f"  {k}: {v}")


def _cmd_kg_query(entity) -> None:
    """Prints the graph facts connected to an entity."""
    from fg.kg.store import KnowledgeGraph

    kg = KnowledgeGraph()
    facts = kg.facts_as_text(entity, limit=50)
    if not facts:
        print(f"No facts for {entity!r}. Is the KG built? Try `fgraph kg stats`.")
        return
    print(f"\nFacts connected to {entity!r}:\n")
    for f in facts:
        print(f"  • {f}")


def _cmd_kg_add_fabrics() -> None:
    """Loads the curated fabric ontology into the KG."""
    from fg.kg.fabrics import FABRICS, add_fabrics_to_kg
    from fg.kg.store import KnowledgeGraph

    added = add_fabrics_to_kg(KnowledgeGraph())
    print(f"Added {added} fabric property/texture/season edges for {len(FABRICS)} fabrics.")


def _cmd_kg_stats() -> None:
    """Prints KG summary statistics."""
    from fg.kg.store import KnowledgeGraph

    s = KnowledgeGraph().stats()
    print(f"\nKnowledge graph: {s['triples']} triples, {s['entities']} entities")
    print("Relations:")
    for rel, c in s["relations"].items():
        print(f"  {rel:20s} {c}")


def _cmd_kg_path(src, dst, hops) -> None:
    """Finds relationship paths between two entities (multi-hop reasoning)."""
    from fg.kg.reasoning import GraphReasoner, format_path
    from fg.kg.store import KnowledgeGraph

    reasoner = GraphReasoner(KnowledgeGraph())
    paths = reasoner.paths(src, dst, max_hops=hops)
    if not paths:
        print(f"No path (≤{hops} hops) between {src!r} and {dst!r}.")
        return
    print(f"\nPaths from {src!r} to {dst!r}:\n")
    for p in paths[:10]:
        print(f"  {format_path(p)}")


def _cmd_kg_who(relation, obj) -> None:
    """Answers a one-hop relational filter: who <relation> <object>."""
    from fg.kg.reasoning import GraphReasoner
    from fg.kg.store import KnowledgeGraph

    subs = GraphReasoner(KnowledgeGraph()).subjects_with(relation, obj)
    if not subs:
        print(f"Nothing matches {relation} → {obj!r}.")
        return
    print(f"\nEntities where {relation.replace('_', ' ')} → {obj!r}:\n")
    for s in subs:
        print(f"  • {s}")


def _cmd_kg_predict(entity, k, add, backend) -> None:
    """Predicts plausible missing edges for an entity (one-shot ICL)."""
    from fg.kg.link_prediction import predict_links
    from fg.kg.store import KnowledgeGraph
    from fg.llm import get_llm

    llm = get_llm(backend) if backend else get_llm()
    kg = KnowledgeGraph()
    preds = predict_links(entity, kg, llm, k=k)
    if not preds:
        print(f"No predictions for {entity!r} (is the KG built?).")
        return
    print(f"\nPredicted missing facts for {entity!r}:\n")
    for t in preds:
        print(f"  ? {t.subject} —{t.relation.replace('_',' ')}→ {t.object}")
    if add:
        n = kg.add_triples(preds)
        print(f"\nAdded {n} predicted edges (source=llm_predicted).")
    else:
        print("\n(use --add to insert these as source=llm_predicted)")


def _cmd_kg_eval(n, backend, judge) -> None:
    """Runs the KG-vs-flat-RAG lift experiment."""
    from fg.kg.evaluate import evaluate_lift
    from fg.kg.store import KnowledgeGraph

    llm, ctx = _build_llm_and_context(backend)
    retriever = getattr(ctx, "retriever", None)
    extra = " + LLM judge" if judge else ""
    print(f"\nRunning KG-vs-RAG lift eval{extra} (calls the LLM per entity)…\n")
    results, summary = evaluate_lift(
        llm, retriever, KnowledgeGraph(), n_entities=n, judge=judge
    )
    head = f"{'entity':22s} {'gold':>4s} {'KG':>6s} {'RAG':>6s}"
    if judge:
        head += "  judge"
    print(head)
    for r in results:
        line = f"{r.entity[:22]:22s} {r.n_gold:>4d} {r.coverage_kg:>6.2f} {r.coverage_rag:>6.2f}"
        if judge:
            line += f"  {r.judge}"
        print(line)
    print(f"\nmean fact-coverage — KG: {summary['mean_coverage_kg']:.3f}  "
          f"RAG: {summary['mean_coverage_rag']:.3f}  "
          f"LIFT: {summary['lift']:+.3f}")
    if judge:
        print(f"LLM judge — KG wins: {summary['judge_kg_wins']}  "
              f"RAG wins: {summary['judge_rag_wins']}  "
              f"ties: {summary['judge_ties']}")


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

    look_p = sub.add_parser("look", help="Personal Stylist look review (image in)")
    look_p.add_argument("image", help="Path to an outfit photo")
    look_p.add_argument("--occasion", default=None, help="Occasion/context, e.g. 'wedding'")
    look_p.add_argument("--out", default=None, help="Write the review to this file")
    look_p.add_argument("--backend", default=None, help="LLM backend: ollama|openai")
    look_p.add_argument("--depth", default="detailed", help="surface|detailed|expert")
    look_p.add_argument("--format", default="report", help="chat|report|visual")

    vision = sub.add_parser("vision", help="Visual index tools").add_subparsers(
        dest="vision_command"
    )
    vbuild = vision.add_parser("build", help="Build the product visual index")
    vbuild.add_argument("--limit", type=int, default=None, help="Cap images (quick build)")
    vrun = vision.add_parser("build-runway", help="Build the runway visual index (Vogue)")
    vrun.add_argument("--limit", type=int, default=None, help="Cap images (quick build)")
    veval = vision.add_parser("eval-runway", help="Designer top-k grounding accuracy")
    veval.add_argument("--holdout", type=float, default=0.2, help="Held-out fraction")
    veval.add_argument("--neighbors", type=int, default=10, help="kNN neighbours to vote")
    vtex = vision.add_parser("build-textures", help="Build fabric-texture index (folder-per-fabric)")
    vtex.add_argument("directory", help="Root dir laid out <fabric>/*.jpg")
    vtex.add_argument("--limit", type=int, default=None, help="Cap images")
    vext = vision.add_parser("extract-runway", help="VLM → runway captions + KG edges")
    vext.add_argument("--per-collection", type=int, default=3, help="Looks sampled per collection")
    vext.add_argument("--limit", type=int, default=None, help="Cap total looks")
    vext.add_argument("--backend", default=None, help="LLM backend: ollama|openai")

    kg = sub.add_parser("kg", help="Knowledge graph tools").add_subparsers(dest="kg_command")
    kgb = kg.add_parser("build", help="Extract triples from the corpus into the KG")
    kgb.add_argument("--source", default="wikipedia", help="Ingest source (default: wikipedia)")
    kgb.add_argument("--limit", type=int, default=15, help="Docs to process (narrow slice)")
    kgb.add_argument("--backend", default=None, help="LLM backend: ollama|openai")
    kgq = kg.add_parser("query", help="Show facts connected to an entity")
    kgq.add_argument("entity", help="Entity name, e.g. 'Prada'")
    kg.add_parser("stats", help="KG summary statistics")
    kg.add_parser("add-fabrics", help="Add curated fabric properties to the KG")
    kge = kg.add_parser("eval", help="KG-vs-flat-RAG lift experiment")
    kge.add_argument("-n", type=int, default=8, help="Entities to test")
    kge.add_argument("--backend", default=None, help="LLM backend: ollama|openai")
    kge.add_argument("--judge", action="store_true", help="Add LLM-as-judge quality comparison")
    kgp = kg.add_parser("path", help="Find relationship paths between two entities")
    kgp.add_argument("src", help="Start entity")
    kgp.add_argument("dst", help="Target entity")
    kgp.add_argument("--hops", type=int, default=3, help="Max path length")
    kgw = kg.add_parser("who", help="One-hop filter: who <relation> <object>")
    kgw.add_argument("relation", help="Relation, e.g. based_in")
    kgw.add_argument("object", help="Object entity, e.g. Milan")
    kgpr = kg.add_parser("predict", help="Predict missing edges (one-shot ICL)")
    kgpr.add_argument("entity", help="Entity to predict links for")
    kgpr.add_argument("-k", type=int, default=5, help="Number of predictions")
    kgpr.add_argument("--add", action="store_true", help="Insert predictions (source=llm_predicted)")
    kgpr.add_argument("--backend", default=None, help="LLM backend: ollama|openai")

    route_p = sub.add_parser("route", help="Classify + dispatch a request")
    route_p.add_argument("query", help="Natural-language request")
    route_p.add_argument("--backend", default=None, help="LLM backend: ollama|openai")

    args = parser.parse_args()

    if args.command == "bootstrap":
        _cmd_bootstrap(args.answers, args.out, args.backend, args.depth, args.format)
    elif args.command == "analyze":
        _cmd_analyze(args.topic, args.out, args.backend, args.depth, args.format)
    elif args.command == "look":
        _cmd_look(args.image, args.occasion, args.out, args.backend, args.depth, args.format)
    elif args.command == "vision":
        if args.vision_command == "build":
            _cmd_vision_build(args.limit)
        elif args.vision_command == "build-runway":
            _cmd_vision_build_runway(args.limit)
        elif args.vision_command == "build-textures":
            _cmd_vision_build_textures(args.directory, args.limit)
        elif args.vision_command == "eval-runway":
            _cmd_vision_eval_runway(args.holdout, args.neighbors)
        elif args.vision_command == "extract-runway":
            _cmd_vision_extract_runway(args.per_collection, args.limit, args.backend)
        else:
            parser.parse_args(["vision", "--help"])
    elif args.command == "kg":
        if args.kg_command == "build":
            _cmd_kg_build(args.source, args.limit, args.backend)
        elif args.kg_command == "query":
            _cmd_kg_query(args.entity)
        elif args.kg_command == "stats":
            _cmd_kg_stats()
        elif args.kg_command == "add-fabrics":
            _cmd_kg_add_fabrics()
        elif args.kg_command == "eval":
            _cmd_kg_eval(args.n, args.backend, args.judge)
        elif args.kg_command == "path":
            _cmd_kg_path(args.src, args.dst, args.hops)
        elif args.kg_command == "who":
            _cmd_kg_who(args.relation, args.object)
        elif args.kg_command == "predict":
            _cmd_kg_predict(args.entity, args.k, args.add, args.backend)
        else:
            parser.parse_args(["kg", "--help"])
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

"""KG-vs-flat-RAG lift evaluation — the thesis result.

Tests the council's demand: does structured knowledge-graph grounding actually
beat flat vector RAG? For each of the richest KG entities we ask the same
relational question two ways — once with KG facts injected, once with RAG
passages only — and measure how many of the entity's known relational facts
each answer surfaces (fact coverage).

Honest caveat (state it in the thesis): KG facts are extracted from the same
corpus RAG retrieves over, so coverage is a *recall of relational facts* metric,
not proof of correctness. It measures whether structured injection surfaces
relational facts that flat retrieval misses — a real, reportable signal, not the
whole story. Pair it with the optional LLM-judge for a quality read.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from fg.brain.context_builder import ContextBuilder
from fg.kg.store import KnowledgeGraph
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)


def fact_coverage(answer: str, gold_terms: list[str]) -> float:
    """Fraction of *gold_terms* mentioned (case-insensitively) in *answer*.

    Args:
        answer: The generated answer text.
        gold_terms: Object surface strings the answer ideally surfaces.

    Returns:
        Coverage in ``[0, 1]`` (``0.0`` if no gold terms).
    """
    if not gold_terms:
        return 0.0
    low = answer.lower()
    hit = sum(1 for t in gold_terms if t and t.lower() in low)
    return hit / len(gold_terms)


@dataclass
class LiftResult:
    """Per-entity comparison of KG-grounded vs RAG-only answers.

    Attributes:
        entity: The entity tested.
        n_gold: Number of gold relational facts.
        coverage_kg: Fact coverage of the KG-grounded answer.
        coverage_rag: Fact coverage of the RAG-only answer.
        judge: Independent LLM verdict — ``"kg"``, ``"rag"``, ``"tie"``, or
            ``""`` if judging was off.
    """

    entity: str
    n_gold: int
    coverage_kg: float
    coverage_rag: float
    judge: str = ""


def parse_judge_verdict(raw: str) -> str:
    """Parses a judge reply into ``"A"``, ``"B"``, or ``"tie"``.

    Prefers an explicit ``VERDICT: X`` line (the judge is asked to end with one
    after brief reasoning); falls back to a whole-string scan.

    Args:
        raw: The judge model's reply.

    Returns:
        ``"A"``, ``"B"``, or ``"tie"`` (defaults to ``"tie"`` when unclear).
    """
    if not raw:
        return "tie"
    up = raw.strip().upper()

    # Prefer the explicit verdict line if present.
    if "VERDICT" in up:
        tail = up.split("VERDICT", 1)[1]
        if "TIE" in tail or "EQUAL" in tail:
            return "tie"
        for ch in tail:
            if ch == "A":
                return "A"
            if ch == "B":
                return "B"

    # Fallback: short replies like "A", "B", "TIE".
    if up.startswith("TIE") or up == "EQUAL":
        return "tie"
    if up.startswith("A"):
        return "A"
    if up.startswith("B"):
        return "B"
    return "tie"


def judge_pair(llm: LLM, question: str, answer_a: str, answer_b: str) -> str:
    """Asks the LLM which answer is more factually complete/specific.

    Args:
        llm: Judge LLM.
        question: The question both answers address.
        answer_a: Candidate A.
        answer_b: Candidate B.

    Returns:
        ``"A"``, ``"B"``, or ``"tie"``.
    """
    system = (
        "You are a strict, decisive judge comparing two answers to a fashion "
        "question. The better answer states MORE specific, correct facts "
        "(names, places, materials, influences). Count the concrete facts in "
        "each. Choose TIE ONLY if they are genuinely indistinguishable in "
        "factual content — this should be rare; commit to a winner otherwise. "
        "Reason in one sentence, then end with a line exactly: 'VERDICT: A', "
        "'VERDICT: B', or 'VERDICT: TIE'."
    )
    user = (
        f"Question: {question}\n\n"
        f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}\n\n"
        "One sentence of reasoning, then the VERDICT line:"
    )
    try:
        raw = llm.chat([Message("system", system), Message("user", user)], max_tokens=120)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge call failed (%s).", exc)
        return "tie"
    return parse_judge_verdict(raw)


def _answer(llm: LLM, question: str, knowledge: str) -> str:
    """Generates a grounded answer from a knowledge block."""
    system = (
        "You are a fashion expert. Answer using ONLY the provided knowledge. "
        "Be specific and factual; if the knowledge lacks something, omit it."
    )
    user = f"Knowledge:\n{knowledge or '(none)'}\n\nQuestion: {question}"
    return llm.chat([Message("system", system), Message("user", user)], max_tokens=350)


def evaluate_lift(
    llm: LLM,
    retriever,
    kg: KnowledgeGraph,
    n_entities: int = 8,
    judge: bool = False,
    seed: int = 42,
) -> tuple[list[LiftResult], dict]:
    """Runs the KG-vs-RAG lift experiment.

    Args:
        llm: LLM backend.
        retriever: A ``FashionRetriever`` (flat RAG); may be ``None``.
        kg: The knowledge graph.
        n_entities: How many top entities to test.
        judge: If ``True``, also run an independent LLM-as-judge quality
            comparison (anonymised, position-bias randomised).
        seed: RNG seed for the judge's A/B ordering.

    Returns:
        ``(per_entity_results, summary)``. Summary has mean coverages, the
        coverage lift, and — if judged — win tallies.
    """
    kg_builder = ContextBuilder(retriever, kg=kg)
    rag_builder = ContextBuilder(retriever, kg=None)
    rng = random.Random(seed)

    results: list[LiftResult] = []
    for display, key, _ in kg.top_subjects(n_entities):
        facts = kg.outgoing(display)
        gold = [f["object"] for f in facts]
        if not gold:
            continue
        question = (
            f"Give the key facts about {display}: who founded it, where it is "
            f"based, its creative director, materials or silhouettes it uses, "
            f"influences, and what it is known for."
        )
        ctx_kg = kg_builder.build(f"{display} {question}")
        ctx_rag = rag_builder.build(f"{display} {question}")
        ans_kg = _answer(llm, question, ctx_kg.knowledge_block())
        ans_rag = _answer(llm, question, ctx_rag.knowledge_block())

        verdict = ""
        if judge:
            # Randomise which answer is A to neutralise position bias.
            kg_is_a = rng.random() < 0.5
            a, b = (ans_kg, ans_rag) if kg_is_a else (ans_rag, ans_kg)
            raw = judge_pair(llm, question, a, b)
            if raw == "tie":
                verdict = "tie"
            elif (raw == "A") == kg_is_a:
                verdict = "kg"
            else:
                verdict = "rag"

        results.append(LiftResult(
            entity=display, n_gold=len(gold),
            coverage_kg=round(fact_coverage(ans_kg, gold), 3),
            coverage_rag=round(fact_coverage(ans_rag, gold), 3),
            judge=verdict,
        ))
        logger.info("eval %s: KG=%.2f RAG=%.2f judge=%s (n_gold=%d)",
                    display, results[-1].coverage_kg, results[-1].coverage_rag,
                    verdict or "-", len(gold))

    if results:
        mk = sum(r.coverage_kg for r in results) / len(results)
        mr = sum(r.coverage_rag for r in results) / len(results)
    else:
        mk = mr = 0.0
    summary = {
        "entities_tested": len(results),
        "mean_coverage_kg": round(mk, 3),
        "mean_coverage_rag": round(mr, 3),
        "lift": round(mk - mr, 3),
    }
    if judge:
        summary["judge_kg_wins"] = sum(1 for r in results if r.judge == "kg")
        summary["judge_rag_wins"] = sum(1 for r in results if r.judge == "rag")
        summary["judge_ties"] = sum(1 for r in results if r.judge == "tie")
    return results, summary

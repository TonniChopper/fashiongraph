"""Shared prompt fragments for capabilities.

Centralised so every capability enforces the same factual discipline. The
grounding rule directly targets the failure we saw: a small model confidently
naming the wrong brands ("Gucci under Michele = quiet luxury") when the
retrieved context had no brand-level facts to anchor on.
"""

from __future__ import annotations

#: Appended to every capability's system prompt to curb hallucinated specifics.
GROUNDING_DISCIPLINE: str = (
    "Grounding discipline: Prefer facts supported by the provided knowledge "
    "context. You may name a brand, designer, or person only if it appears in "
    "that context OR is a well-established, uncontroversial association you are "
    "highly confident about; if you are inferring beyond the context, append "
    "'(inferred)'. Never invent social-media handles, statistics, dates, or "
    "quotes. If the context lacks the specifics a section needs, describe the "
    "archetype (the kind of brand/customer/reference) rather than guessing a "
    "name. It is better to be accurate and slightly general than specific and wrong."
)

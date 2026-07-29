"""Tests for the ReAct agent + web-search helpers (no network, stub LLM)."""

from fg.brain.agent import ReActAgent
from fg.tools.web_search import _chunks, _clean_html


class _StubLLM:
    """Scripted LLM: emits the given replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else "Final: done."

    def complete(self, *a, **k):
        return ""


class _FakeKG:
    def facts_as_text(self, entity, limit=20):
        return ["Dior founded_by Christian Dior", "Dior known_for New Look"]


def test_clean_html_strips_tags_and_scripts():
    out = _clean_html("<p>Dior <b>SS26</b></p><script>x=1</script> minimal")
    assert "Dior" in out and "SS26" in out and "x=1" not in out


def test_chunks_caps_windows():
    assert len(_chunks(" ".join(f"w{i}" for i in range(900)), size=350, max_chunks=2)) == 2


def test_agent_uses_kg_then_answers():
    llm = _StubLLM([
        "Thought: check the graph.\nAction: kg[Dior]",
        "Final: Dior was founded by Christian Dior.",
    ])
    res = ReActAgent(llm, kg=_FakeKG(), max_steps=4).run("Who founded Dior?")
    assert "Christian Dior" in res.answer
    assert res.steps == 2
    assert res.trace and res.trace[0].startswith("kg[Dior]")


def test_agent_direct_final_no_tool():
    llm = _StubLLM(["Final: Beige is a neutral tone."])
    res = ReActAgent(llm, kg=_FakeKG()).run("Is beige neutral?")
    assert "neutral" in res.answer.lower()
    assert res.steps == 1
    assert res.trace == []


def test_agent_forces_final_when_out_of_steps():
    # Always tries to act → never gives Final → loop must force an answer.
    llm = _StubLLM(["Action: kg[X]"] * 3 + ["Final: forced answer."])
    res = ReActAgent(llm, kg=_FakeKG(), max_steps=2).run("loop?")
    assert res.steps == 2
    assert "forced answer" in res.answer or res.answer  # non-empty


class _FakeIndexer:
    def __init__(self):
        self.docs = []

    def add_document(self, text, metadata):
        self.docs.append(metadata)
        return 1


def test_agent_ingest_collected_indexes():
    idx = _FakeIndexer()
    agent = ReActAgent(_StubLLM([]), kg=None, indexer=idx)
    agent._collected = [{"title": "t", "url": "http://x",
                         "snippet": "a fashion snippet with plenty of words " * 4}]
    stats = agent.ingest_collected(fetch_pages=False)   # snippets only → no network
    assert stats["chunks_indexed"] == 1
    assert idx.docs[0]["source"] == "web" and idx.docs[0]["url"] == "http://x"


def test_agent_ingest_noop_when_nothing_collected():
    assert ReActAgent(_StubLLM([]), kg=_FakeKG()).ingest_collected() == {}


def test_agent_dispatches_capability_tool():
    # step 1 → style tool (grounded LLM call), step 2 → final.
    llm = _StubLLM([
        "Thought: styling task.\nAction: style[minimalist autumn outfit]",
        "Wear a charcoal overcoat over black knit and trousers.",  # tool's grounded reply
        "Final: Here's a clean minimalist autumn look: charcoal overcoat, black knit.",
    ])
    res = ReActAgent(llm, kg=_FakeKG(), max_steps=3).run("style me for autumn")
    assert "charcoal" in res.answer.lower() or "minimalist" in res.answer.lower()
    assert res.trace and res.trace[0].startswith("style[")


def test_agent_injects_selection_context():
    seen = {}

    class CapturingLLM:
        def chat(self, messages, **kw):
            seen["user"] = messages[-1].content
            return "Final: reviewed."
        def complete(self, *a, **k):
            return ""

    ReActAgent(CapturingLLM(), kg=_FakeKG()).run(
        "make it less formal", context="- (look) charcoal suit, white shirt")
    assert "Canvas selection" in seen["user"]
    assert "charcoal suit" in seen["user"] and "make it less formal" in seen["user"]


def test_agent_ignores_unknown_tool_as_final():
    # An unknown tool name is not a real action → treated as the answer.
    llm = _StubLLM(["Action: teleport[home]  — actually, beige is neutral."])
    res = ReActAgent(llm, kg=_FakeKG(), max_steps=2).run("?")
    assert "beige" in res.answer.lower()
    assert res.trace == []

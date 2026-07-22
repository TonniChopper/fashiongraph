"""Tests for multimodal (image) support in the LLM interface + look review."""

from fg.brain.context_builder import ContextBuilder
from fg.capabilities.personal_stylist.look_review import LookReview
from fg.llm.base import Message, encode_image


def test_encode_image_from_bytes():
    b64 = encode_image(b"\xff\xd8\xff\xe0jpegbytes")
    assert isinstance(b64, str) and len(b64) > 0


def test_message_as_dict_includes_images():
    m = Message("user", "what is this?", images=["BASE64DATA"])
    d = m.as_dict()
    assert d["images"] == ["BASE64DATA"]
    assert Message("user", "hi").as_dict().get("images") is None  # omitted when empty


def test_ollama_payload_carries_images(monkeypatch):
    from fg.llm import ollama_backend

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "a beret and wide-leg trousers"}}

    def _post(url, json=None, timeout=None):
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(ollama_backend.requests, "post", _post)
    llm = ollama_backend.OllamaLLM(model="qwen2.5vl:7b", host="http://x:1")
    out = llm.chat([Message("user", "describe", images=["IMGB64"])])
    assert out.startswith("a beret")
    assert captured["payload"]["messages"][0]["images"] == ["IMGB64"]


# ---- look review vision wiring ----

class _CapturingLLM:
    model = "vlm"

    def __init__(self):
        self.last_messages = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Silhouette & proportion\nA cream beret, knit vest, wide-leg trousers."


class _FakeImage:
    def convert(self, mode):
        return self

    def save(self, buf, format="JPEG"):
        buf.write(b"jpegdata")


def test_look_review_attaches_image_in_vision_mode():
    reviewer = LookReview(_CapturingLLM(), context_builder=ContextBuilder(None), vision=True)
    reviewer.review(_FakeImage(), occasion="gallery opening")
    system, user = reviewer.llm.last_messages
    assert user.images and len(user.images[0]) > 0     # photo attached
    assert "SEE the outfit photo" in system.content     # told to trust its eyes


def test_look_review_no_image_when_vision_off():
    reviewer = LookReview(_CapturingLLM(), context_builder=ContextBuilder(None), vision=False)
    reviewer.review(_FakeImage(), occasion="")
    _, user = reviewer.llm.last_messages
    assert not user.images

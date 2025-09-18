import sys
import textwrap
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import triage
from triage import (
    _has_reply_cue,
    _looks_like_marketing,
    _safe_load_json,
    answer_question,
    classify,
)


def test_safe_load_json_handles_multiline_rationale():
    raw = textwrap.dedent(
        """
        {
          "importance": false,
          "importance_score": 0.1,
          "reply_needed": false,
          "reply_needed_score": 0.1,
          "rationale": "Line one
continues on the next line
and mentions 'Lounge' explicitly."
        }
        """
    )

    data = _safe_load_json(raw)

    assert data["importance"] is False
    assert data["reply_needed"] is False
    assert data["importance_score"] == 0.1
    assert data["reply_needed_score"] == 0.1
    assert "\n" in data["rationale"]
    assert "Lounge" in data["rationale"]


def test_has_reply_cue_filters_generic_marketing_question():
    marketing_email = """Subject: Weekend Deals\n\nAre you ready for a huge sale?\nDon't miss out!"""

    assert _has_reply_cue(marketing_email) is False


def test_has_reply_cue_recognizes_direct_request_question():
    request_email = """Subject: Project Update\n\nAre you available for a quick sync tomorrow at 3?"""

    assert _has_reply_cue(request_email) is True


def test_has_reply_cue_ignores_marketing_would_you_like_question():
    marketing_email = (
        "Subject: Upgrade and Save\n\nWould you like to upgrade your plan and save 20% today?"
    )

    assert _has_reply_cue(marketing_email) is False


def test_looks_like_marketing_detects_roundup_subject():
    email_text = (
        "From: Deals Newsletter <deals@example.com>\n"
        "Subject: Weekly Digest: Top Stories and Offers\n\n"
        "Here are the latest deals you might enjoy."
    )

    assert _looks_like_marketing(email_text) is True


def test_classify_handles_empty_model_response(monkeypatch):
    class DummyResponse:
        text = ""
        candidates: list[dict[str, str]] = []

    class DummyModel:
        def generate_content(self, *args, **kwargs):  # pragma: no cover - trivial
            return DummyResponse()

    monkeypatch.setattr(triage, "get_classifier_model", lambda: DummyModel())

    email_text = (
        "From: Promotions <promo@example.com>\n"
        "Subject: Weekly Deals\n\n"
        "Don't miss our sale."
    )

    result = classify(email_text)

    assert result["importance"] is False
    assert result["reply_needed"] is False
    assert result["rationale"] == "Model response was empty."


def test_answer_question_handles_text_accessor_value_error(monkeypatch):
    class DummyCandidate:
        finish_reason = None

        class DummyContent:
            parts = [{"text": "Fallback answer"}]

        content = DummyContent()

    class DummyResponse:
        candidates = [DummyCandidate()]

        @property
        def text(self):  # pragma: no cover - accessed indirectly
            raise ValueError("No quick text available")

    class DummyModel:
        def generate_content(self, *args, **kwargs):  # pragma: no cover - trivial
            return DummyResponse()

    monkeypatch.setattr(triage, "get_qa_model", lambda: DummyModel())

    answer = answer_question("Context", "What is up?")

    assert answer == "Fallback answer"


def test_answer_question_defaults_when_no_text(monkeypatch):
    class DummyResponse:
        candidates: list[object] = []

        @property
        def text(self):  # pragma: no cover - accessed indirectly
            raise ValueError("No text available")

    class DummyModel:
        def generate_content(self, *args, **kwargs):  # pragma: no cover - trivial
            return DummyResponse()

    monkeypatch.setattr(triage, "get_qa_model", lambda: DummyModel())

    answer = answer_question("Context", "Tell me something")

    assert answer == "I'm not sure."

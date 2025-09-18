import json
import sys
import textwrap
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import triage
from triage import (
    _has_reply_cue,
    _looks_like_marketing,
    _mentions_user_name,
    _refresh_owner_context,
    _safe_load_json,
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


def test_looks_like_marketing_detects_advertisement_disclaimer():
    email_text = (
        "From: Promo Alerts <ads@example.com>\n"
        "Subject: Member exclusive offer just for you\n\n"
        "This email is an advertisement from Promo Alerts.\n"
        "You are receiving this email because you subscribed to our deals."
    )

    assert _looks_like_marketing(email_text) is True


def test_looks_like_marketing_detects_sponsored_language():
    email_text = (
        "From: Travel Partners <hello@example.com>\n"
        "Subject: A paid partnership highlight\n\n"
        "Enjoy this sponsored message from our brand partners about upcoming getaways."
    )

    assert _looks_like_marketing(email_text) is True


def test_looks_like_marketing_flags_newsletter_issue_header():
    email_text = textwrap.dedent(
        """
        From: Jory at Font Awesome <jory@m.fontawesome.com>
        Subject: Awesome News - Icon Puzzle, More Cowbell, Frame of Preference, Notdog

        ISSUE 003 SEPTEMBER 2025 Hello folks! Welcome back! As the temperature slowly cools in New England
        and the leaves start their march towards red and gold, we're excited to be entering: sweata weatha
        """
    ).strip()

    assert _looks_like_marketing(email_text) is True
    assert _has_reply_cue(email_text) is False


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


def test_mentions_user_name_detects_alias(monkeypatch):
    monkeypatch.setenv("INBOX_OWNER_NAME", "Alex Johnson")
    monkeypatch.setenv("INBOX_OWNER_ALIASES", "AJ,Alex J.")
    _refresh_owner_context()
    try:
        personal_email = (
            "From: Teammate <teammate@example.com>\n"
            "Subject: Need a quick review\n\n"
            "Hi Alex, could you look over the deck today?"
        )
        assert _mentions_user_name(personal_email) is True

        generic_email = (
            "From: Updates <updates@example.com>\n"
            "Subject: Weekly Digest\n\n"
            "Hello team, here is the weekly summary."
        )
        assert _mentions_user_name(generic_email) is False
    finally:
        monkeypatch.delenv("INBOX_OWNER_NAME", raising=False)
        monkeypatch.delenv("INBOX_OWNER_ALIASES", raising=False)
        _refresh_owner_context()


def test_classify_promotes_reply_when_model_signals_and_name_present(monkeypatch):
    monkeypatch.setenv("INBOX_OWNER_NAME", "Alex")
    _refresh_owner_context()

    class DummyResponse:
        def __init__(self, text: str):
            self.text = text
            self.candidates: list[dict[str, str]] = []

    class DummyModel:
        def generate_content(self, *args, **kwargs):  # pragma: no cover - simple stub
            payload = json.dumps(
                {
                    "importance": False,
                    "importance_score": 0.42,
                    "reply_needed": True,
                    "reply_needed_score": 0.52,
                    "rationale": "Sender asked Alex for a deliverable.",
                }
            )
            return DummyResponse(payload)

    monkeypatch.setattr(triage, "get_classifier_model", lambda: DummyModel())

    email_text = (
        "From: Erin <erin@example.com>\n"
        "Subject: Status update\n\n"
        "Hi Alex, can you send the signed contract today?"
    )

    try:
        result = classify(email_text)
    finally:
        monkeypatch.delenv("INBOX_OWNER_NAME", raising=False)
        _refresh_owner_context()

    assert result["reply_needed"] is True
    assert result["importance"] is True
    assert result["reply_needed_score"] >= 0.7
    assert result["importance_score"] >= 0.6

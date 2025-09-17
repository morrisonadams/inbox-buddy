import sys
import textwrap
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from triage import _has_reply_cue, _safe_load_json


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

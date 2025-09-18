import base64
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from gmail_client import extract_payload  # noqa: E402


def _encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")


def test_extract_payload_converts_html_body_to_text():
    html_body = (
        "<div>Project Update</div>"
        "<p>Hey there,<br>We need a quick sync tomorrow.</p>"
        "<ul><li>Review the deck</li><li>Confirm availability</li></ul>"
    )
    message = {
        "snippet": "Project Update — Hey there",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Project Update"},
                {"name": "From", "value": "Alice <alice@example.com>"},
            ],
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {"data": _encode(html_body)},
                }
            ],
        },
    }

    payload = extract_payload(message)

    assert payload["subject"] == "Project Update"
    assert payload["sender"] == "Alice <alice@example.com>"
    assert "Project Update" in payload["body"]
    assert "We need a quick sync tomorrow." in payload["body"]
    assert "- Review the deck" in payload["body"]
    assert "- Confirm availability" in payload["body"]


def test_extract_payload_prefers_plain_text_when_available():
    plain_text = "Plain line one\nLine two"
    html_text = "<p>Plain line one</p><p>Different html</p>"
    message = {
        "snippet": "Plain line one",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Status"},
                {"name": "From", "value": "Bob <bob@example.com>"},
            ],
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _encode(plain_text)},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _encode(html_text)},
                },
            ],
        },
    }

    payload = extract_payload(message)

    assert payload["body"] == plain_text

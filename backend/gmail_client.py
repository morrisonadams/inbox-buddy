import os
import base64
import re
import time
from html import unescape
from typing import Dict, Optional, Tuple

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _decode_body_data(data: Optional[str]) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _html_to_text(html: str) -> str:
    if not html:
        return ""

    # Remove scripts/styles first to avoid noise in the output.
    cleaned = re.sub(r"(?is)<(script|style)[^>]*>.*?</\\1>", "", html)

    replacements = {
        r"(?is)<br\s*/?>": "\n",
        r"(?is)</(p|div|section|article|h[1-6]|tr)>": "\n",
        r"(?is)<(p|div|section|article|h[1-6]|tr)[^>]*>": "\n",
        r"(?is)<li[^>]*>": "\n- ",
        r"(?is)</li>": "\n",
        r"(?is)</?(table|tbody|thead|tfoot)>": "\n",
    }

    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = re.sub(r"(?is)<[^>]+>", "", cleaned)
    cleaned = unescape(cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\t\u00a0]+", " ", cleaned)
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_body(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class AuthRequired(Exception):
    """Raised when Gmail access requires user interaction."""


# state -> (flow, created_at)
_pending_flows: Dict[str, Tuple[InstalledAppFlow, float]] = {}
_FLOW_TTL_SECONDS = 15 * 60


def _get_paths():
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "/app/token.json")
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "/app/credentials.json")
    return token_path, creds_path


def ensure_auth():
    token_path, creds_path = _get_paths()

    if not os.path.exists(creds_path):
        raise AuthRequired("Gmail credentials.json not found")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds:
        raise AuthRequired("Gmail authentication required")

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        # google-api-python-client will refresh lazily when invoked.
        return creds

    raise AuthRequired("Stored Gmail credentials are invalid; please re-authenticate")


def _cleanup_flows(now: Optional[float] = None):
    if not _pending_flows:
        return
    now = now or time.time()
    expired = [state for state, (_, created) in _pending_flows.items() if now - created > _FLOW_TTL_SECONDS]
    for state in expired:
        _pending_flows.pop(state, None)


def start_auth_flow(callback_url: str) -> str:
    token_path, creds_path = _get_paths()
    if not os.path.exists(creds_path):
        raise FileNotFoundError("Gmail credentials.json not found")

    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    flow.redirect_uri = callback_url
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )

    _cleanup_flows()
    _pending_flows[state] = (flow, time.time())

    # Remove stale token to avoid confusion while a new flow is in progress.
    if os.path.exists(token_path):
        try:
            os.remove(token_path)
        except OSError:
            pass

    return auth_url


def complete_auth_flow(state: str, code: str):
    token_path, _ = _get_paths()

    entry = _pending_flows.pop(state, None)
    if not entry:
        raise AuthRequired("Invalid or expired authorization state")

    flow, _ = entry
    flow.fetch_token(code=code)
    creds = flow.credentials

    with open(token_path, "w") as token:
        token.write(creds.to_json())

    return creds


def get_gmail():
    creds = ensure_auth()
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


def list_recent_unread(service, max_results=25, q="is:unread newer_than:7d"):
    res = service.users().messages().list(userId="me", q=q, maxResults=max_results).execute()
    return res.get("messages", [])


def get_message(service, msg_id: str) -> Dict:
    return service.users().messages().get(userId="me", id=msg_id, format="full").execute()


def extract_payload(message: Dict) -> Dict[str, Optional[str]]:
    payload = message.get("payload", {}) or {}
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    subject = headers.get("subject", "")
    sender = headers.get("from", "")
    snippet = message.get("snippet", "")

    plain_parts: list[str] = []
    html_parts: list[str] = []

    def collect(part: Dict) -> None:
        mime = (part.get("mimeType") or "").lower()

        if mime.startswith("multipart/"):
            for sub in part.get("parts", []) or []:
                collect(sub)
            return

        body_data = part.get("body", {}).get("data")
        text = _decode_body_data(body_data)

        if mime == "text/plain":
            if text:
                plain_parts.append(text)
        elif mime == "text/html":
            if text:
                html_parts.append(text)
        elif text and (mime.startswith("text/") or not mime):
            plain_parts.append(text)

        for sub in part.get("parts", []) or []:
            collect(sub)

    collect(payload)

    if plain_parts:
        body_text = "\n".join(part.strip() for part in plain_parts if part.strip())
    else:
        html_texts = [_html_to_text(part) for part in html_parts]
        body_text = "\n\n".join(text for text in html_texts if text)

    body = _normalize_body(body_text)

    return {
        "subject": subject,
        "sender": sender,
        "snippet": snippet,
        "body": body,
    }

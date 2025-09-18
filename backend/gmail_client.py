import os
import base64
import time
from typing import Dict, Optional, Tuple

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


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


DEFAULT_UNREAD_QUERY = "is:unread newer_than:7d -category:promotions"


def list_recent_unread(service, max_results=25, q: Optional[str] = None):
    query = q or DEFAULT_UNREAD_QUERY
    res = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    return res.get("messages", [])


def get_message(service, msg_id: str) -> Dict:
    return service.users().messages().get(userId="me", id=msg_id, format="full").execute()


def extract_payload(message: Dict) -> Dict[str, Optional[str]]:
    headers = {h["name"].lower(): h["value"] for h in message.get("payload", {}).get("headers", [])}
    subject = headers.get("subject", "")
    sender = headers.get("from", "")
    snippet = message.get("snippet", "")

    body = ""

    def walk(parts):
        nonlocal body
        for p in parts:
            if p.get("mimeType") == "text/plain" and p.get("body", {}).get("data"):
                body += base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", errors="ignore") + "\n"
            elif p.get("parts"):
                walk(p["parts"])

    payload = message.get("payload", {})
    if payload.get("body", {}).get("data"):
        try:
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
        except Exception:
            body = ""
    elif payload.get("parts"):
        walk(payload["parts"])

    return {
        "subject": subject,
        "sender": sender,
        "snippet": snippet,
        "body": body,
    }

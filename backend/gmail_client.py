import os
import base64
from typing import List, Dict, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def _get_paths():
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "/app/token.json")
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "/app/credentials.json")
    return token_path, creds_path

def ensure_auth():
    token_path, creds_path = _get_paths()
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Let google-auth refresh lazily when used by client
            pass
        else:
            # Launch local server flow
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(host="0.0.0.0", port=8081, prompt="consent")
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

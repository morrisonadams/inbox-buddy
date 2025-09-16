import asyncio
import os
import json
import time
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from pydantic import BaseModel
from dotenv import load_dotenv

from db import init_db, SessionLocal, Email
from gmail_client import (
    AuthRequired,
    complete_auth_flow,
    ensure_auth,
    get_gmail,
    list_recent_unread,
    get_message,
    extract_payload,
    start_auth_flow,
)
from triage import classify, answer_question

load_dotenv()
init_db()

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "120"))
IMPORTANCE_THRESHOLD = float(os.getenv("REPLY_IMPORTANCE_THRESHOLD", "0.6"))
REPLY_THRESHOLD = float(os.getenv("REPLY_NEEDED_THRESHOLD", "0.6"))

app = FastAPI()

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SSE subscribers
subscribers = set()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/auth/start")
def auth_start(request: Request):
    try:
        ensure_auth()
        return {"already_authenticated": True}
    except AuthRequired:
        callback_url = str(request.url_for("auth_callback"))
        try:
            auth_url = start_auth_flow(callback_url)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"auth_url": auth_url}


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(state: str = "", code: str = "", error: str = ""):
    if error:
        return HTMLResponse(
            f"<html><body><h3>Authentication failed: {error}</h3></body></html>",
            status_code=400,
        )
    if not state or not code:
        raise HTTPException(status_code=400, detail="Missing state or code")

    try:
        complete_auth_flow(state, code)
    except AuthRequired as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    body = """
    <html>
        <body>
            <h3>Authentication complete. You can close this tab.</h3>
            <script>window.close();</script>
        </body>
    </html>
    """
    return HTMLResponse(body)

@app.get("/emails")
def get_emails(limit: int = 50):
    db = SessionLocal()
    try:
        q = db.query(Email).order_by(Email.internal_date.desc()).limit(limit).all()
        return [{
            "msg_id": e.msg_id,
            "thread_id": e.thread_id,
            "subject": e.subject,
            "sender": e.sender,
            "snippet": e.snippet,
            "body": e.body[:2000],
            "internal_date": e.internal_date,
            "is_unread": e.is_unread,
            "is_important": e.is_important,
            "reply_needed": e.reply_needed,
            "importance_score": e.importance_score,
            "reply_needed_score": e.reply_needed_score,
        } for e in q]
    finally:
        db.close()

class AskBody(BaseModel):
    question: str
    limit: int = 100

@app.post("/ask")
def ask(body: AskBody):
    db = SessionLocal()
    try:
        emails = db.query(Email).order_by(Email.internal_date.desc()).limit(body.limit).all()
        ctx = ""
        for e in emails:
            ctx += f"From: {e.sender}\nSubject: {e.subject}\nBody: {e.body[:2000]}\n---\n"
        answer = answer_question(ctx, body.question)
        return {"answer": answer}
    finally:
        db.close()

@app.get("/events")
async def events():
    async def event_stream():
        queue = asyncio.Queue()
        subscribers.add(queue)
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            subscribers.remove(queue)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

async def notify_all(data: Dict):
    for q in list(subscribers):
        await q.put(data)

async def poller():
    await asyncio.sleep(3)
    while True:
        try:
            service = get_gmail()
            msgs = list_recent_unread(service, max_results=25)
            if not msgs:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            db = SessionLocal()
            try:
                known_ids = {x[0] for x in db.query(Email.msg_id).all()}
                for m in msgs:
                    msg_id = m["id"]
                    if msg_id in known_ids:
                        continue
                    full = get_message(service, msg_id)
                    payload = extract_payload(full)
                    internal_date = int(full.get("internalDate", "0"))
                    email_text = f"From: {payload['sender']}\nSubject: {payload['subject']}\n\n{payload['body']}"
                    result = classify(email_text)

                    e = Email(
                        msg_id=msg_id,
                        thread_id=full.get("threadId"),
                        subject=payload["subject"],
                        sender=payload["sender"],
                        snippet=payload["snippet"],
                        body=payload["body"],
                        internal_date=internal_date,
                        is_unread=True,
                        is_important=bool(result.get("importance")) or float(result.get("importance_score", 0)) >= IMPORTANCE_THRESHOLD,
                        reply_needed=bool(result.get("reply_needed")) or float(result.get("reply_needed_score", 0)) >= REPLY_THRESHOLD,
                        importance_score=float(result.get("importance_score", 0)),
                        reply_needed_score=float(result.get("reply_needed_score", 0)),
                    )
                    db.add(e)
                    db.commit()

                    # Send SSE event
                    if e.is_important or e.reply_needed:
                        await notify_all({
                            "type": "important_email",
                            "msg_id": e.msg_id,
                            "subject": e.subject,
                            "sender": e.sender,
                            "reply_needed": e.reply_needed,
                            "importance_score": e.importance_score,
                            "reply_needed_score": e.reply_needed_score,
                            "snippet": e.snippet,
                        })
            finally:
                db.close()
        except Exception as ex:
            await notify_all({"type": "error", "message": str(ex)})
        await asyncio.sleep(POLL_INTERVAL)

@app.on_event("startup")
async def on_start():
    asyncio.create_task(poller())

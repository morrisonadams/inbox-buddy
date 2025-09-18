import asyncio
import json
import logging
import os
from typing import Dict

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
    list_recent_messages,
    get_message,
    extract_payload,
    start_auth_flow,
)
from triage import classify, answer_question, craft_assistant_message

load_dotenv()
init_db()

LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("inbox_buddy")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "120"))
IMPORTANCE_THRESHOLD = float(os.getenv("REPLY_IMPORTANCE_THRESHOLD", "0.6"))
REPLY_THRESHOLD = float(os.getenv("REPLY_NEEDED_THRESHOLD", "0.6"))
logger.info(
    "Backend initialized (poll_interval=%s, importance_threshold=%.2f, reply_threshold=%.2f)",
    POLL_INTERVAL,
    IMPORTANCE_THRESHOLD,
    REPLY_THRESHOLD,
)

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
poll_lock = asyncio.Lock()

PROMOTION_LABEL_HINTS = (
    "CATEGORY_PROMOTIONS",
    "^SMARTLABEL_PROMO",
    "SMARTLABEL_PROMO",
    "PROMOTIONS",
    "PROMOTION",
    "PROMO",
    "ADVERT",
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/auth/start")
def auth_start(request: Request):
    logger.debug("Auth start requested")
    try:
        ensure_auth()
        logger.info("Auth already completed; skipping OAuth flow")
        return {"already_authenticated": True}
    except AuthRequired:
        callback_url = str(request.url_for("auth_callback"))
        try:
            auth_url = start_auth_flow(callback_url)
        except FileNotFoundError as exc:
            logger.error("Auth flow failed: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        logger.info("Starting OAuth flow; redirecting user to Google auth")
        return {"auth_url": auth_url}


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(state: str = "", code: str = "", error: str = ""):
    if error:
        logger.error("Auth callback returned error: %s", error)
        return HTMLResponse(
            f"<html><body><h3>Authentication failed: {error}</h3></body></html>",
            status_code=400,
        )
    if not state or not code:
        logger.error("Auth callback missing state or code")
        raise HTTPException(status_code=400, detail="Missing state or code")

    try:
        complete_auth_flow(state, code)
    except AuthRequired as exc:
        logger.error("Auth completion failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info("OAuth flow completed successfully")
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
def get_emails(limit: int = 50, actionable_only: bool = True):
    db = SessionLocal()
    try:
        logger.debug(
            "Fetching %d emails (actionable_only=%s)",
            limit,
            actionable_only,
        )
        if limit <= 0:
            return []

        query = db.query(Email).order_by(Email.internal_date.desc())
        if actionable_only:
            query = query.filter(Email.is_important.is_(True))

        fetch_limit = max(limit, 1) * 5
        emails = query.limit(fetch_limit).all()

        results = []
        seen_threads = set()

        for e in emails:
            thread_key = e.thread_id or e.msg_id
            if thread_key in seen_threads:
                continue
            seen_threads.add(thread_key)
            summary_lines = [
                line.strip()
                for line in (e.assistant_summary or "").splitlines()
                if line.strip()
            ]
            results.append({
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
                "actionable": e.is_important,
                "assistant_message": e.assistant_message,
                "assistant_summary": summary_lines,
                "assistant_reply": e.assistant_reply,
            })
            if len(results) >= limit:
                break
        return results
    finally:
        db.close()

class AskBody(BaseModel):
    question: str
    limit: int = 100

@app.post("/ask")
def ask(body: AskBody):
    db = SessionLocal()
    try:
        logger.info("Received ask request (limit=%d)", body.limit)
        emails = db.query(Email).order_by(Email.internal_date.desc()).limit(body.limit).all()
        ctx = ""
        for e in emails:
            ctx += f"From: {e.sender}\nSubject: {e.subject}\nBody: {e.body[:2000]}\n---\n"
        try:
            answer = answer_question(ctx, body.question)
        except Exception as exc:
            logger.exception("Failed to answer question")
            raise HTTPException(status_code=500, detail="Failed to answer question") from exc
        return {"answer": answer}
    finally:
        db.close()


@app.post("/reset")
async def reset_inbox():
    logger.info("Reset endpoint invoked; clearing stored emails")
    db = SessionLocal()
    try:
        deleted = db.query(Email).delete()
        db.commit()
    finally:
        db.close()

    await notify_all({"type": "reset", "deleted": deleted})

    try:
        asyncio.create_task(run_poll_cycle(trigger="manual_reset"))
    except RuntimeError:  # pragma: no cover - fallback for closed loop
        logger.warning("Event loop not running; skipping immediate poll trigger")

    return {"deleted": deleted}

@app.get("/events")
async def events():
    async def event_stream():
        queue = asyncio.Queue()
        subscribers.add(queue)
        logger.debug("SSE subscriber added (total=%d)", len(subscribers))
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            subscribers.remove(queue)
            logger.debug("SSE subscriber removed (total=%d)", len(subscribers))
    return StreamingResponse(event_stream(), media_type="text/event-stream")

async def notify_all(data: Dict):
    logger.debug(
        "Dispatching event '%s' to %d subscribers",
        data.get("type", "unknown"),
        len(subscribers),
    )
    for q in list(subscribers):
        await q.put(data)


def _is_promotional_message(message: Dict) -> bool:
    labels = message.get("labelIds") or []
    for raw in labels:
        label = str(raw or "").upper()
        if any(hint in label for hint in PROMOTION_LABEL_HINTS):
            return True
    return False


async def poller():
    logger.info("Poller starting with interval %s seconds", POLL_INTERVAL)
    await asyncio.sleep(3)
    while True:
        try:
            await run_poll_cycle()
        except Exception as ex:  # pragma: no cover - defensive guard
            logger.exception("Poller encountered an error")
            await notify_all({"type": "error", "message": str(ex)})
        await asyncio.sleep(POLL_INTERVAL)


async def run_poll_cycle(trigger: str = "scheduled") -> Dict:
    if poll_lock.locked():
        logger.debug("Skipping poll cycle because another run is active (trigger=%s)", trigger)
        return {"status": "busy"}

    async with poll_lock:
        logger.info("Running inbox poll (trigger=%s)", trigger)
        try:
            service = get_gmail()
        except AuthRequired:
            logger.warning("Gmail authentication required; poll will retry later")
            await notify_all({"type": "auth_required"})
            return {"status": "auth_required"}

        try:
            msgs = list_recent_messages(service, max_results=25)
        except Exception as exc:  # pragma: no cover - network failure guard
            logger.exception("Failed to list Gmail messages")
            await notify_all({"type": "error", "message": str(exc)})
            return {"status": "error", "error": str(exc)}

        logger.debug("Fetched %d messages from Gmail", len(msgs) if msgs else 0)
        if not msgs:
            return {"status": "empty"}

        db = SessionLocal()
        processed = 0
        try:
            known_ids = {x[0] for x in db.query(Email.msg_id).all()}
            for m in msgs:
                msg_id = m.get("id")
                if not msg_id:
                    continue
                if msg_id in known_ids:
                    logger.debug("Skipping known message %s", msg_id)
                    continue

                try:
                    full = get_message(service, msg_id)
                except Exception:
                    logger.exception("Failed to fetch message %s", msg_id)
                    continue

                if _is_promotional_message(full):
                    logger.info("Skipping promotional email msg_id=%s", msg_id)
                    continue

                payload = extract_payload(full)
                labels = [str(label or "").upper() for label in full.get("labelIds", [])]
                is_unread = "UNREAD" in labels
                internal_date = int(full.get("internalDate", "0"))
                email_text = (
                    f"From: {payload['sender']}\n"
                    f"Subject: {payload['subject']}\n\n"
                    f"{payload['body']}"
                )

                try:
                    result = classify(email_text)
                except Exception:
                    logger.exception("Classification failed for msg_id=%s", msg_id)
                    await notify_all({"type": "error", "message": "Classification failed"})
                    continue

                try:
                    importance_score = float(result.get("importance_score", 0))
                except (TypeError, ValueError):
                    importance_score = 0.0
                try:
                    reply_needed_score = float(result.get("reply_needed_score", 0))
                except (TypeError, ValueError):
                    reply_needed_score = 0.0

                importance_score = max(0.0, min(1.0, importance_score))
                reply_needed_score = max(0.0, min(1.0, reply_needed_score))

                importance_flag = bool(result.get("importance")) or importance_score >= IMPORTANCE_THRESHOLD
                reply_flag = bool(result.get("reply_needed")) or reply_needed_score >= REPLY_THRESHOLD

                if importance_score < 0.45:
                    importance_flag = False
                if reply_needed_score < 0.45:
                    reply_flag = False

                actionable_flag = bool(result.get("actionable")) or (importance_flag and reply_flag)

                if actionable_flag and importance_score < reply_needed_score:
                    importance_score = reply_needed_score

                assistant_message = ""
                assistant_summary_text = ""
                assistant_reply = ""
                assistant_payload = None
                if actionable_flag:
                    try:
                        assistant_payload = craft_assistant_message(payload)
                    except Exception:
                        logger.exception(
                            "Failed to craft assistant guidance for msg_id=%s", msg_id
                        )
                        assistant_payload = {
                            "notification": "",
                            "summary": [],
                            "reply_draft": "",
                        }
                    assistant_message = str(assistant_payload.get("notification", "")).strip()
                    assistant_summary_items = [
                        str(item).strip()
                        for item in assistant_payload.get("summary", [])
                        if str(item).strip()
                    ]
                    assistant_summary_text = "\n".join(assistant_summary_items)
                    assistant_reply = str(assistant_payload.get("reply_draft", "")).strip()

                e = Email(
                    msg_id=msg_id,
                    thread_id=full.get("threadId"),
                    subject=payload["subject"],
                    sender=payload["sender"],
                    snippet=payload["snippet"],
                    body=payload["body"],
                    internal_date=internal_date,
                    is_unread=is_unread,
                    is_important=actionable_flag,
                    reply_needed=reply_flag,
                    importance_score=importance_score,
                    reply_needed_score=reply_needed_score,
                    assistant_message=assistant_message,
                    assistant_summary=assistant_summary_text,
                    assistant_reply=assistant_reply,
                )
                db.add(e)
                db.commit()
                known_ids.add(msg_id)
                processed += 1

                logger.info(
                    "Stored email msg_id=%s subject=%s importance=%.2f reply_needed=%.2f actionable=%s",
                    e.msg_id,
                    e.subject,
                    e.importance_score,
                    e.reply_needed_score,
                    actionable_flag,
                )

                if actionable_flag:
                    logger.info(
                        "Notifying subscribers about actionable email msg_id=%s", e.msg_id
                    )
                    assistant_summary = assistant_payload.get("summary", []) if assistant_payload else []
                    await notify_all(
                        {
                            "type": "important_email",
                            "msg_id": e.msg_id,
                            "subject": e.subject,
                            "sender": e.sender,
                            "reply_needed": e.reply_needed,
                            "importance_score": e.importance_score,
                            "reply_needed_score": e.reply_needed_score,
                            "snippet": e.snippet,
                            "actionable": actionable_flag,
                            "assistant_message": assistant_message,
                            "assistant_summary": assistant_summary,
                            "assistant_reply": assistant_reply,
                        }
                    )
        finally:
            db.close()

        logger.info(
            "Poll cycle finished (trigger=%s processed=%d)", trigger, processed
        )
        return {"status": "processed", "processed": processed}

@app.on_event("startup")
async def on_start():
    logger.info("Starting background poller task")
    asyncio.create_task(poller())

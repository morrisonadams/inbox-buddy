import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import Email, SessionLocal, get_emails  # noqa: E402


def _make_email(**kwargs):
    defaults = {
        "msg_id": "",
        "thread_id": "",
        "subject": "Subject",
        "sender": "sender@example.com",
        "snippet": "Snippet",
        "body": "Body text",
        "internal_date": 0,
        "is_unread": True,
        "is_important": True,
        "reply_needed": True,
        "importance_score": 0.8,
        "reply_needed_score": 0.8,
    }
    defaults.update(kwargs)
    return Email(**defaults)


def test_get_emails_dedupes_by_thread_and_keeps_latest():
    db = SessionLocal()
    try:
        db.query(Email).delete()

        db.add_all(
            [
                _make_email(
                    msg_id="m1",
                    thread_id="thread-1",
                    subject="Original",
                    internal_date=100,
                ),
                _make_email(
                    msg_id="m2",
                    thread_id="thread-1",
                    subject="Follow up",
                    internal_date=200,
                ),
                _make_email(
                    msg_id="m3",
                    thread_id="thread-2",
                    subject="Different thread",
                    internal_date=150,
                ),
            ]
        )
        db.commit()

        results = get_emails(limit=10, actionable_only=True)

        assert len(results) == 2
        assert {item["thread_id"] for item in results} == {"thread-1", "thread-2"}
        assert results[0]["msg_id"] == "m2"
        assert results[0]["subject"] == "Follow up"
    finally:
        db.query(Email).delete()
        db.commit()
        db.close()

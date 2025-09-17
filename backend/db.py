from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Float,
    Text,
    DateTime,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = "sqlite:///db.sqlite3"

Base = declarative_base()

class Email(Base):
    __tablename__ = "emails"
    id = Column(Integer, primary_key=True, autoincrement=True)
    msg_id = Column(String, unique=True, index=True, nullable=False)
    thread_id = Column(String, index=True)
    subject = Column(String, default="")
    sender = Column(String, default="")
    snippet = Column(Text, default="")
    body = Column(Text, default="")
    internal_date = Column(Integer)  # epoch ms
    is_unread = Column(Boolean, default=True)
    is_important = Column(Boolean, default=False)
    reply_needed = Column(Boolean, default=False)
    importance_score = Column(Float, default=0.0)
    reply_needed_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    assistant_message = Column(Text, default="")
    assistant_summary = Column(Text, default="")
    assistant_reply = Column(Text, default="")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        result = conn.execute(text("PRAGMA table_info(emails)"))
        columns = {row[1] for row in result}

        def ensure_column(name: str, ddl: str) -> None:
            if name not in columns:
                conn.execute(text(f"ALTER TABLE emails ADD COLUMN {ddl}"))
                columns.add(name)

        ensure_column("assistant_message", "assistant_message TEXT DEFAULT ''")
        ensure_column("assistant_summary", "assistant_summary TEXT DEFAULT ''")
        ensure_column("assistant_reply", "assistant_reply TEXT DEFAULT ''")

from sqlalchemy import Column, Integer, String, Boolean, Float, Text, DateTime, create_engine, select
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

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(engine)

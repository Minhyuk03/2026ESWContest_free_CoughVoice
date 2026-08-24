"""DB 세션 — 개발은 SQLite, 운영 전환 시 DATABASE_URL만 교체."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cough_id.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate()


def _migrate() -> None:
    """create_all은 기존 테이블에 새 컬럼을 추가하지 않으므로 SQLite 한정 보정."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    from sqlalchemy import text

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(persons)"))}
        if "room" not in cols:
            conn.execute(text("ALTER TABLE persons ADD COLUMN room VARCHAR(20)"))
        if "sample_count" not in cols:
            conn.execute(text("ALTER TABLE persons ADD COLUMN sample_count INTEGER DEFAULT 0"))

        # P5 — 알림 규칙 평가 파라미터. 기존 행은 기본값으로 채운 뒤
        # seed 시점의 condition_text를 보고 alerts.py가 backfill한다.
        rule_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(alert_rules)"))}
        for col, ddl in [
            ("kind", "VARCHAR(20) DEFAULT 'count_window'"),
            ("threshold_count", "INTEGER DEFAULT 10"),
            ("window_minutes", "INTEGER DEFAULT 60"),
            ("night_start_hour", "INTEGER DEFAULT 22"),
            ("night_end_hour", "INTEGER DEFAULT 6"),
            ("cooldown_minutes", "INTEGER DEFAULT 30"),
        ]:
            if col not in rule_cols:
                conn.execute(text(f"ALTER TABLE alert_rules ADD COLUMN {col} {ddl}"))


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

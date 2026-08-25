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
        ev_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(cough_events)"))}
        if "event_id" not in ev_cols:
            conn.execute(text("ALTER TABLE cough_events ADD COLUMN event_id VARCHAR(40)"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_cough_events_event_id "
                "ON cough_events (event_id)"))

        # P6 — 음향 지표. 기존 행은 NULL로 남는다(당시 저장하지 않았으므로).
        # 집계 시 NULL을 0으로 취급하면 "휘징이 없었다"는 잘못된 사실이 되므로
        # cough_metrics는 NULL 행을 표본에서 빼는 방식으로 다룬다.
        for col, ddl in [("cough_score", "FLOAT"), ("wheeze_prob", "FLOAT"),
                         ("gasp_prob", "FLOAT")]:
            if col not in ev_cols:
                conn.execute(text(f"ALTER TABLE cough_events ADD COLUMN {col} {ddl}"))

        # 원음 보존 정책(NFR-06) — 등록에 쓰인 이벤트는 만료 삭제에서 제외한다.
        if "enrolled" not in ev_cols:
            conn.execute(text(
                "ALTER TABLE cough_events ADD COLUMN enrolled BOOLEAN DEFAULT 0"))

        al_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(alerts)"))}
        for col, ddl in [("severity", "VARCHAR(10) DEFAULT 'info'"),
                         ("source", "VARCHAR(120)")]:
            if col not in al_cols:
                conn.execute(text(f"ALTER TABLE alerts ADD COLUMN {col} {ddl}"))

        rule_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(alert_rules)"))}
        for col, ddl in [
            ("kind", "VARCHAR(20) DEFAULT 'count_window'"),
            ("threshold_count", "INTEGER DEFAULT 10"),
            ("window_minutes", "INTEGER DEFAULT 60"),
            ("night_start_hour", "INTEGER DEFAULT 22"),
            ("night_end_hour", "INTEGER DEFAULT 6"),
            ("cooldown_minutes", "INTEGER DEFAULT 30"),
            # P6 — 변화·기간 경고 파라미터
            ("baseline_days", "INTEGER DEFAULT 7"),
            ("ratio_threshold", "FLOAT DEFAULT 2.0"),
            ("sustain_hours", "INTEGER DEFAULT 24"),
            ("duration_days", "INTEGER DEFAULT 14"),
            ("allowed_gap_days", "INTEGER DEFAULT 2"),
        ]:
            if col not in rule_cols:
                conn.execute(text(f"ALTER TABLE alert_rules ADD COLUMN {col} {ddl}"))

        # 웹훅은 구현하지 않았는데 규칙 카드가 "보호자 웹훅 · 관리자 웹훅"으로 표시해
        # 화면이 거짓을 말하고 있었다. 기본값 그대로인 행만 실제 동작으로 바로잡는다
        # (사용자가 직접 고친 문구는 건드리지 않는다).
        conn.execute(text(
            "UPDATE alert_rules SET channels_text = '대시보드 표시' "
            "WHERE channels_text = '보호자 웹훅 · 관리자 웹훅'"))


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

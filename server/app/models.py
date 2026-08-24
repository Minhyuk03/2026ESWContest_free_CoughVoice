"""SQLAlchemy 모델 — Class 다이어그램의 Person·CoughEvent·Alert 대응."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alias: Mapped[str] = mapped_column(String(50), unique=True)  # 실명 대신 alias (NFR-06)
    room: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # 호실 표기 (예: "301호")
    embedding_ref: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)  # 임베딩 평균
    sample_count: Mapped[int] = mapped_column(Integer, default=0)  # 등록 시 녹음한 샘플 수
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    events: Mapped[List["CoughEvent"]] = relationship(back_populates="person")


class CoughEvent(Base):
    __tablename__ = "cough_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str] = mapped_column(String(50))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    person_id: Mapped[Optional[int]] = mapped_column(ForeignKey("persons.id"), nullable=True)  # None = unknown
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_rms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_path: Mapped[str] = mapped_column(String(255))  # 저장된 wav 경로

    person: Mapped[Optional[Person]] = relationship(back_populates="events")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[Optional[int]] = mapped_column(ForeignKey("persons.id"), nullable=True)
    rule: Mapped[str] = mapped_column(String(100))       # 예: "1h>=10"
    message: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AlertRule(Base):
    """S4 알림 규칙 카드.

    condition_text는 화면 표시용 문구이고, 실제 평가는 kind/threshold_count/
    window_minutes로 한다. 문구를 파싱해서 판단하면 사용자가 문구를 고치는 순간
    동작이 바뀌어버리므로 분리해 둔다.
    """

    __tablename__ = "alert_rules"

    KIND_COUNT = "count_window"    # 지정 시간 안에 N회 이상
    KIND_NIGHT = "night_window"    # 야간 시간대에 한해 N회 이상
    KIND_UNKNOWN = "unknown"       # 미등록 화자의 기침이 발생하면 즉시

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50))            # 예: "이상 징후"
    condition_text: Mapped[str] = mapped_column(String(100))  # 예: "기침 ≥ 10회 / 1시간"
    target_text: Mapped[str] = mapped_column(String(50), default="전체 화자")
    channels_text: Mapped[str] = mapped_column(String(100), default="보호자 웹훅 · 관리자 웹훅")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # --- 평가 파라미터 ---
    kind: Mapped[str] = mapped_column(String(20), default=KIND_COUNT)
    threshold_count: Mapped[int] = mapped_column(Integer, default=10)
    window_minutes: Mapped[int] = mapped_column(Integer, default=60)
    night_start_hour: Mapped[int] = mapped_column(Integer, default=22)   # 현지 시각 기준
    night_end_hour: Mapped[int] = mapped_column(Integer, default=6)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=30)   # 재알림 억제


class User(Base):
    """대시보드 로그인 계정 — 비밀번호는 PBKDF2 해시로만 저장."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    salt: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20), default="admin")  # admin | guardian
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

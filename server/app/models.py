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
    """S4 알림 규칙 카드 — 조건 문구는 표시용, 평가 파라미터는 count/window로 저장."""

    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50))            # 예: "이상 징후"
    condition_text: Mapped[str] = mapped_column(String(100))  # 예: "기침 ≥ 10회 / 1시간"
    target_text: Mapped[str] = mapped_column(String(50), default="전체 화자")
    channels_text: Mapped[str] = mapped_column(String(100), default="보호자 웹훅 · 관리자 웹훅")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    """대시보드 로그인 계정 — 비밀번호는 PBKDF2 해시로만 저장."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    salt: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20), default="admin")  # admin | guardian
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

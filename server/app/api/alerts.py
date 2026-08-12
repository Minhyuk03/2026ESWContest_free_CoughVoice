"""AlertsAPI — 알림 이력·규칙 (S1 배너, S4 알림 센터)."""
from __future__ import annotations

from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Alert, AlertRule, Person

router = APIRouter(tags=["알림"])

DEFAULT_RULES = [
    ("이상 징후", "기침 ≥ 10회 / 1시간", "전체 화자", True),
    ("야간 기침", "기침 ≥ 5회 / 22–06시", "전체 화자", True),
    ("미등록 감지", "미등록 기침 발생 시", "—", False),
]


def seed_rules(db: Session) -> None:
    if db.scalar(select(AlertRule)) is not None:
        return
    for name, cond, target, enabled in DEFAULT_RULES:
        db.add(AlertRule(name=name, condition_text=cond, target_text=target, enabled=enabled))
    db.commit()


@router.get("/alerts", summary="알림 이력")
def list_alerts(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.scalars(select(Alert).order_by(Alert.created_at.desc()).limit(limit)).all()
    out = []
    for a in rows:
        person = db.get(Person, a.person_id) if a.person_id else None
        out.append({
            "id": a.id,
            "rule": a.rule,
            "message": a.message,
            "person_id": a.person_id,
            "person_alias": person.alias if person else None,
            "person_room": person.room if person else None,
            "created_at": (a.created_at.replace(tzinfo=timezone.utc)
                           if a.created_at.tzinfo is None else a.created_at).isoformat(),
        })
    return out


@router.get("/alert-rules", summary="알림 규칙 목록")
def list_rules(db: Session = Depends(get_db)):
    return [
        {
            "id": r.id,
            "name": r.name,
            "condition_text": r.condition_text,
            "target_text": r.target_text,
            "channels_text": r.channels_text,
            "enabled": r.enabled,
        }
        for r in db.scalars(select(AlertRule).order_by(AlertRule.id)).all()
    ]


class RuleBody(BaseModel):
    name: Optional[str] = None
    condition_text: Optional[str] = None
    target_text: Optional[str] = None
    channels_text: Optional[str] = None
    enabled: Optional[bool] = None


@router.patch("/alert-rules/{rule_id}", summary="알림 규칙 수정 (ON/OFF 토글 포함)")
def update_rule(rule_id: int, body: RuleBody, db: Session = Depends(get_db)):
    r = db.get(AlertRule, rule_id)
    if r is None:
        raise HTTPException(status_code=404, detail="규칙을 찾을 수 없습니다")
    for field in ("name", "condition_text", "target_text", "channels_text", "enabled"):
        v = getattr(body, field)
        if v is not None:
            setattr(r, field, v)
    db.commit()
    return {"ok": True}


@router.post("/alert-rules", status_code=201, summary="알림 규칙 추가")
def create_rule(body: RuleBody, db: Session = Depends(get_db)):
    r = AlertRule(
        name=body.name or "새 규칙",
        condition_text=body.condition_text or "",
        target_text=body.target_text or "전체 화자",
        channels_text=body.channels_text or "보호자 웹훅 · 관리자 웹훅",
        enabled=body.enabled if body.enabled is not None else True,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return {"id": r.id}

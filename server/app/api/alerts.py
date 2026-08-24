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

# 표시 문구와 평가 파라미터를 함께 정의한다. 문구를 파싱해 판단하지 않는 이유는
# 사용자가 문구만 고쳐도 동작이 바뀌는 사고를 막기 위함이다(models.AlertRule 참조).
DEFAULT_RULES = [
    dict(name="이상 징후", condition_text="기침 ≥ 10회 / 1시간", target_text="전체 화자",
         enabled=True, kind=AlertRule.KIND_COUNT, threshold_count=10,
         window_minutes=60, cooldown_minutes=30),
    dict(name="야간 기침", condition_text="기침 ≥ 5회 / 22–06시", target_text="전체 화자",
         enabled=True, kind=AlertRule.KIND_NIGHT, threshold_count=5,
         window_minutes=480, night_start_hour=22, night_end_hour=6, cooldown_minutes=60),
    dict(name="미등록 감지", condition_text="미등록 기침 발생 시", target_text="—",
         enabled=False, kind=AlertRule.KIND_UNKNOWN, cooldown_minutes=30),
]

EVAL_FIELDS = ("kind", "threshold_count", "window_minutes",
               "night_start_hour", "night_end_hour", "cooldown_minutes")


def seed_rules(db: Session) -> None:
    """규칙이 없으면 기본 3종을 넣는다. 이미 있으면 평가 파라미터만 보정한다."""
    existing = db.scalars(select(AlertRule)).all()
    if not existing:
        for spec in DEFAULT_RULES:
            db.add(AlertRule(**spec))
        db.commit()
        return

    # P5 이전에 만들어진 행은 kind가 비어 있다. 이름으로 찾아 채운다.
    by_name = {r.name: r for r in existing}
    changed = False
    for spec in DEFAULT_RULES:
        r = by_name.get(spec["name"])
        if r is not None and not getattr(r, "kind", None):
            for f in EVAL_FIELDS:
                if f in spec:
                    setattr(r, f, spec[f])
            changed = True
    if changed:
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
            "kind": r.kind,
            "threshold_count": r.threshold_count,
            "window_minutes": r.window_minutes,
            "cooldown_minutes": r.cooldown_minutes,
        }
        for r in db.scalars(select(AlertRule).order_by(AlertRule.id)).all()
    ]


class RuleBody(BaseModel):
    name: Optional[str] = None
    condition_text: Optional[str] = None
    target_text: Optional[str] = None
    channels_text: Optional[str] = None
    enabled: Optional[bool] = None
    kind: Optional[str] = None
    threshold_count: Optional[int] = None
    window_minutes: Optional[int] = None
    night_start_hour: Optional[int] = None
    night_end_hour: Optional[int] = None
    cooldown_minutes: Optional[int] = None


@router.patch("/alert-rules/{rule_id}", summary="알림 규칙 수정 (ON/OFF 토글 포함)")
def update_rule(rule_id: int, body: RuleBody, db: Session = Depends(get_db)):
    r = db.get(AlertRule, rule_id)
    if r is None:
        raise HTTPException(status_code=404, detail="규칙을 찾을 수 없습니다")
    for field in ("name", "condition_text", "target_text", "channels_text",
                  "enabled") + EVAL_FIELDS:
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
        kind=body.kind or AlertRule.KIND_COUNT,
        threshold_count=body.threshold_count if body.threshold_count is not None else 10,
        window_minutes=body.window_minutes if body.window_minutes is not None else 60,
        cooldown_minutes=body.cooldown_minutes if body.cooldown_minutes is not None else 30,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return {"id": r.id}

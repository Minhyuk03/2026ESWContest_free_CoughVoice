"""AlertsAPI — 알림 이력·규칙 (S1 배너, S4 알림 센터)."""
from __future__ import annotations

from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core import guidance
from ..db import get_db
from ..models import Alert, AlertRule, Person

router = APIRouter(tags=["알림"])

# 표시 문구와 평가 파라미터를 함께 정의한다. 문구를 파싱해 판단하지 않는 이유는
# 사용자가 문구만 고쳐도 동작이 바뀌는 사고를 막기 위함이다(models.AlertRule 참조).
DEFAULT_RULES = [
    # --- 사용자 지정 관찰 규칙 (절대 횟수) ---
    dict(name="이상 징후", condition_text="기침 ≥ 10회 / 1시간", target_text="전체 화자",
         enabled=True, kind=AlertRule.KIND_COUNT, threshold_count=10,
         window_minutes=60, cooldown_minutes=30),
    dict(name="야간 기침", condition_text="기침 ≥ 5회 / 22–06시", target_text="전체 화자",
         enabled=True, kind=AlertRule.KIND_NIGHT, threshold_count=5,
         window_minutes=480, night_start_hour=22, night_end_hour=6, cooldown_minutes=60),
    dict(name="미등록 감지", condition_text="미등록 기침 발생 시", target_text="—",
         enabled=False, kind=AlertRule.KIND_UNKNOWN, cooldown_minutes=30),

    # --- 참고자료 권고 경고 구조 (P6) ---
    # 변화 경고의 2배는 지침값이 아니라 탐색용 기준이다. 그래서 알림도 info로 뜬다.
    dict(name="평소 대비 증가", condition_text="개인 기준선의 2배 / 최근 24시간",
         target_text="전체 화자", enabled=True, kind=AlertRule.KIND_BASELINE,
         baseline_days=7, ratio_threshold=2.0, sustain_hours=24, cooldown_minutes=360),
    # 2주는 질병관리청 결핵검진 권고 기준이다. '결핵 의심'이 아니라 '검진 안내'다.
    dict(name="기침 지속 기간", condition_text="기침 2주 이상 지속",
         target_text="전체 화자", enabled=True, kind=AlertRule.KIND_DURATION,
         duration_days=guidance.DURATION_TB_SCREENING_DAYS, allowed_gap_days=2,
         cooldown_minutes=1440),
    # 긴급은 억제하지 않는다(cooldown 0). 두 번째 입력이 묻히면 안 된다.
    dict(name="긴급 증상", condition_text="객혈·호흡곤란 등 입력 시 즉시",
         target_text="전체 화자", enabled=True, kind=AlertRule.KIND_URGENT,
         cooldown_minutes=0),
]

EVAL_FIELDS = ("kind", "threshold_count", "window_minutes",
               "night_start_hour", "night_end_hour", "cooldown_minutes",
               "baseline_days", "ratio_threshold", "sustain_hours",
               "duration_days", "allowed_gap_days")


def _evaluation_text(r: AlertRule) -> str:
    """규칙이 실제로 무엇을 보는지 한 줄로 설명한다.

    화면이 이 문자열을 만들면 표시와 동작이 어긋난다 — 조건 문구는 사용자가
    고칠 수 있지만 평가는 파라미터로 도는 구조이기 때문이다. 서버가 만들어 내려준다.
    """
    if r.kind == AlertRule.KIND_UNKNOWN:
        return "미등록 화자 기침 발생 시 즉시"
    if r.kind == AlertRule.KIND_URGENT:
        return "긴급 증상이 입력되면 기침 횟수와 무관하게 즉시"
    if r.kind == AlertRule.KIND_DURATION:
        return (f"기침이 {r.duration_days}일 이상 이어질 때"
                f" (기침 없는 날 {r.allowed_gap_days}일까지는 이어진 것으로 봄)")
    if r.kind == AlertRule.KIND_BASELINE:
        return (f"최근 {r.sustain_hours}시간이 개인 기준선({r.baseline_days}일 학습)의"
                f" {r.ratio_threshold:g}배 이상일 때")
    base = f"{r.threshold_count}회 / {r.window_minutes}분"
    return base + (" (야간 시간대만)" if r.kind == AlertRule.KIND_NIGHT else "")


def seed_rules(db: Session) -> None:
    """규칙이 없으면 기본 3종을 넣는다. 이미 있으면 평가 파라미터만 보정한다."""
    existing = db.scalars(select(AlertRule)).all()
    if not existing:
        for spec in DEFAULT_RULES:
            db.add(AlertRule(**spec))
        db.commit()
        return

    # 이미 규칙이 있는 DB에도 P6에서 새로 생긴 종류는 넣어 준다. 이름으로 판별하므로
    # 사용자가 지운 규칙이 되살아나지는 않는다 — 지운 뒤 같은 이름으로 다시 만들면
    # 그때만 중복이 생기는데, 그건 사용자가 의도한 이름 충돌이다.
    names = {r.name for r in existing}
    added = False
    for spec in DEFAULT_RULES:
        if spec["kind"] in (AlertRule.KIND_BASELINE, AlertRule.KIND_DURATION,
                            AlertRule.KIND_URGENT) and spec["name"] not in names:
            db.add(AlertRule(**spec))
            added = True
    if added:
        db.commit()
        existing = db.scalars(select(AlertRule)).all()

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
            "severity": a.severity or guidance.SEV_INFO,
            "source": a.source,
            "person_id": a.person_id,
            "person_alias": person.alias if person else None,
            "person_room": person.room if person else None,
            "created_at": (a.created_at.replace(tzinfo=timezone.utc)
                           if a.created_at.tzinfo is None else a.created_at).isoformat(),
        })
    # 화면이 문구를 직접 들고 있지 않도록 서버가 함께 내려준다. 면책 표시를
    # 빠뜨린 화면이 생기는 것을 구조적으로 막기 위함이다.
    return {"items": out, "disclaimer": guidance.DISCLAIMER}


@router.get("/alert-rules", summary="알림 규칙 목록")
def list_rules(db: Session = Depends(get_db)):
    return {
        "items": [
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
                "baseline_days": r.baseline_days,
                "ratio_threshold": r.ratio_threshold,
                "sustain_hours": r.sustain_hours,
                "duration_days": r.duration_days,
                "allowed_gap_days": r.allowed_gap_days,
                # 임상 지침에 근거가 있는 규칙과 사용자가 정한 관찰 기준을 구분한다.
                # 섞어서 보여주면 "10회/1시간"도 의학적 기준처럼 읽힌다.
                "clinical": r.kind in AlertRule.CLINICAL_KINDS,
                "evaluation_text": _evaluation_text(r),
            }
            for r in db.scalars(select(AlertRule).order_by(AlertRule.id)).all()
        ],
        "disclaimer": guidance.DISCLAIMER,
    }


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
    baseline_days: Optional[int] = None
    ratio_threshold: Optional[float] = None
    sustain_hours: Optional[int] = None
    duration_days: Optional[int] = None
    allowed_gap_days: Optional[int] = None


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
        channels_text=body.channels_text or "대시보드 표시",
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

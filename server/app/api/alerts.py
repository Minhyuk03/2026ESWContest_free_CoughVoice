"""AlertsAPI — 알림 이력·규칙 (S1 배너, S4 알림 센터)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core import guidance
from ..core.alert_engine import engine as alert_engine
from ..db import get_db
from ..models import Alert, AlertRule, AlertRuleChange, Person
from .auth import current_user

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


STATUS_LABELS = {
    Alert.STATUS_OPEN: "미확인",
    Alert.STATUS_ACK: "확인함",
    Alert.STATUS_DONE: "조치 완료",
}


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).isoformat()


def _rule_kinds(db: Session) -> dict:
    """규칙 이름 → 종류. 알림은 규칙 이름만 들고 있는데, 화면은 종류별로 다른 아이콘을
    보여야 한다(전부 같은 표시면 훑어볼 때 구분이 안 된다). 이름이 바뀌거나 규칙이
    지워진 알림은 매칭되지 않고 None으로 남는다 — 그때는 심각도 표시로 물러선다."""
    return {r.name: r.kind for r in db.scalars(select(AlertRule)).all()}


def _alert_row(db: Session, a: Alert, kinds: Optional[dict] = None) -> dict:
    person = db.get(Person, a.person_id) if a.person_id else None
    kinds = _rule_kinds(db) if kinds is None else kinds
    severity = a.severity or guidance.SEV_INFO
    status = a.status or Alert.STATUS_OPEN
    return {
        "id": a.id,
        "rule": a.rule,
        "message": a.message,
        "severity": severity,
        # 화면이 코드값을 한글로 옮기지 않도록 라벨까지 함께 준다. 화면마다 매핑을
        # 들고 있으면 한쪽만 고쳐져 같은 알림이 화면에 따라 다르게 읽힌다.
        "severity_label": guidance.severity_label(severity),
        "severity_hint": guidance.SEV_HINTS.get(severity, ""),
        "severity_rank": guidance.SEV_RANK.get(severity, 9),
        "rule_kind": kinds.get(a.rule),
        "source": a.source,
        "person_id": a.person_id,
        "person_alias": person.alias if person else None,
        "person_room": person.room if person else None,
        "created_at": _iso(a.created_at),
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "assignee": a.assignee,
        "acked_at": _iso(a.acked_at),
        "note": a.note,
    }


@router.get("/alerts", summary="알림 이력")
def list_alerts(limit: int = 50, status: Optional[str] = None,
                severity: Optional[str] = None, db: Session = Depends(get_db)):
    q = select(Alert).order_by(Alert.created_at.desc())
    if status:
        q = q.where(Alert.status == status)
    if severity:
        q = q.where(Alert.severity == severity)
    rows = db.scalars(q.limit(limit)).all()
    # 미확인 건수는 목록을 잘라도 정확해야 한다(배지에 쓰인다). 따로 센다.
    open_count = len(db.scalars(select(Alert).where(
        (Alert.status == Alert.STATUS_OPEN) | (Alert.status.is_(None)))).all())
    # 화면이 문구를 직접 들고 있지 않도록 서버가 함께 내려준다. 면책 표시를
    # 빠뜨린 화면이 생기는 것을 구조적으로 막기 위함이다.
    kinds = _rule_kinds(db)
    return {"items": [_alert_row(db, a, kinds) for a in rows],
            "open_count": open_count,
            "status_labels": STATUS_LABELS,
            "disclaimer": guidance.DISCLAIMER}


class AlertPatchBody(BaseModel):
    status: Optional[str] = None
    note: Optional[str] = None


@router.patch("/alerts/{alert_id}", summary="알림 확인·조치 상태 변경")
def update_alert(alert_id: int, body: AlertPatchBody,
                 actor: str = Depends(current_user), db: Session = Depends(get_db)):
    """상태를 바꾼 사람과 시각을 함께 남긴다.

    담당자를 사용자가 고르게 하지 않고 '상태를 바꾼 계정'으로 기록하는 이유는,
    고를 수 있게 하면 실제로 확인한 사람과 적어 넣은 이름이 갈라지기 때문이다.
    """
    a = db.get(Alert, alert_id)
    if a is None:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다")
    if body.status is not None:
        if body.status not in Alert.STATUSES:
            raise HTTPException(status_code=400, detail="알 수 없는 상태입니다")
        a.status = body.status
        if body.status == Alert.STATUS_OPEN:
            # 미확인으로 되돌리면 담당자·확인 시각도 지운다. 남겨두면 "아무도 안 봤는데
            # 확인자가 있는" 상태가 된다.
            a.assignee = None
            a.acked_at = None
        else:
            a.assignee = actor
            a.acked_at = datetime.now(timezone.utc).replace(tzinfo=None)
    if body.note is not None:
        a.note = body.note.strip()[:500] or None
    db.commit()
    db.refresh(a)
    return _alert_row(db, a)


def _rule_row(r: AlertRule) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "condition_text": r.condition_text,
        "target_text": r.target_text,
        "channels_text": r.channels_text,
        "enabled": r.enabled,
        "kind": r.kind,
        "threshold_count": r.threshold_count,
        "window_minutes": r.window_minutes,
        "night_start_hour": r.night_start_hour,
        "night_end_hour": r.night_end_hour,
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


@router.get("/alert-rules", summary="알림 규칙 목록")
def list_rules(db: Session = Depends(get_db)):
    return {
        "items": [_rule_row(r) for r in db.scalars(select(AlertRule).order_by(AlertRule.id)).all()],
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


# 변경 이력에 쓰는 한글 이름. 컬럼명을 그대로 적으면(threshold_count) 이력을 읽는
# 사람이 무엇이 바뀐 건지 알 수 없다.
FIELD_LABELS = {
    "name": "이름",
    "condition_text": "조건 문구",
    "target_text": "대상",
    "channels_text": "수신 채널",
    "enabled": "사용",
    "kind": "규칙 종류",
    "threshold_count": "기준 횟수",
    "window_minutes": "관찰 구간(분)",
    "night_start_hour": "야간 시작(시)",
    "night_end_hour": "야간 종료(시)",
    "cooldown_minutes": "재알림 억제(분)",
    "baseline_days": "기준선 학습(일)",
    "ratio_threshold": "기준선 대비 배수",
    "sustain_hours": "증가 지속(시간)",
    "duration_days": "지속 기간(일)",
    "allowed_gap_days": "허용 공백(일)",
}


def _fmt_value(field: str, v) -> str:
    if field == "enabled":
        return "ON" if v else "OFF"
    return "—" if v is None or v == "" else str(v)


def _log_change(db: Session, rule: AlertRule, action: str, actor: str, summary: str) -> None:
    db.add(AlertRuleChange(rule_id=rule.id, rule_name=rule.name, action=action,
                           summary=summary[:300], actor=actor))


@router.patch("/alert-rules/{rule_id}", summary="알림 규칙 수정 (ON/OFF 토글 포함)")
def update_rule(rule_id: int, body: RuleBody, actor: str = Depends(current_user),
                db: Session = Depends(get_db)):
    r = db.get(AlertRule, rule_id)
    if r is None:
        raise HTTPException(status_code=404, detail="규칙을 찾을 수 없습니다")
    # 무엇이 어떻게 바뀌었는지 이력에 남긴다. 저장 전 값과 비교해야 하므로
    # setattr 하기 전에 모아 둔다.
    diffs = []
    for field in ("name", "condition_text", "target_text", "channels_text",
                  "enabled") + EVAL_FIELDS:
        v = getattr(body, field)
        if v is None:
            continue
        before = getattr(r, field)
        if before == v:
            continue
        diffs.append(f"{FIELD_LABELS.get(field, field)} {_fmt_value(field, before)} → "
                     f"{_fmt_value(field, v)}")
        setattr(r, field, v)
    if diffs:
        _log_change(db, r, "update", actor, " · ".join(diffs))
    db.commit()
    return {"ok": True, "changed": diffs}


@router.post("/alert-rules", status_code=201, summary="알림 규칙 추가")
def create_rule(body: RuleBody, actor: str = Depends(current_user),
                db: Session = Depends(get_db)):
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
    _log_change(db, r, "create", actor, f"신규 생성 — {_evaluation_text(r)}")
    db.commit()
    return {"id": r.id}


@router.post("/alert-rules/{rule_id}/duplicate", status_code=201, summary="알림 규칙 복제")
def duplicate_rule(rule_id: int, actor: str = Depends(current_user),
                   db: Session = Depends(get_db)):
    """같은 조건을 값만 바꿔 쓰고 싶을 때 쓴다.

    복제본은 **꺼진 상태로** 만든다. 켜진 채로 복제되면 원본과 사본이 같은 상황에서
    두 번 울린다 — 값을 고치기도 전에 알림이 두 배가 된다.
    """
    src = db.get(AlertRule, rule_id)
    if src is None:
        raise HTTPException(status_code=404, detail="규칙을 찾을 수 없습니다")
    fields = ("condition_text", "target_text", "channels_text") + EVAL_FIELDS
    copy = AlertRule(name=f"{src.name} (복사본)", enabled=False,
                     **{f: getattr(src, f) for f in fields})
    db.add(copy)
    db.commit()
    db.refresh(copy)
    _log_change(db, copy, "duplicate", actor, f"'{src.name}' 복제 (꺼진 상태로 생성)")
    db.commit()
    return {"id": copy.id}


@router.post("/alert-rules/{rule_id}/test", summary="알림 규칙 시험 실행 (알림을 만들지 않음)")
def test_rule(rule_id: int, db: Session = Depends(get_db)):
    """지금 이 순간의 데이터로 규칙을 돌려 보고, 울릴지 여부만 돌려준다.

    실제 알림은 만들지 않는다. 규칙을 켜기 전에 "이 기준이면 지금 울리는가"를
    확인하는 용도다. 꺼져 있는 규칙도 시험할 수 있다 — 켜기 전에 보는 게 목적이므로.
    """
    r = db.get(AlertRule, rule_id)
    if r is None:
        raise HTTPException(status_code=404, detail="규칙을 찾을 수 없습니다")
    results = alert_engine.dry_run(db, r)
    fired = [x for x in results if x["would_fire"]]
    return {
        "rule_id": r.id,
        "name": r.name,
        "enabled": r.enabled,
        "evaluation_text": _evaluation_text(r),
        "checked_at": _iso(datetime.now(timezone.utc)),
        "would_fire": bool(fired),
        "results": results,
        "note": ("시험 실행은 알림을 만들지 않으며, 재알림 억제(쿨다운)도 적용하지 않습니다. "
                 "실제 운영에서는 억제 중이면 울리지 않을 수 있습니다."),
    }


@router.get("/alert-rule-changes", summary="알림 규칙 변경 이력")
def list_rule_changes(limit: int = 30, db: Session = Depends(get_db)):
    rows = db.scalars(select(AlertRuleChange)
                      .order_by(AlertRuleChange.created_at.desc(), AlertRuleChange.id.desc())
                      .limit(limit)).all()
    return {"items": [{"id": c.id, "rule_id": c.rule_id, "rule_name": c.rule_name,
                       "action": c.action, "summary": c.summary, "actor": c.actor,
                       "created_at": _iso(c.created_at)} for c in rows]}

"""SymptomsAPI — 동반 증상 입력 (P6, 긴급 경고의 입력원).

참고자료의 긴급 징후(객혈·호흡곤란·청색증·지속적 흉통·의식저하·낮은 SpO₂)는
**소리로는 알 수 없다.** 마이크가 아무리 잘 들어도 객혈은 검출되지 않는다.
그래서 사람이 입력하는 경로가 필요하고, 그 입력이 들어오는 순간 기침 횟수와
무관하게 즉시 진료 안내를 띄운다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core import guidance
from ..core.alert_engine import engine as alert_engine
from ..db import get_db
from ..models import Person, SymptomReport

router = APIRouter(tags=["증상"])


def _iso(dt: datetime) -> str:
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).isoformat()


class SymptomBody(BaseModel):
    person_id: Optional[int] = None
    symptoms: List[str] = []
    spo2: Optional[float] = None
    temperature_c: Optional[float] = None
    respiratory_rate: Optional[int] = None
    note: Optional[str] = None


@router.get("/symptom-options", summary="입력 가능한 증상 목록")
def symptom_options():
    """화면이 증상 코드를 하드코딩하지 않도록 서버가 목록을 준다.

    코드가 양쪽에 흩어지면 한쪽만 고쳤을 때 조용히 어긋난다 — 화면에는 있는데
    서버가 모르는 증상은 입력해도 아무 일도 일어나지 않는다.
    """
    return {
        "urgent": [{"code": c, "label": lab} for c, lab in guidance.URGENT_SYMPTOMS.items()],
        "other": [{"code": c, "label": lab} for c, lab in guidance.OTHER_SYMPTOMS.items()],
        "spo2_urgent_below": guidance.SPO2_URGENT_BELOW,
        "disclaimer": guidance.DISCLAIMER,
    }


@router.post("/symptoms", status_code=201, summary="증상 입력")
def create_symptom(body: SymptomBody, db: Session = Depends(get_db)):
    unknown = [c for c in body.symptoms if c not in guidance.SYMPTOM_LABELS]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"알 수 없는 증상 코드: {', '.join(unknown)}")
    if body.person_id is not None and db.get(Person, body.person_id) is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")

    report = SymptomReport(
        person_id=body.person_id,
        symptoms=",".join(body.symptoms),
        spo2=body.spo2,
        temperature_c=body.temperature_c,
        respiratory_rate=body.respiratory_rate,
        note=body.note,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    alerts = alert_engine.evaluate_symptom(db, report)
    return {
        "id": report.id,
        "urgent_reasons": guidance.urgent_reasons(report.codes(), report.spo2),
        "alerts": [{"rule": a.rule, "message": a.message, "severity": a.severity}
                   for a in alerts],
        "disclaimer": guidance.DISCLAIMER,
    }


@router.get("/symptoms", summary="증상 입력 이력")
def list_symptoms(limit: int = 50, person: Optional[int] = None,
                  db: Session = Depends(get_db)):
    q = select(SymptomReport).order_by(SymptomReport.reported_at.desc()).limit(limit)
    if person is not None:
        q = q.where(SymptomReport.person_id == person)
    rows = db.scalars(q).all()
    persons = {p.id: p for p in db.scalars(select(Person)).all()}
    return {
        "items": [
            {
                "id": r.id,
                "person_id": r.person_id,
                "person_alias": persons[r.person_id].alias if r.person_id in persons else None,
                "reported_at": _iso(r.reported_at),
                "symptoms": [{"code": c, "label": guidance.SYMPTOM_LABELS.get(c, c)}
                             for c in r.codes()],
                "spo2": r.spo2,
                "temperature_c": r.temperature_c,
                "respiratory_rate": r.respiratory_rate,
                "note": r.note,
                "urgent_reasons": guidance.urgent_reasons(r.codes(), r.spo2),
            }
            for r in rows
        ],
        "disclaimer": guidance.DISCLAIMER,
    }

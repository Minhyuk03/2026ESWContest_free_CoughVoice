"""StatsAPI — S1 대시보드 통계 (스탯 카드·24h 추이·화자별 현황).

이벤트 수가 적은 데모 규모라 DB에서 당일 이벤트를 통째로 가져와
파이썬에서 로컬 시간 기준으로 집계한다 (SQLite 날짜 함수 의존 제거).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Alert, CoughEvent, Person

router = APIRouter(prefix="/stats", tags=["통계"])

DEVICE_ONLINE_WINDOW = timedelta(minutes=5)  # 최근 5분 내 수신이 있으면 온라인으로 간주


def _today_events(db: Session):
    start_local = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = db.scalars(
        select(CoughEvent).where(CoughEvent.captured_at >= start_local)
    ).all()
    return rows


@router.get("/overview", summary="대시보드 요약")
def overview(db: Session = Depends(get_db)):
    events = _today_events(db)
    now = datetime.now().astimezone()

    last_received = db.scalar(select(func.max(CoughEvent.received_at)))
    online = False
    if last_received is not None:
        if last_received.tzinfo is None:
            last_received = last_received.astimezone()
        online = (now - last_received) <= DEVICE_ONLINE_WINDOW

    day_ago = now - timedelta(hours=24)
    active_alerts = db.scalar(select(func.count(Alert.id)).where(Alert.created_at >= day_ago)) or 0

    person_count = db.scalar(select(func.count(Person.id))) or 0

    return {
        "today_cough_count": len(events),
        "active_alerts": active_alerts,
        "person_count": person_count,
        "device_online": online,
    }


@router.get("/hourly", summary="시간대별 기침 발생 추이 (오늘 24h)")
def hourly(db: Session = Depends(get_db)):
    counts = [0] * 24
    for e in _today_events(db):
        t = e.captured_at
        if t.tzinfo is None:
            t = t.astimezone()
        counts[t.astimezone().hour] += 1
    return {"counts": counts}


@router.get("/by-person", summary="화자별 오늘 기침 현황")
def by_person(db: Session = Depends(get_db)):
    events = _today_events(db)
    persons = {p.id: p for p in db.scalars(select(Person)).all()}
    tally = {}
    unknown = 0
    for e in events:
        if e.person_id and e.person_id in persons:
            tally[e.person_id] = tally.get(e.person_id, 0) + 1
        else:
            unknown += 1
    rows = [
        {
            "person_id": pid,
            "alias": persons[pid].alias,
            "room": persons[pid].room,
            "count": n,
        }
        for pid, n in sorted(tally.items(), key=lambda kv: -kv[1])
    ]
    # 등록됐지만 오늘 기침이 없는 화자도 0회로 표시
    for pid, p in persons.items():
        if pid not in tally:
            rows.append({"person_id": pid, "alias": p.alias, "room": p.room, "count": 0})
    rows.append({"person_id": None, "alias": "미등록", "room": None, "count": unknown})
    return rows

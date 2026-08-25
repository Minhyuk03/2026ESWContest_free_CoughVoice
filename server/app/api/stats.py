"""StatsAPI — S1 대시보드 통계 (스탯 카드·24h 추이·화자별 현황).

DB의 시각 컬럼은 전부 UTC(naive) 기준으로 저장된다 (SQLite가 tz를 버리므로).
필터·집계 시에는 UTC로 간주해 로컬 시간으로 변환한다.
이벤트 수가 적은 데모 규모라 당일 이벤트를 통째로 가져와 파이썬에서 집계한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core import guidance
from ..core.cough_metrics import burden, hourly_baseline
from ..db import get_db
from ..models import Alert, CoughEvent, Person

router = APIRouter(prefix="/stats", tags=["통계"])

DEVICE_ONLINE_WINDOW = timedelta(minutes=5)  # 최근 5분 내 수신이 있으면 온라인으로 간주


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _utc_naive(dt: datetime) -> datetime:
    """aware → DB 비교용 UTC naive."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _today_events(db: Session):
    start_local = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.scalars(
        select(CoughEvent).where(CoughEvent.captured_at >= _utc_naive(start_local))
    ).all()


@router.get("/overview", summary="대시보드 요약")
def overview(db: Session = Depends(get_db)):
    events = _today_events(db)
    now = datetime.now(timezone.utc)

    last_received = db.scalar(select(func.max(CoughEvent.received_at)))
    online = last_received is not None and (now - _as_utc(last_received)) <= DEVICE_ONLINE_WINDOW

    day_ago = _utc_naive(now - timedelta(hours=24))
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
        counts[_as_utc(e.captured_at).astimezone().hour] += 1
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


@router.get("/cough-burden", summary="기침 부담 지표 (참고자료 권고 지표)")
def cough_burden(days: int = 7, person: Optional[int] = None, all_persons: bool = True,
                 db: Session = Depends(get_db)):
    """단순 일일 횟수 대신 참고자료가 기록을 권한 지표들을 돌려준다.

    person을 주면 그 화자만, 주지 않으면 전체(all_persons=true)를 집계한다.
    all_persons=false이고 person이 없으면 **미등록 화자 묶음**을 뜻한다.

    응답의 unavailable은 "우리 구성으로는 낼 수 없는 지표"와 그 이유다. 빈칸을
    0으로 채우지 않는 이유는 보고서에서 '측정했다'로 읽히지 않게 하기 위함이다.
    """
    b = burden(db, person_id=person, all_persons=all_persons and person is None, days=days)
    out = b.to_dict()
    out["person_id"] = person
    out["disclaimer"] = guidance.DISCLAIMER
    return out


@router.get("/baseline", summary="개인 기준선 (시간대별 중앙값)")
def baseline(days: int = 7, person: Optional[int] = None, all_persons: bool = True,
             exclude_recent_hours: int = 0, db: Session = Depends(get_db)):
    """참고자료: "건강하거나 안정된 7~14일의 시간대별 중앙값을 계산한다."

    관측이 없는 시간대는 null로 남긴다 — 0으로 채우면 '관측이 없었다'와
    '기침이 없었다'가 구분되지 않는다.
    """
    values = hourly_baseline(db, person_id=person,
                             all_persons=all_persons and person is None,
                             days=days, exclude_recent_hours=exclude_recent_hours)
    return {
        "days": days,
        "person_id": person,
        "hourly_median": values,
        "note": "권고 학습 창은 7~14일이며, 안정된 기간의 값이어야 의미가 있습니다.",
        "disclaimer": guidance.DISCLAIMER,
    }

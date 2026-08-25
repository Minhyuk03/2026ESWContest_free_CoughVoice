"""DevicesAPI — 엣지 생존 신호 (P6).

엣지는 기침이 있을 때만 이벤트를 보낸다. 조용한 밤에 아무 이벤트가 없는 것과 파이가
죽어 있는 것이 서버에서 똑같이 보이는데, 그러면 24시간 연속 동작을 검증할 수도 없고
기준선이 가동 중단 구간을 '기침 0회'로 세어버린다. 그래서 주기적인 생존 신호를 받는다.

신호는 가볍다 — 60초에 한 번, 본문은 장치 ID뿐이다. 기록은 (장치, 시각) 단위로 묶어
개수만 올리므로 하루 24행이면 끝난다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.cough_metrics import as_utc, utc_naive
from ..db import get_db
from ..models import DeviceUptime
from .security import require_device_token

router = APIRouter(tags=["장치"])

# 이 시간 안에 신호가 있으면 살아 있는 것으로 본다. 비트 간격(60초)의 3배 —
# 한두 번 놓쳐도 오프라인으로 뒤집히지 않을 만큼의 여유다.
ONLINE_WINDOW = timedelta(seconds=180)

# 한 시간을 '가동'으로 인정할 최소 비트 수. 60초 간격이면 정상은 60회이므로 절반.
# 이 값을 낮추면 잠깐 켜졌던 시간까지 기준선 표본에 들어가 평균을 끌어내린다.
MIN_BEATS_PER_HOUR = 30


class HeartbeatBody(BaseModel):
    device_id: str = "unknown"


@router.post("/heartbeat", summary="엣지 생존 신호",
             description="엣지가 주기적으로 호출한다. (장치, 시각) 단위로 묶어 횟수만 센다.")
def heartbeat(body: HeartbeatBody, db: Session = Depends(get_db),
              _token: None = Depends(require_device_token)):
    now = datetime.now(timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    row = db.scalar(select(DeviceUptime).where(
        DeviceUptime.device_id == body.device_id,
        DeviceUptime.hour_utc == utc_naive(hour)))
    if row is None:
        row = DeviceUptime(device_id=body.device_id, hour_utc=utc_naive(hour),
                           beat_count=1, first_seen=utc_naive(now), last_seen=utc_naive(now))
        db.add(row)
    else:
        row.beat_count += 1
        row.last_seen = utc_naive(now)
    db.commit()
    return {"ok": True, "device_id": body.device_id, "beat_count": row.beat_count}


def last_seen(db: Session, device_id: Optional[str] = None) -> Optional[datetime]:
    q = select(DeviceUptime).order_by(DeviceUptime.last_seen.desc()).limit(1)
    if device_id:
        q = q.where(DeviceUptime.device_id == device_id)
    row = db.scalar(q)
    return as_utc(row.last_seen) if row else None


def is_online(db: Session, device_id: Optional[str] = None,
              now: Optional[datetime] = None) -> Optional[bool]:
    """생존 여부. 하트비트 기록이 아예 없으면 None — '모른다'와 '꺼졌다'는 다르다."""
    seen = last_seen(db, device_id)
    if seen is None:
        return None
    return (now or datetime.now(timezone.utc)) - seen <= ONLINE_WINDOW


def covered_hours(db: Session, start: datetime, end: datetime) -> set:
    """구간 안에서 장치가 실제로 돌고 있던 (현지 날짜, 시각) 집합.

    기준선 계산이 가동 중단 구간을 표본에서 빼는 데 쓴다.
    """
    from ..core.cough_metrics import to_local
    rows = db.scalars(select(DeviceUptime).where(
        DeviceUptime.hour_utc >= utc_naive(start),
        DeviceUptime.hour_utc < utc_naive(end))).all()
    out = set()
    for r in rows:
        if r.beat_count >= MIN_BEATS_PER_HOUR:
            lt = to_local(r.hour_utc)
            out.add((lt.date(), lt.hour))
    return out


@router.get("/devices", summary="장치 상태")
def list_devices(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    rows = db.scalars(select(DeviceUptime)).all()
    by_dev: dict = {}
    for r in rows:
        d = by_dev.setdefault(r.device_id, {"device_id": r.device_id, "beats": 0,
                                            "hours_covered": 0, "last_seen": None})
        d["beats"] += r.beat_count
        if r.beat_count >= MIN_BEATS_PER_HOUR:
            d["hours_covered"] += 1
        ls = as_utc(r.last_seen)
        if d["last_seen"] is None or ls > d["last_seen"]:
            d["last_seen"] = ls
    out = []
    for d in by_dev.values():
        seen = d["last_seen"]
        out.append({**d,
                    "last_seen": seen.isoformat() if seen else None,
                    "online": bool(seen and now - seen <= ONLINE_WINDOW),
                    "seconds_since_seen": round((now - seen).total_seconds(), 1) if seen else None})
    return {"items": sorted(out, key=lambda x: x["device_id"]),
            "online_window_seconds": ONLINE_WINDOW.total_seconds(),
            "min_beats_per_hour": MIN_BEATS_PER_HOUR}

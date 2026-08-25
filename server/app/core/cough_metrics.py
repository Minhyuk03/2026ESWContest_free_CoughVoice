"""기침 부담(cough burden) 지표 — 참고자료가 "기기에 기록할 핵심 지표"로 든 항목들.

설계 원칙 두 가지.

**1. 다시 계산할 수 있는 값은 저장하지 않는다.**
시간당 횟수·발작 수·무기침 간격은 전부 CoughEvent.captured_at에서 다시 뽑을 수 있다.
컬럼으로 복제해 두면 이벤트를 보정(M1 화자 수정, 중복 제거)할 때마다 어긋난다.
반대로 음향 확률(cough_score·wheeze·gasp)은 원음을 지우면 복구할 수 없으므로 컬럼에 남긴다.

**2. 측정할 수 없는 지표는 0이나 추정치로 채우지 않고 '측정 불가'로 표시한다.**
참고자료의 지표 목록 중 우리 파이프라인이 낼 수 없는 것이 있다. 그 자리를 그럴듯한
값으로 채우면 보고서에서 "측정했다"로 읽힌다. UNAVAILABLE에 이유와 함께 남긴다.

발작(bout) 정의에 대하여 — core/guidance.py에 자세히 적었지만 요약하면:
엣지 CoughDetector의 쿨다운 2.0초가 ERS 기침 측정 지침의 bout 경계(공백 ≤2초)와 같다.
따라서 **이벤트 1건 = 발작 1회(bout)** 이지 개별 기침 1회가 아니다. 이름을 그렇게 붙인다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import CoughEvent

# 야간 판정·일자 구분에 쓸 현지 시각 오프셋. 배포지가 바뀌면 이 값만 조정한다.
LOCAL_TZ_OFFSET = timedelta(hours=float(os.environ.get("COUGHID_TZ_OFFSET", "9")))

# 참고자료 지표 중 현재 구성으로는 낼 수 없는 것들. 이유를 함께 남긴다.
UNAVAILABLE: Dict[str, str] = {
    "individual_cough_count":
        "엣지 쿨다운 2초가 발작 경계와 같아 발작 내부의 개별 기침을 세지 않는다",
    "coughs_per_bout":
        "individual_cough_count가 없으므로 계산할 수 없다",
    "max_bout_duration":
        "엣지가 고정 길이 클립만 보내 발작의 실제 지속시간을 알 수 없다",
    "cough_seconds_per_hour":
        "발작 지속시간을 모르므로 '기침이 포함된 초'를 셀 수 없다",
    "wet_dry":
        "습성/건성 분류 모델이 없다",
    "whoop":
        'AudioSet "Whoop"은 환호성이라 백일해의 흡기성 whoop에 대응하지 않는다',
}


def as_utc(dt: datetime) -> datetime:
    """SQLite가 tz를 버리므로 naive 값은 UTC로 간주한다 (저장 규약)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def to_local(dt: datetime) -> datetime:
    return as_utc(dt) + LOCAL_TZ_OFFSET


def local_hour(dt: datetime) -> int:
    return to_local(dt).hour


def local_date(dt: datetime) -> date:
    return to_local(dt).date()


def utc_naive(dt: datetime) -> datetime:
    """aware → DB 비교용 UTC naive."""
    return as_utc(dt).replace(tzinfo=None)


def in_night(hour: int, start: int = 22, end: int = 6) -> bool:
    """22–06처럼 자정을 넘는 구간을 다룬다."""
    return start <= hour or hour < end if start > end else start <= hour < end


def fetch_events(db: Session, *, person_id: Optional[int] = None,
                 all_persons: bool = False,
                 since: Optional[datetime] = None,
                 until: Optional[datetime] = None) -> List[CoughEvent]:
    """대상 이벤트를 시간순으로 가져온다.

    all_persons=False이고 person_id=None이면 **미등록 화자 묶음**을 뜻한다
    (alert_engine의 집계 규약과 같다). 전체를 원하면 all_persons=True.
    """
    q = select(CoughEvent)
    if not all_persons:
        q = q.where(CoughEvent.person_id.is_(None) if person_id is None
                    else CoughEvent.person_id == person_id)
    if since is not None:
        q = q.where(CoughEvent.captured_at >= utc_naive(since))
    if until is not None:
        q = q.where(CoughEvent.captured_at < utc_naive(until))
    rows = db.scalars(q).all()
    return sorted(rows, key=lambda e: as_utc(e.captured_at))


@dataclass
class Burden:
    """한 대상·한 기간의 기침 부담."""

    window_days: int
    window_start: datetime          # UTC aware
    window_end: datetime
    bout_count: int = 0
    bouts_per_hour: float = 0.0
    max_bouts_per_hour: float = 0.0
    peak_hour_local: Optional[int] = None
    day_count: int = 0              # 주간(기본 06–22시)
    night_count: int = 0            # 수면시간(기본 22–06시)
    longest_cough_free_minutes: Optional[float] = None
    first_event_at: Optional[datetime] = None
    duration_days: int = 0          # 연속 지속일수
    active_days: int = 0            # 기침이 한 번이라도 있던 날 수
    change_3d: Optional[float] = None    # 직전 동일 기간 대비 배수
    change_7d: Optional[float] = None
    change_14d: Optional[float] = None
    wheeze_mean: Optional[float] = None
    gasp_mean: Optional[float] = None
    scored_events: int = 0          # 음향 지표가 저장된 이벤트 수
    unavailable: Dict[str, str] = field(default_factory=lambda: dict(UNAVAILABLE))

    def to_dict(self) -> dict:
        return {
            "window_days": self.window_days,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "bout_count": self.bout_count,
            "bouts_per_hour": round(self.bouts_per_hour, 3),
            "max_bouts_per_hour": self.max_bouts_per_hour,
            "peak_hour_local": self.peak_hour_local,
            "day_count": self.day_count,
            "night_count": self.night_count,
            "longest_cough_free_minutes": (
                None if self.longest_cough_free_minutes is None
                else round(self.longest_cough_free_minutes, 1)),
            "first_event_at": (self.first_event_at.isoformat()
                               if self.first_event_at else None),
            "duration_days": self.duration_days,
            "active_days": self.active_days,
            "change_3d": self.change_3d,
            "change_7d": self.change_7d,
            "change_14d": self.change_14d,
            "wheeze_mean": self.wheeze_mean,
            "gasp_mean": self.gasp_mean,
            "scored_events": self.scored_events,
            "unavailable": self.unavailable,
            # 이름이 오해를 부르지 않도록 규약을 응답에 함께 싣는다.
            "bout_definition": "개별 기침 간 공백 ≤2초를 한 발작으로 묶음 (ERS 기침 측정 지침)",
        }


def _count_in(events: List[CoughEvent], start: datetime, end: datetime) -> int:
    return sum(1 for e in events if start <= as_utc(e.captured_at) < end)


def _ratio(recent: int, prior: int) -> Optional[float]:
    """직전 동일 기간 대비 배수. 직전 기간이 0이면 배수가 정의되지 않는다."""
    if prior <= 0:
        return None
    return round(recent / prior, 2)


def burden(db: Session, *, person_id: Optional[int] = None, all_persons: bool = False,
           days: int = 7, now: Optional[datetime] = None,
           night_start: int = 22, night_end: int = 6,
           allowed_gap_days: int = 2) -> Burden:
    """지정 기간의 기침 부담 지표를 계산한다."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    events = fetch_events(db, person_id=person_id, all_persons=all_persons, since=start, until=now)

    b = Burden(window_days=days, window_start=start, window_end=now, bout_count=len(events))
    hours = max(days * 24, 1)
    b.bouts_per_hour = len(events) / hours

    # 시간대별 분포 — 최대 시간당 횟수와 그 시각
    per_hour: Dict[datetime, int] = {}
    for e in events:
        key = to_local(e.captured_at).replace(minute=0, second=0, microsecond=0)
        per_hour[key] = per_hour.get(key, 0) + 1
        if in_night(local_hour(e.captured_at), night_start, night_end):
            b.night_count += 1
        else:
            b.day_count += 1
    if per_hour:
        peak = max(per_hour.items(), key=lambda kv: kv[1])
        b.max_bouts_per_hour = float(peak[1])
        b.peak_hour_local = peak[0].hour

    # 최장 무기침 간격 — 창 경계도 후보로 넣어야 "요즘 잠잠하다"가 반영된다
    marks = [start] + [as_utc(e.captured_at) for e in events] + [now]
    gaps = [(marks[i + 1] - marks[i]).total_seconds() / 60 for i in range(len(marks) - 1)]
    if gaps:
        b.longest_cough_free_minutes = max(gaps)

    if events:
        b.first_event_at = as_utc(events[0].captured_at)

    # 음향 지표 — 저장된 이벤트만 평균한다. 없는 행을 0으로 세면 "휘징이 없었다"는
    # 거짓이 된다(P6 이전 이벤트는 아예 저장하지 않았다).
    wz = [e.wheeze_prob for e in events if e.wheeze_prob is not None]
    gp = [e.gasp_prob for e in events if e.gasp_prob is not None]
    b.scored_events = len(wz)
    if wz:
        b.wheeze_mean = round(sum(wz) / len(wz), 5)
    if gp:
        b.gasp_mean = round(sum(gp) / len(gp), 5)

    # 최근 3·7·14일 변화율 — 직전 동일 길이 기간과 비교한다
    for n, attr in ((3, "change_3d"), (7, "change_7d"), (14, "change_14d")):
        recent_start = now - timedelta(days=n)
        prior_start = now - timedelta(days=2 * n)
        wide = fetch_events(db, person_id=person_id, all_persons=all_persons,
                            since=prior_start, until=now)
        setattr(b, attr, _ratio(_count_in(wide, recent_start, now),
                                _count_in(wide, prior_start, recent_start)))

    streak, first = duration_streak(db, person_id=person_id, all_persons=all_persons,
                                    now=now, allowed_gap_days=allowed_gap_days)
    b.duration_days = streak
    b.active_days = len({local_date(e.captured_at) for e in events})
    if first is not None and (b.first_event_at is None or first < local_date(b.first_event_at)):
        # 지속 구간이 조회 창보다 앞에서 시작했으면 그 날짜를 쓴다
        b.first_event_at = datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc)
    return b


def duration_streak(db: Session, *, person_id: Optional[int] = None,
                    all_persons: bool = False, now: Optional[datetime] = None,
                    allowed_gap_days: int = 2,
                    lookback_days: int = 180) -> tuple[int, Optional[date]]:
    """현재 기침이 며칠째 이어지고 있는지와 그 시작일.

    매일 기침해야 '지속'으로 보는 것은 지나치게 엄격하다 — 하루 조용한 날이 있다고
    2주 기준이 초기화되면 기간 경고가 사실상 발동하지 않는다. 기침 없는 날이
    allowed_gap_days를 **연속으로** 넘길 때만 끊긴 것으로 본다.

    돌려주는 값은 (지속일수, 시작일). 최근 활동이 없으면 (0, None).
    """
    now = now or datetime.now(timezone.utc)
    events = fetch_events(db, person_id=person_id, all_persons=all_persons,
                          since=now - timedelta(days=lookback_days), until=now)
    if not events:
        return 0, None

    days = sorted({local_date(e.captured_at) for e in events})
    today = to_local(now).date()

    # 마지막 기침이 너무 오래됐으면 진행 중인 지속 구간이 아니다
    if (today - days[-1]).days > allowed_gap_days:
        return 0, None

    start = days[-1]
    for prev, cur in zip(reversed(days[1:]), reversed(days[:-1])):
        if (prev - cur).days > allowed_gap_days + 1:
            break
        start = cur
    return (today - start).days + 1, start


def hourly_baseline(db: Session, *, person_id: Optional[int] = None,
                    all_persons: bool = False, days: int = 7,
                    now: Optional[datetime] = None,
                    exclude_recent_hours: int = 0) -> List[Optional[float]]:
    """시간대별(0–23시) 기침 수의 중앙값 — 개인 기준선.

    참고자료: "건강하거나 안정된 7~14일의 시간대별 중앙값을 계산한다."
    평균이 아니라 중앙값인 이유는 발작이 한 번 크게 나면 평균이 통째로 끌려가기 때문이다.

    exclude_recent_hours를 주면 최근 구간을 기준선에서 뺀다. 지금 판단하려는 구간이
    기준선에 섞이면 "평소보다 늘었나"를 자기 자신과 비교하게 된다.

    **가동 중단 구간은 표본에서 뺀다.** 엣지는 기침이 있을 때만 이벤트를 보내므로
    "그날 조용했다"와 "그날 장치가 꺼져 있었다"가 이벤트만으로는 구분되지 않는다.
    그대로 두면 정전·네트워크 장애 구간이 '기침 0회'로 세어져 기준선이 실제보다 낮게
    잡히고, 복구 후 '평소의 N배' 경고가 잘못 뜬다. 그래서 하트비트가 남아 있는
    (날짜, 시각) 칸만 센다 — DeviceUptime 참조.

    하트비트 기록이 아예 없으면(구버전 엣지) 예전처럼 창 안의 모든 날짜를 쓴다.
    그 경우 위 왜곡이 그대로 남으므로, 판단이 중요한 자리에서는 /devices로
    가동 이력이 있는지 함께 확인할 것.
    """
    now = now or datetime.now(timezone.utc)
    end = now - timedelta(hours=exclude_recent_hours)
    start = end - timedelta(days=days)
    events = fetch_events(db, person_id=person_id, all_persons=all_persons,
                          since=start, until=end)

    # (날짜, 시각) 칸마다 개수를 센 뒤 시간대별로 날짜에 걸쳐 중앙값을 낸다
    cells: Dict[tuple, int] = {}
    for e in events:
        lt = to_local(e.captured_at)
        cells[(lt.date(), lt.hour)] = cells.get((lt.date(), lt.hour), 0) + 1

    days_in_window = _dates_between(start, end)
    from ..api.devices import covered_hours          # 순환 임포트를 피해 지역에서 부른다
    covered = covered_hours(db, start, end)

    out: List[Optional[float]] = []
    for h in range(24):
        if covered:
            vals = [cells.get((d, h), 0) for d in days_in_window if (d, h) in covered]
        else:
            vals = [cells.get((d, h), 0) for d in days_in_window]
        # 관측된 칸이 하나도 없으면 None. 0으로 채우면 '가동 안 함'과 '기침 없음'이
        # 구분되지 않고, 배수 계산에서 근거 없는 기준선이 된다.
        out.append(float(median(vals)) if vals else None)
    return out


def _dates_between(start: datetime, end: datetime) -> List[date]:
    """기준선 창에 포함된 현지 날짜들 — 기침이 0이던 날도 표본에 넣기 위함."""
    d, last = to_local(start).date(), to_local(end).date()
    out: List[date] = []
    while d <= last:
        out.append(d)
        d += timedelta(days=1)
    return out

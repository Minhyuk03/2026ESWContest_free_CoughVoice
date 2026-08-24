"""AlertEngine — 기침 이벤트가 들어올 때마다 알림 규칙을 평가한다 (P5, FR-07).

화면(S4)에는 규칙 카드가 있었지만 실제로 평가하는 코드가 없어 "1시간에 10회" 같은
조건이 동작하지 않았다. 이 모듈이 그 자리를 채운다.

설계상 유의점:
  - **시각은 UTC로 저장되고 야간 규칙은 현지 시각 기준이다.** 그대로 비교하면
    한국 기준 9시간이 어긋나 야간 알림이 대낮에 뜬다. LOCAL_TZ_OFFSET로 변환한다.
  - 같은 규칙·같은 대상이 연달아 울리지 않도록 cooldown_minutes 동안 억제한다.
    억제가 없으면 임계치를 넘긴 뒤 기침 한 번마다 알림이 쌓인다.
  - 문구는 비의료적 표현만 쓴다. '진단'·'증상 악화' 같은 표현은 쓰지 않는다.

웹훅 전송은 여기서 하지 않는다(send_webhook 자리만 남겨둠). 외부 전송은 실패·재시도
설계가 따로 필요하고, 대회 데모 범위에서는 DB 기록과 대시보드 표시로 충분하다.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Alert, AlertRule, CoughEvent, Person

# 야간 판정에 쓸 현지 시각 오프셋. 배포지가 바뀌면 이 값만 조정한다.
LOCAL_TZ_OFFSET = timedelta(hours=float(os.environ.get("COUGHID_TZ_OFFSET", "9")))


def _as_utc(dt: datetime) -> datetime:
    """SQLite가 tz를 버리므로 naive 값은 UTC로 간주한다 (저장 규약)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _local_hour(dt: datetime) -> int:
    return (_as_utc(dt) + LOCAL_TZ_OFFSET).hour


def _in_night(hour: int, start: int, end: int) -> bool:
    """22–06처럼 자정을 넘는 구간을 다룬다."""
    return start <= hour or hour < end if start > end else start <= hour < end


class AlertEngine:
    def evaluate(self, db: Session, event: CoughEvent) -> List[Alert]:
        """이벤트 1건을 받아 발동한 알림들을 생성해 반환한다."""
        rules = db.scalars(select(AlertRule).where(AlertRule.enabled.is_(True))).all()
        person = db.get(Person, event.person_id) if event.person_id else None
        fired: List[Alert] = []

        for rule in rules:
            if not self._targets(rule, person):
                continue
            message = self._check(db, rule, event, person)
            if message is None:
                continue
            if self._suppressed(db, rule, event):
                continue
            alert = Alert(person_id=event.person_id, rule=rule.name, message=message)
            db.add(alert)
            fired.append(alert)

        if fired:
            db.commit()
            for a in fired:
                db.refresh(a)
        return fired

    # ------------------------------------------------------------------ 판정
    def _targets(self, rule: AlertRule, person: Optional[Person]) -> bool:
        """대상 지정이 '전체 화자'가 아니면 별칭이 일치할 때만 평가한다."""
        t = (rule.target_text or "").strip()
        if t in ("", "—", "전체 화자"):
            return True
        return person is not None and person.alias == t

    def _check(self, db: Session, rule: AlertRule,
               event: CoughEvent, person: Optional[Person]) -> Optional[str]:
        who = f"{person.room + ' ' if person and person.room else ''}" \
              f"{person.alias if person else '미등록 화자'}"

        if rule.kind == AlertRule.KIND_UNKNOWN:
            if event.person_id is not None:
                return None
            return "등록되지 않은 화자의 기침이 감지되었습니다. 확인이 필요합니다."

        if rule.kind == AlertRule.KIND_NIGHT:
            hour = _local_hour(event.captured_at)
            if not _in_night(hour, rule.night_start_hour, rule.night_end_hour):
                return None

        n = self._count_recent(db, event, rule.window_minutes,
                               night=(rule.kind == AlertRule.KIND_NIGHT), rule=rule)
        if n < rule.threshold_count:
            return None

        span = (f"{rule.night_start_hour}–{rule.night_end_hour}시"
                if rule.kind == AlertRule.KIND_NIGHT
                else f"최근 {rule.window_minutes}분")
        return f"{who} · {span} 기침 {n}회 (기준 {rule.threshold_count}회). 확인이 필요합니다."

    def _count_recent(self, db: Session, event: CoughEvent, window_minutes: int,
                      night: bool, rule: AlertRule) -> int:
        since = _as_utc(event.captured_at) - timedelta(minutes=window_minutes)
        q = select(CoughEvent).where(CoughEvent.captured_at >= since.replace(tzinfo=None))
        # person_id가 None인 이벤트끼리는 '미등록' 한 묶음으로 센다
        q = q.where(CoughEvent.person_id.is_(None) if event.person_id is None
                    else CoughEvent.person_id == event.person_id)
        rows = db.scalars(q).all()
        if night:
            rows = [e for e in rows
                    if _in_night(_local_hour(e.captured_at),
                                 rule.night_start_hour, rule.night_end_hour)]
        return len(rows)

    def _suppressed(self, db: Session, rule: AlertRule, event: CoughEvent) -> bool:
        """쿨다운 안에 같은 규칙·같은 대상 알림이 이미 있으면 중복으로 본다.

        기준 시각은 **이벤트 발생 시각이 아니라 현재 시각**이다. 쿨다운의 의미가
        "사람에게 다시 알리지 않는 간격"이고, Alert.created_at도 저장 시각이라
        같은 시계로 비교해야 한다. 이벤트 시각으로 비교하면 엣지가 네트워크 복구 후
        밀린 이벤트를 몰아 보낼 때 억제가 풀려 알림이 쏟아진다.
        """
        if rule.cooldown_minutes <= 0:
            return False
        since = datetime.now(timezone.utc) - timedelta(minutes=rule.cooldown_minutes)
        q = select(Alert).where(Alert.rule == rule.name,
                                Alert.created_at >= since.replace(tzinfo=None))
        q = q.where(Alert.person_id.is_(None) if event.person_id is None
                    else Alert.person_id == event.person_id)
        return db.scalar(q) is not None


engine = AlertEngine()

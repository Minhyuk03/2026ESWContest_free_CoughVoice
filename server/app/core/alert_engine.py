"""AlertEngine — 기침 이벤트·증상 입력이 들어올 때마다 알림 규칙을 평가한다 (P5·P6, FR-07).

규칙은 두 갈래다.

**관찰 규칙** (KIND_COUNT·KIND_NIGHT·KIND_UNKNOWN)
    사용자가 직접 정한 절대 횟수 기준. 근거는 사용자의 관심사이지 임상 지침이 아니다.
    참고자료가 분명히 한 대로, 기침 횟수로 질환을 가르는 규칙은 성립하지 않는다
    (질환별 분포가 크게 겹친다 — core/guidance.py). 그래서 이 갈래의 알림은
    severity=info, source=탐색용으로 표시해 임상 근거가 있는 알림과 섞이지 않게 한다.

**임상 근거 규칙** (KIND_BASELINE·KIND_DURATION·KIND_URGENT)
    참고자료가 권고한 경고 구조를 그대로 옮긴 것:
      - 변화: 개인 기준선 대비 급증이 이어지는가
      - 기간: 2주/3주/4주(소아)/8주 경계를 넘겼는가
      - 긴급: 객혈·호흡곤란 등이 입력되면 횟수와 무관하게 즉시
    단 '변화' 경고의 2배 기준은 지침값이 아니라 탐색용이므로 severity=info로 둔다.
    경계값에 출처가 있는 것은 기간·긴급 두 가지다.

설계상 유의점:
  - **시각은 UTC로 저장되고 야간·일자 판정은 현지 시각 기준이다.** 그대로 비교하면
    한국 기준 9시간이 어긋나 야간 알림이 대낮에 뜬다. cough_metrics의 변환기를 쓴다.
  - 같은 규칙·같은 대상이 연달아 울리지 않도록 cooldown_minutes 동안 억제한다.
  - 문구는 core/guidance.py에서만 만든다. 진단 표현이 새어나가지 않게 하기 위함이다.

웹훅 전송은 여기서 하지 않는다(자리만 남겨둠). 외부 전송은 실패·재시도 설계가 따로
필요하고, 대회 데모 범위에서는 DB 기록과 대시보드 표시로 충분하다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Alert, AlertRule, CoughEvent, Person, SymptomReport
from . import guidance
from .cough_metrics import (
    LOCAL_TZ_OFFSET,  # noqa: F401  — 기존 임포트 경로 호환
    as_utc,
    duration_streak,
    hourly_baseline,
    in_night,
    local_hour,
    to_local,
    utc_naive,
)

# 기준선을 신뢰하려면 최소 이만큼의 관측이 필요하다. 이벤트가 거의 없는 상태에서
# "평소의 3배"를 말하면 1건이 3건이 된 것도 경고가 된다.
MIN_BASELINE_EVENTS = 10


class AlertEngine:
    # ------------------------------------------------------------- 진입점
    def evaluate(self, db: Session, event: CoughEvent) -> List[Alert]:
        """기침 이벤트 1건을 받아 발동한 알림들을 생성해 반환한다."""
        person = db.get(Person, event.person_id) if event.person_id else None
        return self._run(db, event.person_id, person,
                         lambda rule: self._check_event(db, rule, event, person))

    def evaluate_symptom(self, db: Session, report: SymptomReport) -> List[Alert]:
        """증상 입력 1건을 받아 긴급 안내를 평가한다.

        기침 이벤트와 별개의 진입점인 이유: 참고자료의 긴급 징후(객혈·호흡곤란·청색증·
        흉통·의식저하·SpO₂ 저하)는 **기침 횟수와 무관하게** 즉시 진료 대상이다.
        기침이 한 번도 없어도 발동해야 하므로 이벤트 경로에 얹을 수 없다.
        """
        person = db.get(Person, report.person_id) if report.person_id else None
        return self._run(db, report.person_id, person,
                         lambda rule: self._check_symptom(rule, report))

    def _run(self, db: Session, person_id: Optional[int],
             person: Optional[Person], check) -> List[Alert]:
        rules = db.scalars(select(AlertRule).where(AlertRule.enabled.is_(True))).all()
        fired: List[Alert] = []
        for rule in rules:
            if not self._targets(rule, person):
                continue
            verdict = check(rule)
            if verdict is None:
                continue
            message, severity, source = verdict
            if self._suppressed(db, rule, person_id):
                continue
            alert = Alert(person_id=person_id, rule=rule.name, message=message,
                          severity=severity, source=source)
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

    def _who(self, person: Optional[Person]) -> str:
        return (f"{person.room + ' ' if person and person.room else ''}"
                f"{person.alias if person else '미등록 화자'}")

    def _check_event(self, db: Session, rule: AlertRule, event: CoughEvent,
                     person: Optional[Person]) -> Optional[tuple]:
        who = self._who(person)

        if rule.kind == AlertRule.KIND_URGENT:
            return None                      # 증상 입력 경로에서만 평가한다

        if rule.kind == AlertRule.KIND_UNKNOWN:
            if event.person_id is not None:
                return None
            return ("등록되지 않은 화자의 기침이 감지되었습니다. 확인이 필요합니다.",
                    guidance.SEV_INFO, "사용자 지정 관찰 기준")

        if rule.kind == AlertRule.KIND_DURATION:
            return self._check_duration(db, rule, event, who)

        if rule.kind == AlertRule.KIND_BASELINE:
            return self._check_baseline(db, rule, event, who)

        # --- 절대 횟수 관찰 규칙 ---
        if rule.kind == AlertRule.KIND_NIGHT:
            hour = local_hour(event.captured_at)
            if not in_night(hour, rule.night_start_hour, rule.night_end_hour):
                return None

        n = self._count_recent(db, event, rule.window_minutes,
                               night=(rule.kind == AlertRule.KIND_NIGHT), rule=rule)
        if n < rule.threshold_count:
            return None
        span = (f"{rule.night_start_hour}–{rule.night_end_hour}시"
                if rule.kind == AlertRule.KIND_NIGHT
                else f"최근 {rule.window_minutes}분")
        return (f"{who} · {span} 기침 {n}회 (기준 {rule.threshold_count}회). 확인이 필요합니다.",
                guidance.SEV_INFO,
                "사용자 지정 관찰 기준 — 횟수만으로는 질환을 구분할 수 없습니다")

    # ------------------------------------------------------- 기간 경고
    def _check_duration(self, db: Session, rule: AlertRule,
                        event: CoughEvent, who: str) -> Optional[tuple]:
        """기침이 며칠째 이어지는지 보고, 넘긴 경계에 해당하는 안내를 고른다."""
        streak, start = duration_streak(
            db, person_id=event.person_id, allowed_gap_days=rule.allowed_gap_days,
            now=as_utc(event.captured_at))
        if streak < rule.duration_days:
            return None

        # 넘긴 경계 중 가장 높은 단계의 안내를 쓴다 (DURATION_GUIDANCE는 내림차순)
        for days, severity, text, source in guidance.DURATION_GUIDANCE:
            if streak >= days:
                since = f"{start:%m월 %d일}부터 " if start else ""
                return (f"{who} · {since}{streak}일째 기침이 관찰됩니다. {text}",
                        severity, source)

        return (f"{who} · {streak}일째 기침이 관찰됩니다. 경과를 지켜봐 주세요.",
                guidance.SEV_INFO, "사용자 지정 관찰 기준")

    # ------------------------------------------------------- 변화 경고
    def _check_baseline(self, db: Session, rule: AlertRule,
                        event: CoughEvent, who: str) -> Optional[tuple]:
        """개인 기준선 대비 몇 배로 늘었는지 본다.

        비교는 **같은 시간대끼리** 한다. 사람의 기침은 하루 안에서도 분포가 크게
        다르므로(기상 직후·취침 전), 하루 총량으로 비교하면 시간대 쏠림이 증가로 둔갑한다.
        """
        now = as_utc(event.captured_at)
        window_start = now - timedelta(hours=rule.sustain_hours)

        # 기준선에는 지금 판단하려는 구간을 넣지 않는다. 넣으면 자기 자신과 비교하게 된다.
        baseline = hourly_baseline(db, person_id=event.person_id, days=rule.baseline_days,
                                   now=now, exclude_recent_hours=rule.sustain_hours)

        base_events = self._count_between(db, event.person_id,
                                          now - timedelta(days=rule.baseline_days
                                                          + rule.sustain_hours / 24),
                                          window_start)
        if base_events < MIN_BASELINE_EVENTS:
            return None      # 기준선을 세울 만큼 관측이 쌓이지 않았다

        # 최근 창이 덮는 시간대들의 기준선 합 = '평소라면 이만큼'.
        # 정확히 sustain_hours칸만 더한다. 시작·끝 시각을 비교하며 도는 방식은
        # 경계 시간대를 양쪽에서 한 번씩 세어(예: 24시간 창인데 19시가 두 번)
        # 기댓값을 한 시간치 부풀리고, 그만큼 증가를 놓친다.
        expected = 0.0
        cursor = to_local(window_start).replace(minute=0, second=0, microsecond=0)
        for _ in range(max(rule.sustain_hours, 1)):
            v = baseline[cursor.hour]
            expected += v if v is not None else 0.0
            cursor += timedelta(hours=1)

        recent = self._count_between(db, event.person_id, window_start, now)
        # 기준선이 0인 사람에게 배수는 정의되지 않는다. 평소 거의 안 하던 사람이
        # 갑자기 하는 것도 의미가 있으므로 최소 기준선 1회를 깔고 비교한다.
        denom = max(expected, 1.0)
        ratio = recent / denom
        if ratio < rule.ratio_threshold:
            return None

        span = (f"{rule.sustain_hours}시간" if rule.sustain_hours < 48
                else f"{rule.sustain_hours // 24}일")
        return (f"{who} · 최근 {span} 기침 {recent}회로 평소 수준({expected:.0f}회)의 "
                f"{ratio:.1f}배입니다. 이런 상태가 며칠 이어지면 진료를 고려해 보세요.",
                guidance.SEV_INFO,
                "개인 기준선 대비 변화 — 탐색용 기준이며 임상 진단 경계값이 아닙니다")

    # ------------------------------------------------------- 긴급 경고
    def _check_symptom(self, rule: AlertRule,
                       report: SymptomReport) -> Optional[tuple]:
        if rule.kind != AlertRule.KIND_URGENT:
            return None
        reasons = guidance.urgent_reasons(report.codes(), report.spo2)
        if not reasons:
            return None
        return ("입력된 증상: " + " · ".join(reasons)
                + ". 기침 횟수와 관계없이 지금 진료를 받으세요.",
                guidance.SEV_URGENT,
                "CDC 호흡기 응급징후 · NHS 객혈 안내")

    # ------------------------------------------------------------------ 집계
    def _count_recent(self, db: Session, event: CoughEvent, window_minutes: int,
                      night: bool, rule: AlertRule) -> int:
        since = as_utc(event.captured_at) - timedelta(minutes=window_minutes)
        q = select(CoughEvent).where(CoughEvent.captured_at >= utc_naive(since))
        # person_id가 None인 이벤트끼리는 '미등록' 한 묶음으로 센다
        q = q.where(CoughEvent.person_id.is_(None) if event.person_id is None
                    else CoughEvent.person_id == event.person_id)
        rows = db.scalars(q).all()
        if night:
            rows = [e for e in rows
                    if in_night(local_hour(e.captured_at),
                                rule.night_start_hour, rule.night_end_hour)]
        return len(rows)

    def _count_between(self, db: Session, person_id: Optional[int],
                       start: datetime, end: datetime) -> int:
        q = select(CoughEvent).where(CoughEvent.captured_at >= utc_naive(start),
                                     CoughEvent.captured_at < utc_naive(end))
        q = q.where(CoughEvent.person_id.is_(None) if person_id is None
                    else CoughEvent.person_id == person_id)
        return len(db.scalars(q).all())

    def _suppressed(self, db: Session, rule: AlertRule,
                    person_id: Optional[int]) -> bool:
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
        q = q.where(Alert.person_id.is_(None) if person_id is None
                    else Alert.person_id == person_id)
        return db.scalar(q) is not None


engine = AlertEngine()

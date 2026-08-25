#!/usr/bin/env python3
"""P6 알림 규칙·지표 검증 — 메모리 DB에 합성 이벤트를 넣어 규칙이 의도대로 도는지 본다.

pytest 없이 그대로 실행한다:
    python3 tools/test_alert_rules.py

여기서 확인하는 것은 "규칙이 코드대로 도는가"이지 "임상적으로 맞는가"가 아니다.
경계값의 임상 근거는 server/app/core/guidance.py에 출처와 함께 적혀 있다.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

from sqlalchemy import create_engine                       # noqa: E402
from sqlalchemy.orm import sessionmaker                    # noqa: E402

from app.core import guidance                              # noqa: E402
from app.core.alert_engine import AlertEngine              # noqa: E402
from app.core.cough_metrics import burden, duration_streak, hourly_baseline  # noqa: E402
from app.models import Alert, AlertRule, Base, CoughEvent, Person, SymptomReport  # noqa: E402

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def fresh():
    """규칙 없는 빈 DB. 규칙은 시험마다 필요한 것만 넣는다."""
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()
    db.add(Person(id=1, alias="s01", room="301호"))
    db.commit()
    return db


def add_events(db, when_list, person_id=1):
    for t in when_list:
        db.add(CoughEvent(device_id="pi", captured_at=t.astimezone(timezone.utc).replace(tzinfo=None),
                          person_id=person_id, audio_path="x.wav"))
    db.commit()


def daily(days_ago_list, per_day=3, hour=10):
    """지정한 '며칠 전'들에 하루 per_day건씩."""
    out = []
    for d in days_ago_list:
        base = NOW - timedelta(days=d)
        out += [base.replace(hour=hour) + timedelta(minutes=5 * i) for i in range(per_day)]
    return out


def last_event(db):
    return db.query(CoughEvent).order_by(CoughEvent.captured_at.desc()).first()


# --------------------------------------------------------------- 지속일수
print("\n[지속일수 계산]")
db = fresh()
add_events(db, daily(range(0, 20)))          # 20일 연속
streak, start = duration_streak(db, person_id=1, now=NOW, allowed_gap_days=2)
check("연속 20일이면 20일로 센다", streak == 20, f"streak={streak}")

db = fresh()
add_events(db, daily([0, 1, 2, 3, 5, 6, 7]))  # 4일 전 하루 비었음
streak, _ = duration_streak(db, person_id=1, now=NOW, allowed_gap_days=2)
check("하루 빈 날은 지속을 끊지 않는다", streak == 8, f"streak={streak}")

db = fresh()
add_events(db, daily([0, 1, 2]) + daily([10, 11, 12]))  # 중간에 7일 공백
streak, _ = duration_streak(db, person_id=1, now=NOW, allowed_gap_days=2)
check("긴 공백은 지속을 끊는다", streak == 3, f"streak={streak}")

db = fresh()
add_events(db, daily([10, 11, 12]))          # 최근 활동 없음
streak, _ = duration_streak(db, person_id=1, now=NOW, allowed_gap_days=2)
check("최근 기침이 없으면 진행 중이 아니다", streak == 0, f"streak={streak}")

# --------------------------------------------------------------- 기간 경고
print("\n[기간 경고]")
engine = AlertEngine()


def duration_rule(db, days=14):
    db.add(AlertRule(name="기침 지속 기간", condition_text="", target_text="전체 화자",
                     enabled=True, kind=AlertRule.KIND_DURATION,
                     duration_days=days, allowed_gap_days=2, cooldown_minutes=0))
    db.commit()


db = fresh(); duration_rule(db)
add_events(db, daily(range(0, 10)))
check("10일이면 2주 규칙은 울리지 않는다", engine.evaluate(db, last_event(db)) == [])

db = fresh(); duration_rule(db)
add_events(db, daily(range(0, 15)))
fired = engine.evaluate(db, last_event(db))
check("15일이면 울린다", len(fired) == 1)
check("2주 문구는 '검진'이지 '진단'이 아니다",
      fired and "검진" in fired[0].message and "진단" not in fired[0].message,
      fired[0].message if fired else "")
check("출처가 질병관리청으로 기록된다",
      fired and "질병관리청" in (fired[0].source or ""), fired[0].source if fired else "")
check("심각도는 진료 안내", fired and fired[0].severity == guidance.SEV_ADVISORY)

db = fresh(); duration_rule(db)
add_events(db, daily(range(0, 60)))
fired = engine.evaluate(db, last_event(db))
check("60일이면 만성 기침(8주) 안내로 올라간다",
      fired and "만성" in fired[0].message, fired[0].message if fired else "")

# --------------------------------------------------------------- 변화 경고
print("\n[변화 경고]")


def baseline_rule(db, ratio=2.0, base_days=7, sustain=24):
    db.add(AlertRule(name="평소 대비 증가", condition_text="", target_text="전체 화자",
                     enabled=True, kind=AlertRule.KIND_BASELINE,
                     baseline_days=base_days, ratio_threshold=ratio,
                     sustain_hours=sustain, cooldown_minutes=0))
    db.commit()


db = fresh(); baseline_rule(db)
add_events(db, daily(range(1, 8), per_day=1))     # 기준선 표본 7건뿐
add_events(db, daily([0], per_day=20))
check("기준선 표본이 부족하면 울리지 않는다", engine.evaluate(db, last_event(db)) == [])

db = fresh(); baseline_rule(db)
add_events(db, daily(range(1, 8), per_day=3))     # 평소 하루 3회 (21건)
add_events(db, daily([0], per_day=3))             # 오늘도 평소 수준
check("평소 수준이면 울리지 않는다", engine.evaluate(db, last_event(db)) == [])

db = fresh(); baseline_rule(db)
add_events(db, daily(range(1, 8), per_day=3))
add_events(db, daily([0], per_day=12))            # 4배
fired = engine.evaluate(db, last_event(db))
check("기준선의 4배면 울린다", len(fired) == 1)
check("변화 경고는 탐색용으로 표시된다",
      fired and fired[0].severity == guidance.SEV_INFO
      and "임상 진단 경계값이 아닙니다" in (fired[0].source or ""),
      fired[0].source if fired else "")

# --------------------------------------------------------------- 긴급 경고
print("\n[긴급 경고]")


def urgent_rule(db):
    db.add(AlertRule(name="긴급 증상", condition_text="", target_text="전체 화자",
                     enabled=True, kind=AlertRule.KIND_URGENT, cooldown_minutes=0))
    db.commit()


db = fresh(); urgent_rule(db)
r = SymptomReport(person_id=1, symptoms="fever")
db.add(r); db.commit(); db.refresh(r)
check("발열만으로는 긴급이 아니다", engine.evaluate_symptom(db, r) == [])

db = fresh(); urgent_rule(db)
r = SymptomReport(person_id=1, symptoms="fever,hemoptysis")
db.add(r); db.commit(); db.refresh(r)
fired = engine.evaluate_symptom(db, r)
check("객혈이면 긴급", len(fired) == 1)
check("긴급은 횟수와 무관함을 문구에 밝힌다",
      fired and "횟수와 관계없이" in fired[0].message, fired[0].message if fired else "")
check("심각도 urgent", fired and fired[0].severity == guidance.SEV_URGENT)

db = fresh(); urgent_rule(db)
r = SymptomReport(person_id=1, symptoms="", spo2=88.0)
db.add(r); db.commit(); db.refresh(r)
fired = engine.evaluate_symptom(db, r)
check("SpO₂ 88%면 증상 코드 없이도 긴급", len(fired) == 1)

db = fresh(); urgent_rule(db)
add_events(db, daily(range(0, 30)))
check("기침 이벤트로는 긴급 규칙이 울리지 않는다", engine.evaluate(db, last_event(db)) == [])

# --------------------------------------------------------------- 부담 지표
print("\n[기침 부담 지표]")
db = fresh()
night = [NOW.replace(hour=16) + timedelta(minutes=i) for i in range(5)]     # 현지 01시 = 야간
day = [NOW.replace(hour=3) + timedelta(minutes=i) for i in range(3)]        # 현지 12시 = 주간
add_events(db, night + day)
b = burden(db, person_id=1, days=1, now=NOW + timedelta(hours=8))
check("총 발작 수", b.bout_count == 8, f"{b.bout_count}")
check("야간/주간 분리", (b.night_count, b.day_count) == (5, 3),
      f"night={b.night_count} day={b.day_count}")
check("최대 시간당 횟수", b.max_bouts_per_hour == 5.0, f"{b.max_bouts_per_hour}")
check("측정 불가 지표가 값 없이 이유와 함께 남는다",
      "individual_cough_count" in b.unavailable and b.unavailable["individual_cough_count"])
check("발작 정의가 응답에 실린다", "2초" in b.to_dict()["bout_definition"])

db = fresh()
add_events(db, daily(range(1, 8), per_day=2, hour=10))
base = hourly_baseline(db, person_id=1, days=7, now=NOW)
check("기준선은 24칸", len(base) == 24)
check("기침 있던 시간대(현지 19시)의 중앙값이 2", base[19] == 2.0, f"{base[19]}")
check("기침 없던 시간대는 0", base[3] == 0.0, f"{base[3]}")

# --------------------------------------------------------------- 하트비트
print("\n[하트비트 · 가동시간 보정]")
from app.api.devices import ONLINE_WINDOW, MIN_BEATS_PER_HOUR, covered_hours, is_online  # noqa: E402
from app.models import DeviceUptime                                                       # noqa: E402
from app.core.cough_metrics import utc_naive                                              # noqa: E402


def beat(db, when, count=MIN_BEATS_PER_HOUR, dev="rpi5-01"):
    """지정 시각의 정시 칸에 비트를 기록한다."""
    hour = when.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    db.add(DeviceUptime(device_id=dev, hour_utc=utc_naive(hour), beat_count=count,
                        first_seen=utc_naive(when), last_seen=utc_naive(when)))
    db.commit()


db = fresh()
check("하트비트가 없으면 생존 여부는 '모름'(None)", is_online(db, now=NOW) is None)

db = fresh(); beat(db, NOW - timedelta(seconds=60))
check("최근 비트가 있으면 온라인", is_online(db, now=NOW) is True)

db = fresh(); beat(db, NOW - timedelta(seconds=600))
check("오래된 비트면 오프라인", is_online(db, now=NOW) is False)

db = fresh(); beat(db, NOW - timedelta(hours=1), count=MIN_BEATS_PER_HOUR - 1)
check("비트가 적은 시간은 가동으로 치지 않는다",
      covered_hours(db, NOW - timedelta(days=1), NOW) == set())

# 기준선: 7일 중 3일만 가동, 나머지는 정전이라 가정
db = fresh()
add_events(db, daily([1, 2, 3], per_day=4, hour=10))     # 가동한 날에만 기침 4회
for d in (1, 2, 3):
    beat(db, (NOW - timedelta(days=d)).replace(hour=10))
base = hourly_baseline(db, person_id=1, days=7, now=NOW)
check("가동한 시간만 표본에 넣어 중앙값 4", base[19] == 4.0, f"{base[19]}")
check("가동 기록이 없는 시간대는 None", base[3] is None, f"{base[3]}")

# 같은 데이터인데 하트비트가 없으면 정전일이 0으로 섞여 기준선이 내려간다
db = fresh()
add_events(db, daily([1, 2, 3], per_day=4, hour=10))
base_nohb = hourly_baseline(db, person_id=1, days=7, now=NOW)
check("하트비트가 없으면 예전처럼 전 날짜를 세어 중앙값이 낮아진다",
      base_nohb[19] == 0.0, f"{base_nohb[19]}")

# --------------------------------------------------------------- 쿨다운
print("\n[쿨다운]")
db = fresh()
db.add(AlertRule(name="기침 지속 기간", condition_text="", target_text="전체 화자",
                 enabled=True, kind=AlertRule.KIND_DURATION,
                 duration_days=14, allowed_gap_days=2, cooldown_minutes=1440))
db.commit()
add_events(db, daily(range(0, 20)))
first = engine.evaluate(db, last_event(db))
second = engine.evaluate(db, last_event(db))
check("쿨다운 안에서는 다시 울리지 않는다", len(first) == 1 and second == [],
      f"first={len(first)} second={len(second)}")

print(f"\n{'='*46}\n통과 {PASS} · 실패 {FAIL}\n{'='*46}")
sys.exit(1 if FAIL else 0)

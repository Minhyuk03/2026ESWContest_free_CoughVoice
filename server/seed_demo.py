"""데모/화면검증용 시드 — 화자 3명 + 오늘 이벤트 + 알림 2건.

실행: .venv/bin/python seed_demo.py   (이미 시드된 DB에는 중복 생성하지 않음)
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import Alert, CoughEvent, Person

SPEAKERS = [("A", "301호", 12), ("B", "302호", 10), ("C", "305호", 11)]


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        if db.scalar(select(Person)):
            print("이미 화자가 있어 시드를 건너뜁니다")
            return

        persons = []
        for alias, room, samples in SPEAKERS:
            p = Person(alias=alias, room=room, sample_count=samples)
            db.add(p)
            persons.append(p)
        db.flush()

        # 재생 테스트용으로 기존 저장 오디오가 있으면 재활용
        wavs = sorted(Path("audio_store").glob("*.wav"))
        rng = random.Random(42)
        now = datetime.now().astimezone()
        for i in range(28):
            captured = now - timedelta(minutes=rng.randint(3, 60 * 14))
            registered = rng.random() < 0.8
            p = rng.choice(persons) if registered else None
            db.add(CoughEvent(
                device_id="rpi5-01",
                captured_at=captured,
                person_id=p.id if p else None,
                similarity=round(rng.uniform(0.80, 0.95), 2) if p else round(rng.uniform(0.30, 0.55), 2),
                peak_rms=round(rng.uniform(0.05, 0.4), 3),
                audio_path=str(rng.choice(wavs)) if wavs else "",
            ))

        a = persons[0]
        db.add(Alert(person_id=a.id, rule="이상 징후 (10회/1h)", message="1시간 내 기침 10회 (규칙: 10회/1h)"))
        db.add(Alert(person_id=persons[1].id, rule="야간 기침 (5회/야간)", message="야간 기침 5회"))
        db.commit()
        print("시드 완료: 화자 3명, 이벤트 28건, 알림 2건")
    finally:
        db.close()


if __name__ == "__main__":
    main()

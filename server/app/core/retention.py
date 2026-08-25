"""원음 보존 정책 (NFR-06) — 저장된 기침 WAV를 일정 기간 후 자동 삭제한다.

설계 배경: 화자 등록(enroll-from-events)·오식별 보정(M1 청취)·이력 재생은 저장된 원음에
의존한다. 그래서 "원음 즉시 삭제"를 문자 그대로 지키면 이 기능들이 전부 깨진다. 반대로
지금까지처럼 무기한 보관하면 사생활 위험이 남고 24시간 도는 서버의 디스크가 무한정 는다.

절충: 원음은 **최근 N일(기본 7일)만** 보관하고 초과분을 자동 삭제한다. 특징량(cough_score·
wheeze·gasp)과 이벤트 시각은 남으므로 지표·이력·알림은 그대로 유지된다. 참고자료의
"이벤트 시각·특징량만 보존" 권고와 정합적이다.

예외: 등록에 쓴 이벤트(enrolled=True)의 WAV는 남긴다 — 재등록·등록 구성 확인에 필요하다.
등록 샘플은 화자당 10~20개뿐이라 양이 작다.

삭제는 파일만 지운다. audio_path 컬럼은 그대로 두는데, /events/{id}/audio와 enroll은
이미 "파일이 없으면 404/에러"를 처리하므로 별도 마이그레이션 없이 자연스럽게 만료된다.

COUGHID_AUDIO_RETENTION_DAYS=0 이면 보존 정책을 끈다(무기한 보관).
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from ..db import SessionLocal
from ..models import CoughEvent

RETENTION_DAYS = int(os.environ.get("COUGHID_AUDIO_RETENTION_DAYS", "7"))
SWEEP_INTERVAL_S = 3600.0   # 한 시간마다 한 번 청소한다


def purge_expired_audio(db, now: datetime | None = None) -> tuple[int, int]:
    """보존 기간이 지난 (등록에 안 쓰인) 이벤트의 WAV 파일을 지운다.

    반환: (지운 파일 수, 확보한 바이트).  RETENTION_DAYS<=0이면 아무것도 안 한다.
    """
    if RETENTION_DAYS <= 0:
        return (0, 0)
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=RETENTION_DAYS)).replace(tzinfo=None)  # DB는 naive UTC
    rows = db.scalars(
        select(CoughEvent).where(
            CoughEvent.received_at < cutoff,
            CoughEvent.enrolled.is_(False),
        )
    ).all()
    deleted, freed = 0, 0
    for e in rows:
        if not e.audio_path:
            continue
        p = Path(e.audio_path)
        if not p.exists():
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            deleted += 1
            freed += size
        except OSError:
            # 이미 지워졌거나 접근 불가 — 다음 청소에서 다시 만난다. 여기서 죽지 않는다.
            continue
    return (deleted, freed)


def _worker() -> None:
    """기동 직후 한 번, 그 뒤 한 시간마다 만료 원음을 청소한다. 절대 죽지 않는다."""
    while True:
        db = SessionLocal()
        try:
            deleted, freed = purge_expired_audio(db)
            if deleted:
                print(f"[retention] 만료 원음 {deleted}건 삭제 "
                      f"({freed/1024:.0f}KB 확보, 보존 {RETENTION_DAYS}일)", flush=True)
        except Exception as exc:   # 무슨 일이 있어도 스레드를 유지한다
            print(f"[retention] 청소 오류(계속): {exc!r}", flush=True)
        finally:
            db.close()
        time.sleep(SWEEP_INTERVAL_S)


def start_retention_worker() -> None:
    if RETENTION_DAYS <= 0:
        print("[retention] COUGHID_AUDIO_RETENTION_DAYS=0 — 원음 보존 정책 비활성(무기한 보관)",
              flush=True)
        return
    threading.Thread(target=_worker, daemon=True, name="retention").start()
    print(f"[retention] 원음 보존 {RETENTION_DAYS}일 — 한 시간마다 만료분을 삭제한다", flush=True)

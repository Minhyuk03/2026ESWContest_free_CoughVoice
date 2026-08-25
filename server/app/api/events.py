"""IngestAPI — POST /events (엣지 수신), GET /events (이력 조회)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.alert_engine import engine as alert_engine
from ..db import get_db
from ..ml.cough_gate import gate
from ..ml.identifier import identifier
from ..models import CoughEvent, Person

router = APIRouter(tags=["기침 이벤트"])

AUDIO_DIR = Path("audio_store")
AUDIO_DIR.mkdir(exist_ok=True)


def iso_utc(dt: datetime) -> str:
    """SQLite는 tz를 버리고 저장하므로, naive 값은 UTC로 간주해 오프셋을 붙여 반환한다."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@router.post(
    "/events",
    status_code=201,
    summary="기침 이벤트 수신",
    description="엣지 디바이스가 검출한 기침 오디오(WAV)와 메타데이터를 업로드한다. "
    "서버는 오디오를 저장한 뒤 (1) 기침인지 판정하고 (2) 기침일 때만 화자를 식별한다. "
    "기침이 아니면 이벤트를 만들지 않고 200과 함께 rejected를 돌려준다. "
    "등록 화자가 없거나 유사도가 임계치 미만이면 unknown으로 남는다(FR-05).",
)
async def create_event(
    response: Response,
    audio: UploadFile = File(...),
    meta: str = Form(...),
    db: Session = Depends(get_db),
):
    m = json.loads(meta)

    # 같은 이벤트를 이미 받았으면 다시 처리하지 않는다. 엣지의 재전송 큐가 네트워크
    # 복구 후 같은 클립을 다시 보낼 수 있는데, 그대로 저장하면 기침 횟수가 부풀려지고
    # 알림 규칙이 실제보다 일찍 발동한다.
    event_id = m.get("event_id")
    if event_id:
        dup = db.scalar(select(CoughEvent).where(CoughEvent.event_id == event_id))
        if dup is not None:
            response.status_code = 200
            return {"id": dup.id, "person_id": dup.person_id,
                    "similarity": dup.similarity, "duplicate": True, "alerts": []}

    wav_path = AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    wav_path.write_bytes(await audio.read())

    # 1차 게이트 — 기침이 아니면 화자 식별로 넘기지 않는다.
    # 엣지 검출기는 에너지만 보므로 박수·문 닫기·말소리가 여기까지 올라온다.
    # 게이트와 식별은 동기 CPU 작업(torch)이다. async 핸들러 안에서 그대로 호출하면
    # 이벤트 루프가 막혀 처리 중에는 서버 전체가 멈춘다 — 실제로 기침 한 건을
    # 처리하는 동안 대시보드 로그인이 타임아웃됐다(2026-08-24). 스레드풀로 넘긴다.
    # analyze()는 한 번의 forward에서 판정 점수와 부가 지표(wheeze·gasp)를 함께 준다.
    # 판정에 쓰는 것은 cough_score 하나뿐이고 나머지는 기록용이다(P6).
    g = await run_in_threadpool(gate.analyze, str(wav_path))
    if not g.is_cough:
        wav_path.unlink(missing_ok=True)
        response.status_code = 200
        return {"id": None, "rejected": True, "reason": "not_cough",
                "cough_score": g.cough_score}
    cough_score = g.cough_score

    # 등록 임베딩이 있는 화자만 후보로 넘긴다 (P3 — 스텁 교체)
    registry = [(p.id, p.embedding_ref)
                for p in db.scalars(select(Person)).all() if p.embedding_ref]
    result = await run_in_threadpool(identifier.identify, str(wav_path), registry)

    captured = datetime.fromisoformat(m["captured_at"])
    if captured.tzinfo is not None:
        captured = captured.astimezone(timezone.utc)  # DB에는 UTC 기준으로 통일 저장
    # 미래 시각 방어: 엣지 시계가 어긋나거나 수동 POST로 미래 captured_at이 들어오면
    # 기준선·지연 통계가 왜곡된다(실제로 received_at보다 앞선 이벤트가 관측됐다).
    # 허용 오차(2분)를 넘는 미래 값은 수신 시각으로 당긴다.
    now_utc = datetime.now(timezone.utc)
    cap_aware = captured if captured.tzinfo is not None else captured.replace(tzinfo=timezone.utc)
    if cap_aware > now_utc + timedelta(minutes=2):
        captured = now_utc
    event = CoughEvent(
        event_id=event_id,
        device_id=m.get("device_id", "unknown"),
        captured_at=captured,
        person_id=result.person_id,
        similarity=result.similarity,
        peak_rms=m.get("peak_rms"),
        audio_path=str(wav_path),
        cough_score=g.cough_score,
        wheeze_prob=g.wheeze,
        gasp_prob=g.gasp,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    alerts = alert_engine.evaluate(db, event)   # P5 — 규칙 평가 (FR-07)
    return {"id": event.id, "person_id": event.person_id,
            "similarity": event.similarity, "cough_score": cough_score,
            "alerts": [{"rule": a.rule, "message": a.message} for a in alerts]}


@router.get(
    "/events",
    summary="기침 이벤트 이력 조회",
    description="최근 이벤트를 조회한다. unknown=true(미등록 화자만), person=화자ID, limit=개수 필터 지원.",
)
def list_events(
    limit: int = 50,
    unknown: Optional[bool] = None,
    person: Optional[int] = None,
    db: Session = Depends(get_db),
):
    q = select(CoughEvent).order_by(CoughEvent.received_at.desc()).limit(limit)
    if unknown:
        q = q.where(CoughEvent.person_id.is_(None))
    if person is not None:
        q = q.where(CoughEvent.person_id == person)
    rows = db.scalars(q).all()
    persons = {p.id: p for p in db.scalars(select(Person)).all()}
    out = []
    for e in rows:
        p = persons.get(e.person_id) if e.person_id else None
        out.append({
            "id": e.id,
            "device_id": e.device_id,
            "captured_at": iso_utc(e.captured_at),
            "received_at": iso_utc(e.received_at),
            "person_id": e.person_id,
            "person_alias": p.alias if p else None,
            "person_room": p.room if p else None,
            "similarity": e.similarity,
            "peak_rms": e.peak_rms,
            "cough_score": e.cough_score,
            # 미검증 부가 지표 — 판정에 쓰지 않는다(models.CoughEvent 참조)
            "wheeze_prob": e.wheeze_prob,
            "gasp_prob": e.gasp_prob,
        })
    return out


@router.get("/events/{event_id}/audio", summary="이벤트 오디오 재생", response_class=FileResponse)
def event_audio(event_id: int, db: Session = Depends(get_db)):
    e = db.get(CoughEvent, event_id)
    if e is None or not e.audio_path or not Path(e.audio_path).exists():
        raise HTTPException(status_code=404, detail="오디오를 찾을 수 없습니다")
    return FileResponse(e.audio_path, media_type="audio/wav")


class EventPersonBody(BaseModel):
    person_id: Optional[int] = None  # None = 미등록으로 변경


@router.patch("/events/{event_id}/person", summary="화자 수정 (오식별 보정, M1)")
def update_event_person(event_id: int, body: EventPersonBody, db: Session = Depends(get_db)):
    e = db.get(CoughEvent, event_id)
    if e is None:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    if body.person_id is not None and db.get(Person, body.person_id) is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    e.person_id = body.person_id
    db.commit()
    return {"ok": True}

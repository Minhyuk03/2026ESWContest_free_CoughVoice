"""IngestAPI — POST /events (엣지 수신), GET /events (이력 조회)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
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
    "서버는 오디오를 저장하고 화자 식별을 수행한다(현재는 스텁 — 항상 unknown).",
)
async def create_event(
    audio: UploadFile = File(...),
    meta: str = Form(...),
    db: Session = Depends(get_db),
):
    m = json.loads(meta)
    wav_path = AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    wav_path.write_bytes(await audio.read())

    result = identifier.identify(str(wav_path))  # P2: 항상 unknown

    captured = datetime.fromisoformat(m["captured_at"])
    if captured.tzinfo is not None:
        captured = captured.astimezone(timezone.utc)  # DB에는 UTC 기준으로 통일 저장
    event = CoughEvent(
        device_id=m.get("device_id", "unknown"),
        captured_at=captured,
        person_id=result.person_id,
        similarity=result.similarity,
        peak_rms=m.get("peak_rms"),
        audio_path=str(wav_path),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    # P5에서 여기에 AlertEngine.evaluate() + WebSocket 브로드캐스트 추가
    return {"id": event.id, "person_id": event.person_id, "similarity": event.similarity}


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

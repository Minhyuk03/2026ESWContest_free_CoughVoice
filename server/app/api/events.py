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
from .security import require_device_token
from ..core import bout as bout_mod
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
    _token: None = Depends(require_device_token),
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

    # 기침 원음은 등록·오식별 보정(청취)·이력 재생에 필요해 저장한다. 무기한은 아니고
    # core/retention.py가 기본 7일 후 자동 삭제한다(NFR-06). 비기침은 아래에서 즉시 지운다.
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
    # 상위 2인을 남긴다. 규약: person_id가 있으면 person_id/similarity = 1등,
    # runner_up_* = 2등. person_id가 None(임계치 미달)이면 runner_up_* 에 **1등**을
    # 넣는다 — 그러지 않으면 1등이 누구였는지가 사라져 발작 단위 재판정을 못 한다.
    if result.person_id is None:
        runner_id, runner_sim = result.top_id, result.similarity
    else:
        runner_id, runner_sim = result.runner_up_id, result.runner_up_similarity

    event = CoughEvent(
        event_id=event_id,
        device_id=m.get("device_id", "unknown"),
        captured_at=captured,
        person_id=result.person_id,
        similarity=result.similarity,
        runner_up_id=runner_id,
        runner_up_sim=runner_sim,
        peak_rms=m.get("peak_rms"),
        audio_path=str(wav_path),
        cough_score=g.cough_score,
        wheeze_prob=g.wheeze,
        gasp_prob=g.gasp,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # 발작 단위 재판정 — 클립 하나가 아니라 60초 안의 연속 기침을 묶어 화자를 정한다.
    # 알림 평가보다 **먼저** 해야 한다. 알림은 화자별로 세므로, 판정이 바뀐 뒤에
    # 세지 않으면 방금 이름이 붙은 앞선 기침들이 집계에서 빠진다.
    bout = bout_mod.judge(db, event) if bout_mod.ENABLED else None
    if bout is not None:
        db.refresh(event)

    alerts = alert_engine.evaluate(db, event)   # P5 — 규칙 평가 (FR-07)
    return {"id": event.id, "person_id": event.person_id,
            "similarity": event.similarity, "cough_score": cough_score,
            "bout": bout.as_dict() if bout else None,
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
        # 원음은 보존 기간(기본 7일)이 지나면 지워진다(core/retention.py). 화면이
        # 재생 버튼을 항상 띄우면 눌러 봐야 404를 만나므로, 파일이 실제로 있는지
        # 여기서 확인해 함께 내려준다.
        has_audio = bool(e.audio_path) and Path(e.audio_path).exists()
        out.append({
            "id": e.id,
            "audio_available": has_audio,
            "enrolled": bool(e.enrolled),
            "device_id": e.device_id,
            "captured_at": iso_utc(e.captured_at),
            "received_at": iso_utc(e.received_at),
            "person_id": e.person_id,
            "person_alias": p.alias if p else None,
            "person_room": p.room if p else None,
            "similarity": e.similarity,
            # 사람이 지정한 건이면 similarity는 지정 이전 모델 점수다 — 화면이
            # 그것을 지금 라벨의 확신도처럼 보여 주지 않도록 출처를 함께 준다.
            "person_source": e.person_source or "model",
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


class EventAssignBody(BaseModel):
    event_ids: list[int]
    person_id: Optional[int] = None   # None = 미등록으로 되돌리기


@router.post("/events/assign", summary="여러 이벤트의 화자를 한 번에 지정 (미등록 → 기존 화자 연결)")
def assign_events(body: EventAssignBody, db: Session = Depends(get_db)):
    """미등록으로 남은 기침들을 골라 기존 화자에 붙인다.

    **식별 모델은 바뀌지 않는다.** 여기서 바뀌는 것은 이력·통계에 쓰이는 라벨뿐이고,
    다음 기침을 같은 사람으로 알아보게 하려면 화자 등록(재등록)으로 임베딩을 갱신해야
    한다. 그래야 "연결했는데 왜 또 미등록이지?"가 되지 않는다.
    """
    if body.person_id is not None and db.get(Person, body.person_id) is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    if not body.event_ids:
        raise HTTPException(status_code=400, detail="선택된 이벤트가 없습니다")
    rows = db.scalars(select(CoughEvent).where(CoughEvent.id.in_(body.event_ids))).all()
    for e in rows:
        e.person_id = body.person_id
        e.person_source = "manual"
    db.commit()
    return {"ok": True, "updated": len(rows)}


@router.get("/audio-policy", summary="원음 보존 정책")
def audio_policy():
    """화면이 "무엇을 재생하는가"를 정확히 설명할 수 있도록 실제 정책값을 준다.

    화면마다 "원본 음성 비보존"이라고 적어 두고 재생 버튼을 함께 두면 사용자는
    둘 중 무엇이 사실인지 알 수 없다. 서버가 실제 설정값을 내려준다.
    """
    from ..core.retention import RETENTION_DAYS
    keeps = RETENTION_DAYS > 0
    return {
        "retention_days": RETENTION_DAYS,
        "enabled": keeps,
        "summary": (f"감지된 기침 소리는 {RETENTION_DAYS}일 동안만 보관하고 자동으로 지웁니다."
                    if keeps else "현재 설정에서는 기침 소리를 자동 삭제하지 않습니다."),
        "detail": (
            "화자 등록에 사용한 기침은 재등록을 위해 계속 보관합니다. "
            "화자를 구분하는 특징 정보(임베딩)와 발생 시각·지표는 소리를 지운 뒤에도 남아 "
            "이력과 통계는 그대로 유지됩니다."
            if keeps else
            "COUGHID_AUDIO_RETENTION_DAYS 값이 0이라 보존 기간 제한이 꺼져 있습니다."
        ),
    }


@router.patch("/events/{event_id}/person", summary="화자 수정 (오식별 보정, M1)")
def update_event_person(event_id: int, body: EventPersonBody, db: Session = Depends(get_db)):
    e = db.get(CoughEvent, event_id)
    if e is None:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    if body.person_id is not None and db.get(Person, body.person_id) is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    e.person_id = body.person_id
    e.person_source = "manual"
    db.commit()
    return {"ok": True}

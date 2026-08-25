"""PersonsAPI — 화자 관리 (S3·S3a).

원본 음성은 보존하지 않고 임베딩만 저장한다는 원칙(NFR-06)에 따라
여기서는 별칭·호실·샘플 수 등 메타데이터만 다룬다.
"""
from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..ml.identifier import identifier
from ..models import CoughEvent, Person


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).isoformat()

router = APIRouter(prefix="/persons", tags=["화자 관리"])

MIN_ENROLL_SAMPLES = 5   # 이보다 적으면 평균 임베딩이 그날의 발성 하나에 끌려간다


class PersonBody(BaseModel):
    alias: str
    room: Optional[str] = None
    sample_count: int = 0


def _person_row(db: Session, p: Person) -> dict:
    last = db.scalar(
        select(func.max(CoughEvent.captured_at)).where(CoughEvent.person_id == p.id)
    )
    return {
        "id": p.id,
        "alias": p.alias,
        "room": p.room,
        "sample_count": p.sample_count or 0,
        "created_at": _iso_utc(p.created_at),
        "last_cough_at": _iso_utc(last),
    }


@router.get("", summary="화자 목록")
def list_persons(db: Session = Depends(get_db)):
    return [_person_row(db, p) for p in db.scalars(select(Person).order_by(Person.id)).all()]


@router.post("", status_code=201, summary="화자 등록")
def create_person(body: PersonBody, db: Session = Depends(get_db)):
    if db.scalar(select(Person).where(Person.alias == body.alias)):
        raise HTTPException(status_code=409, detail="이미 존재하는 별칭입니다")
    p = Person(alias=body.alias, room=body.room, sample_count=body.sample_count)
    db.add(p)
    db.commit()
    db.refresh(p)
    return _person_row(db, p)


@router.patch("/{person_id}", summary="화자 수정 (재등록 시 샘플 수 갱신 포함)")
def update_person(person_id: int, body: PersonBody, db: Session = Depends(get_db)):
    p = db.get(Person, person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    p.alias = body.alias
    p.room = body.room
    if body.sample_count:
        p.sample_count = body.sample_count
    db.commit()
    return _person_row(db, p)


class EnrollFromEventsBody(BaseModel):
    event_ids: List[int]


def _enrollment_report(events: List[CoughEvent]) -> dict:
    """등록 세트가 어떤 것들로 이뤄졌는지 요약한다.

    등록 품질은 나중에 되짚기가 어렵다 — 임베딩 평균만 남고 어떤 샘플로 만들었는지는
    사라지기 때문이다(2026-08-25에 화자 2명의 등록 구성을 알아내려고 이벤트 시각을
    역추적해야 했다). 등록 시점에 요약을 돌려주고 화면이 보여 주게 한다.

    수치를 판정에 쓰지는 않는다. 어떤 등록 구성이 잘 되는지는 아직 모른다 —
    같은 날 실측에서 5개·5분 단일 세션 등록(10/10 정답)이 17개·16시간 분산 등록
    (11/20 정답)보다 나았다. 표본이 작아 결론을 낼 수 없으므로 기록만 남긴다.
    """
    times = sorted(_as_utc_dt(e.captured_at) for e in events)
    scores = [e.cough_score for e in events if e.cough_score is not None]
    span_min = (times[-1] - times[0]).total_seconds() / 60 if len(times) > 1 else 0.0
    return {
        "sample_count": len(events),
        "time_span_minutes": round(span_min, 1),
        "first_sample_at": _iso_utc(times[0]) if times else None,
        "last_sample_at": _iso_utc(times[-1]) if times else None,
        "distinct_days": len({t.date() for t in times}),
        "gate_score_mean": round(sum(scores) / len(scores), 4) if scores else None,
        "gate_score_min": round(min(scores), 4) if scores else None,
        "scored_samples": len(scores),
    }


def _as_utc_dt(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


@router.post("/{person_id}/enroll-from-events",
             summary="이미 수신된 기침 이벤트로 등록 (S3a) — 권장 경로",
             description="장치가 보내 저장해 둔 기침 오디오를 골라 평균 임베딩을 만든다. "
                         "등록과 식별이 같은 캡처 경로를 타므로 파일을 따로 옮길 필요가 없고, "
                         "마이크·샘플레이트가 달라 생기는 불일치도 없다. "
                         "**업로드 방식(/samples) 대신 이 경로를 써야 한다** — 근거는 그쪽 설명 참조.")
async def enroll_from_events(person_id: int, body: EnrollFromEventsBody,
                             db: Session = Depends(get_db)):
    p = db.get(Person, person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    if len(body.event_ids) < MIN_ENROLL_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"등록에는 최소 {MIN_ENROLL_SAMPLES}개 샘플이 필요합니다")

    paths = []
    for eid in body.event_ids:
        e = db.get(CoughEvent, eid)
        if e is None or not e.audio_path or not Path(e.audio_path).exists():
            raise HTTPException(status_code=404, detail=f"이벤트 {eid}의 오디오가 없습니다")
        paths.append(e.audio_path)

    try:
        blob, n = await run_in_threadpool(identifier.enroll, paths)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"임베딩 생성 실패: {exc}")

    p.embedding_ref = blob
    p.sample_count = n
    db.commit()

    # 등록에 쓴 이벤트는 본인 것으로 확정된 셈이니 화자를 붙여 둔다.
    used = []
    for eid in body.event_ids:
        ev = db.get(CoughEvent, eid)
        if ev is not None:
            ev.person_id = person_id
            used.append(ev)
    db.commit()

    row = _person_row(db, p)
    row["enrollment"] = _enrollment_report(used)
    return row


@router.post("/{person_id}/samples", summary="등록용 샘플 업로드 (권장하지 않음)",
             deprecated=True,
             description="⚠ **권장하지 않는다. /enroll-from-events를 쓸 것.** "
                         "외부에서 녹음한 파일로 등록하면 식별이 뒤집힐 수 있다 "
                         "(2026-08-25 실측: 같은 검증 세트에서 동일인 0.419 / 타인 0.636으로 "
                         "타인이 더 높게 나왔다. 같은 세트를 엣지 클립으로 등록하면 "
                         "동일인 0.504 / 타인 0.364로 정상). "
                         "임베딩이 화자보다 녹음 경로를 크게 반영하기 때문이다. "
                         "원본 음성은 저장하지 않고 임시 파일로만 쓰고 지운다(NFR-06).")
async def upload_samples(person_id: int,
                         files: List[UploadFile] = File(...),
                         db: Session = Depends(get_db)):
    p = db.get(Person, person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    if len(files) < MIN_ENROLL_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"등록에는 최소 {MIN_ENROLL_SAMPLES}개 샘플이 필요합니다")

    tmp_dir = Path(tempfile.mkdtemp(prefix="enroll_"))
    try:
        paths = []
        for i, f in enumerate(files):
            q = tmp_dir / f"{i:03d}.wav"
            q.write_bytes(await f.read())
            paths.append(str(q))
        try:
            blob, n = await run_in_threadpool(identifier.enroll, paths)
        except Exception as exc:   # 포맷 오류·모델 로딩 실패를 400으로 되돌린다
            raise HTTPException(status_code=400, detail=f"임베딩 생성 실패: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)   # 원본 음성 비보존

    p.embedding_ref = blob
    p.sample_count = n
    db.commit()

    # 호출자가 OpenAPI 설명을 안 읽었을 수 있으므로 응답에도 남긴다.
    row = _person_row(db, p)
    row["warning"] = (
        "업로드한 파일로 등록했습니다. 엣지 장치가 녹음한 클립과 특성이 달라 "
        "식별 정확도가 크게 떨어지거나 뒤집힐 수 있습니다. "
        "/persons/{id}/enroll-from-events 로 다시 등록하는 것을 권합니다.")
    return row


@router.delete("/{person_id}", summary="화자 삭제")
def delete_person(person_id: int, db: Session = Depends(get_db)):
    p = db.get(Person, person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="화자를 찾을 수 없습니다")
    # 이벤트는 남기되 미등록(unknown) 처리 — 이력 보존
    for e in db.scalars(select(CoughEvent).where(CoughEvent.person_id == person_id)).all():
        e.person_id = None
    db.delete(p)
    db.commit()
    return {"ok": True}

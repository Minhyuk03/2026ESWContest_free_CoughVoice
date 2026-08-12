"""PersonsAPI — 화자 관리 (S3·S3a).

원본 음성은 보존하지 않고 임베딩만 저장한다는 원칙(NFR-06)에 따라
여기서는 별칭·호실·샘플 수 등 메타데이터만 다룬다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import CoughEvent, Person


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).isoformat()

router = APIRouter(prefix="/persons", tags=["화자 관리"])


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

"""AuthAPI — 대시보드 로그인 (S0).

비밀번호는 PBKDF2-SHA256 해시로만 저장하고, 로그인 성공 시 랜덤 토큰을 발급한다.
토큰은 서버 메모리에 유지된다 — 재시작하면 재로그인 필요 (개발/데모 수준으로 충분).
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User

router = APIRouter(prefix="/auth", tags=["인증"])

# token -> username (메모리 세션 저장소)
_sessions: Dict[str, str] = {}


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()


def seed_admin(db: Session) -> None:
    """admin00 계정이 없으면 생성한다 (초기 관리자)."""
    if db.scalar(select(User).where(User.username == "admin00")):
        return
    salt = secrets.token_hex(16)
    db.add(User(username="admin00", password_hash=hash_password("admin00", salt), salt=salt, role="admin"))
    db.commit()


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/login", summary="로그인", description="아이디/비밀번호 검증 후 Bearer 토큰을 발급한다.")
def login(body: LoginBody, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or hash_password(body.password, user.salt) != user.password_hash:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다")
    token = secrets.token_hex(32)
    _sessions[token] = user.username
    return {"token": token, "username": user.username, "role": user.role}


def current_user(authorization: Optional[str] = Header(None)) -> str:
    """`Authorization: Bearer <token>` 헤더를 검증하는 의존성."""
    if authorization and authorization.startswith("Bearer "):
        username = _sessions.get(authorization[7:])
        if username:
            return username
    raise HTTPException(status_code=401, detail="로그인이 필요합니다")


@router.get("/me", summary="세션 확인")
def me(username: str = Depends(current_user)):
    return {"username": username}


@router.post("/logout", summary="로그아웃")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        _sessions.pop(authorization[7:], None)
    return {"ok": True}

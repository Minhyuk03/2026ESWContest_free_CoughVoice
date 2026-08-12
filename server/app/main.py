"""FastAPI 엔트리 — uvicorn app.main:app --reload --host 0.0.0.0"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.events import router as events_router
from .db import init_db

app = FastAPI(
    title="Cough-ID API — 기침 화자 식별 시스템",
    description="엣지(라즈베리파이)에서 검출된 기침 이벤트를 수신·저장하고, "
    "화자 식별 결과와 이력을 제공하는 API. 제26회 임베디드SW 경진대회 출품작.",
    version="0.1.0",
)
# 대시보드가 다른 기기(Mac 등)의 브라우저에서 이 서버로 접속하므로 전체 허용.
# 핫스팟 IP가 매번 바뀌어 origin을 고정할 수 없는 개발 단계 한정 설정.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(events_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health", summary="서버 상태 확인", description="서버가 살아있는지 확인하는 헬스체크.")
def health():
    return {"status": "ok"}

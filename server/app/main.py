"""FastAPI 엔트리 — uvicorn app.main:app --reload --host 0.0.0.0"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.alerts import router as alerts_router, seed_rules
from .api.auth import router as auth_router, seed_admin
from .api.events import router as events_router
from .api.persons import router as persons_router
from .api.stats import router as stats_router
from .api.symptoms import router as symptoms_router
from .db import SessionLocal, init_db

app = FastAPI(
    title="Cough-ID API — 기침 화자 식별 시스템",
    description="엣지(라즈베리파이)에서 검출된 기침 이벤트를 수신·저장하고, "
    "화자 식별 결과와 이력을 제공하는 API. 제24회 임베디드SW 경진대회 출품작.",
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
app.include_router(auth_router)
app.include_router(persons_router)
app.include_router(alerts_router)
app.include_router(stats_router)
app.include_router(symptoms_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    db = SessionLocal()
    try:
        seed_admin(db)   # admin00 초기 계정
        seed_rules(db)   # 기본 알림 규칙 3종
    finally:
        db.close()


@app.get("/health", summary="서버 상태 확인", description="서버가 살아있는지 확인하는 헬스체크.")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# 웹 대시보드 정적 서빙
#
# 빌드 결과(dashboard/dist)가 있으면 이 서버가 직접 서빙한다. 그러면 프로세스와
# 포트가 하나로 줄고, 대시보드가 API와 같은 출처에서 열려 CORS도 서버 주소 입력도
# 필요 없어진다. 빌드가 없으면 API 전용으로 동작한다(개발 중에는 vite dev 서버 사용).
#
# **이 블록은 모든 API 라우터 등록 뒤에 와야 한다.** catch-all이 먼저 잡히면
# API 요청이 index.html로 응답된다.
DIST = Path(__file__).resolve().parents[2] / "dashboard" / "dist"

if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        """React Router가 클라이언트에서 경로를 처리하므로 없는 경로는 index.html로 넘긴다."""
        candidate = DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)          # favicon 등 루트 정적 파일
        index = DIST / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="대시보드 빌드를 찾을 수 없습니다")
        return FileResponse(index)

"""FastAPI 엔트리 — uvicorn app.main:app --reload --host 0.0.0.0"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.alerts import router as alerts_router, seed_rules
from .api.auth import router as auth_router, seed_admin
from .api.devices import router as devices_router
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
app.include_router(devices_router)
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

    from .api.security import token_configured
    if not token_configured():
        print("[startup] ⚠ COUGHID_DEVICE_TOKEN 미설정 — POST /events·/heartbeat가 "
              "무인증입니다. LAN의 누구나 가짜 이벤트를 주입할 수 있으니 상시 운영에서는 "
              "서버·엣지 양쪽에 동일 토큰을 설정하세요.", flush=True)


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
DIST_RESOLVED = DIST.resolve()


def _safe_static(full_path: str) -> Optional[Path]:
    """DIST 안의 실제 파일이면 그 경로를, 아니면 None을 돌려준다.

    Starlette는 URL을 디코드해서 넘기므로 `%2e%2e%2f`(../)가 그대로 들어온다.
    `DIST / full_path`를 그대로 쓰면 DIST 밖 파일(예: server/cough_id.db — 비밀번호
    해시가 든 DB)까지 서빙돼 경로 순회로 임의 파일이 유출된다. resolve() 후
    DIST 하위인지 반드시 검증한다.
    """
    if not full_path:
        return None
    try:
        candidate = (DIST / full_path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate == DIST_RESOLVED or DIST_RESOLVED not in candidate.parents:
        return None                              # DIST 밖 → 거부
    return candidate if candidate.is_file() else None


if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        """React Router가 클라이언트에서 경로를 처리하므로 없는 경로는 index.html로 넘긴다."""
        static = _safe_static(full_path)
        if static is not None:
            return FileResponse(static)             # favicon 등 루트 정적 파일
        index = DIST / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="대시보드 빌드를 찾을 수 없습니다")
        return FileResponse(index)

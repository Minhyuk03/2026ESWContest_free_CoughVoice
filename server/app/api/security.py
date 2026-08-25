"""장치 인증 — 엣지가 서버로 데이터를 넣을 때 쓰는 공유 토큰.

`POST /events`·`POST /heartbeat`는 지금까지 무인증이었다. LAN에 붙은 누구나 가짜 기침
이벤트를 주입해 기침 횟수·알림·개인 기준선을 오염시킬 수 있었다(실제로 수동 POST로
들어온 테스트 이벤트가 통계를 흔든 사례가 있다).

`COUGHID_DEVICE_TOKEN` 환경변수가 설정돼 있으면 두 엔드포인트에 `X-Device-Token` 헤더를
요구한다. **설정돼 있지 않으면 통과시킨다** — 토큰을 양쪽(서버·엣지)에 넣기 전에 강제하면
기존 배포의 수집이 조용히 끊기기 때문이다. 대신 기동 시 경고를 찍는다.
상시 운영에서는 반드시 설정할 것.

엣지 쪽은 edge/event_sender.py가 같은 값을 `X-Device-Token`으로 보낸다.
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException

DEVICE_TOKEN = os.environ.get("COUGHID_DEVICE_TOKEN", "").strip()


def token_configured() -> bool:
    return bool(DEVICE_TOKEN)


def require_device_token(x_device_token: Optional[str] = Header(None)) -> None:
    """엣지 데이터 수신 엔드포인트용 의존성.

    토큰이 설정돼 있지 않으면(호환 모드) 통과. 설정돼 있으면 상수 시간 비교로 검증한다.
    """
    if not DEVICE_TOKEN:
        return
    if not x_device_token or not secrets.compare_digest(x_device_token, DEVICE_TOKEN):
        raise HTTPException(status_code=401, detail="장치 토큰이 유효하지 않습니다")

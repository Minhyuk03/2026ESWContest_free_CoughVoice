#!/usr/bin/env python3
"""엣지 디스크 큐 회귀 테스트 — 2026-08-25 장애 재발 방지.

그날 무슨 일이 있었나:
    파이의 mDNS 해석이 간헐적으로 실패해 이벤트가 디스크 큐에 쌓였다. 그러다 전송 도중
    프로세스가 죽으면서 0바이트 큐 파일이 남았고, `json.loads`가 던진 예외가 데몬
    스레드인 재시도 루프를 조용히 끝내버렸다. 그 뒤로 재시도가 영원히 멈춰
    이벤트가 4.5시간 늦게 도착하고 큐가 4,314파일 177MB까지 자랐다
    (SD 여유 977MB, 사용률 93%).

여기서 지키는 것:
    1. 손상된 큐 파일이 있어도 재시도 스레드가 죽지 않는다
    2. 큐가 상한을 넘으면 오래된 것부터 버린다
    3. 서버가 거절하는 항목 하나가 뒤의 정상 항목을 막지 않는다
    4. 서버에 닿지 못하면 항목을 버리지 않고 남긴다

실행:  python3 tools/test_edge_queue.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "edge"))

import event_sender as es  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


class FakeSender(es.EventSender):
    """스레드와 네트워크 없이 큐 로직만 시험한다."""

    def __init__(self, results):
        self.endpoint = "http://x/events"
        self.heartbeat_endpoint = "http://x/heartbeat"
        self.timeout = 1.0
        self.max_backoff = 1.0
        self.device_id = "test"
        self.results = results           # 파일 stem → 반환값, 또는 고정 문자열
        self.posted = []
        import threading
        self._stop = threading.Event()

    def _post(self, wav_bytes, meta):
        self.posted.append(meta.get("tag"))
        r = self.results
        return r if isinstance(r, str) else r.get(meta.get("tag"), "ok")


def setup_queue(tmp):
    es.QUEUE_DIR = Path(tmp)
    es.BAD_DIR = Path(tmp) / "bad"
    es.QUEUE_DIR.mkdir(exist_ok=True)


def put(tag, mtime=None, meta_bytes=None, wav_bytes=b"RIFFdata"):
    stem = es.QUEUE_DIR / f"{tag}"
    stem.with_suffix(".wav").write_bytes(wav_bytes)
    if meta_bytes is None:
        stem.with_suffix(".json").write_text(json.dumps({"tag": tag}))
    else:
        stem.with_suffix(".json").write_bytes(meta_bytes)
    if mtime:
        for q in (stem.with_suffix(".wav"), stem.with_suffix(".json")):
            os.utime(q, (mtime, mtime))
    return stem


def names():
    return sorted(p.stem for p in es.QUEUE_DIR.glob("*.json"))


# --------------------------------------------------------------- 1
print("\n[손상된 큐 파일]")
tmp = tempfile.mkdtemp(); setup_queue(tmp)
put("good1"); put("broken", meta_bytes=b"")          # 0바이트 — 그날의 원인
put("good2")
s = FakeSender("ok")
sent, reachable = s._drain_once({})
check("손상 파일이 있어도 예외 없이 한 바퀴 돈다", sent is True and reachable is True)
check("정상 항목 2건은 전송됐다", sorted(x for x in s.posted if x) == ["good1", "good2"],
      f"{s.posted}")
check("손상 항목은 큐에서 제거된다", names() == [], f"{names()}")
shutil.rmtree(tmp)

tmp = tempfile.mkdtemp(); setup_queue(tmp)
put("orphan"); (es.QUEUE_DIR / "orphan.wav").unlink()
s = FakeSender("ok"); s._drain_once({})
check("짝이 없는 메타 파일도 정리된다", names() == [], f"{names()}")
shutil.rmtree(tmp)

# --------------------------------------------------------------- 2
print("\n[큐 상한]")
tmp = tempfile.mkdtemp(); setup_queue(tmp)
es.QUEUE_MAX_EVENTS = 5
for i in range(9):
    put(f"e{i}", mtime=1000 + i)
FakeSender("ok")._trim_queue()
check("상한을 넘으면 오래된 것부터 버린다", names() == ["e4", "e5", "e6", "e7", "e8"], f"{names()}")
check("남은 개수가 상한과 같다", len(names()) == 5)
shutil.rmtree(tmp)

# --------------------------------------------------------------- 3
print("\n[서버가 거절하는 항목]")
tmp = tempfile.mkdtemp(); setup_queue(tmp)
es.QUEUE_MAX_EVENTS = 500
put("bad", mtime=1000); put("okA", mtime=1001); put("okB", mtime=1002)
s = FakeSender({"bad": "server_error", "okA": "ok", "okB": "ok"})
attempts = {}
sent, _ = s._drain_once(attempts)
check("거절 항목이 앞에 있어도 뒤 항목이 전송된다",
      "okA" in s.posted and "okB" in s.posted, f"{s.posted}")
check("거절 항목은 큐에 남는다", "bad" in names(), f"{names()}")
for _ in range(es.MAX_ATTEMPTS_PER_ITEM):
    s._drain_once(attempts)
check("반복 거절되면 격리된다", "bad" not in names(), f"{names()}")
check("격리 폴더에 보관된다", (es.BAD_DIR / "bad.json").exists())
shutil.rmtree(tmp)

# --------------------------------------------------------------- 4
print("\n[서버에 닿지 못할 때]")
tmp = tempfile.mkdtemp(); setup_queue(tmp)
put("keep1"); put("keep2")
s = FakeSender("unreachable")
sent, reachable = s._drain_once({})
check("닿지 못하면 reachable=False", reachable is False)
check("항목을 버리지 않는다", names() == ["keep1", "keep2"], f"{names()}")
check("첫 실패에서 멈춘다(전부 시도하지 않음)", len(s.posted) == 1, f"{s.posted}")
shutil.rmtree(tmp)

print(f"\n{'='*46}\n통과 {PASS} · 실패 {FAIL}\n{'='*46}")
sys.exit(1 if FAIL else 0)

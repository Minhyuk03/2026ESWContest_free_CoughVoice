# -*- coding: utf-8 -*-
"""실시간 기침/대화 판별 대시보드 서버 (FastAPI + WebSocket).

실행 예:
    python server.py --source sim                  # 마이크 없이 데모
    python server.py --source sd                   # Mac 내장/USB 마이크
    python server.py --source arecord --device hw:2 --gain 5   # RPi I2S
    python server.py --list-devices                # 입력 장치 목록

브라우저에서 http://localhost:8000 접속.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import audio_source
from classifier import Config, StreamAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")


@asynccontextmanager
async def lifespan(_app):
    assert HUB is not None
    HUB.start(asyncio.get_running_loop())
    task = asyncio.ensure_future(HUB.broadcaster())
    try:
        yield
    finally:
        task.cancel()
        HUB.stop()


app = FastAPI(title="Cough-ID Live Dashboard", lifespan=lifespan)


# --------------------------------------------------------------------------
class Hub(object):
    """오디오 캡처 -> 분석 -> WebSocket 브로드캐스트."""

    def __init__(self, args):
        self.args = args
        self.cfg = Config()
        self.cfg.cough_threshold = args.threshold
        self.analyzer = StreamAnalyzer(self.cfg)
        self.source = audio_source.create_source(
            args.source,
            device=args.device,
            gain=args.gain,
            channel=args.channel,
            blocksize=args.blocksize,
        )
        self.clients = set()  # type: Set[WebSocket]
        self.loop = None  # type: Optional[asyncio.AbstractEventLoop]
        self.out_q = None  # type: Optional[asyncio.Queue]
        self.counts = {"cough": 0, "speech": 0, "other": 0}
        self.recent = []  # type: List[Dict[str, Any]]
        self._worker = None  # type: Optional[threading.Thread]
        self._stop = threading.Event()
        self.status = "starting"

    # ---------------- 캡처 스레드 ----------------
    def _capture_loop(self):
        frame_batch = []  # type: List[Dict[str, Any]]
        last_send = 0.0
        while not self._stop.is_set():
            block = self.source.read(timeout=0.5)
            if block is None:
                if self.source.error:
                    self.status = "error"
                    self._push({"type": "status", "status": "error", "message": self.source.error})
                    return
                continue
            self.status = "running"
            frames, events = self.analyzer.push(block)
            frame_batch.extend(frames)

            for ev in events:
                self.counts[ev["label"]] = self.counts.get(ev["label"], 0) + 1
                ev["t"] = time.time()
                self.recent.insert(0, ev)
                del self.recent[200:]
                self._push({"type": "event", "event": ev, "counts": dict(self.counts)})

            now = time.time()
            if frame_batch and (now - last_send) >= 0.05:
                last_send = now
                payload = {
                    "type": "frames",
                    "db": [f["db"] for f in frame_batch],
                    "active": frame_batch[-1]["active"],
                    "noise_db": frame_batch[-1]["noise_db"],
                    "on_th": frame_batch[-1]["on_th"],
                    "centroid": frame_batch[-1]["centroid"],
                    "flatness": frame_batch[-1]["flatness"],
                    "zcr": frame_batch[-1]["zcr"],
                    "bands": frame_batch[-1]["bands"],
                }
                frame_batch = []
                self._push(payload)

    def _push(self, msg: Dict[str, Any]):
        if self.loop is None or self.out_q is None:
            return
        try:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, msg)
        except RuntimeError:
            pass

    # ---------------- 브로드캐스트 태스크 ----------------
    async def broadcaster(self):
        assert self.out_q is not None
        while True:
            msg = await self.out_q.get()
            if not self.clients:
                continue
            text = json.dumps(msg, ensure_ascii=False)
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(text)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    # ---------------- 라이프사이클 ----------------
    def start(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.out_q = asyncio.Queue(maxsize=2000)
        self.source.start()
        self._worker = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        self.source.stop()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "type": "hello",
            "source": self.source.name,
            "device": str(self.args.device),
            "sr": self.cfg.sr,
            "status": self.status,
            "counts": dict(self.counts),
            "recent": self.recent[:30],
            "config": {
                "trigger_delta": self.cfg.trigger_delta,
                "cough_threshold": self.cfg.cough_threshold,
                "hang_frames": self.cfg.hang_frames,
                "abs_floor_db": self.cfg.abs_floor_db,
            },
        }


HUB = None  # type: Optional[Hub]


# --------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    assert HUB is not None
    await ws.accept()
    HUB.clients.add(ws)
    await ws.send_text(json.dumps(HUB.snapshot(), ensure_ascii=False))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "config":
                HUB.cfg.update(msg.get("values", {}))
            elif msg.get("type") == "reset":
                HUB.counts = {"cough": 0, "speech": 0, "other": 0}
                HUB.recent = []
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        HUB.clients.discard(ws)


# --------------------------------------------------------------------------
def main():
    global HUB
    p = argparse.ArgumentParser(description="Cough-ID 실시간 대시보드")
    p.add_argument("--source", default="auto", choices=["auto", "sd", "arecord", "sim"])
    p.add_argument("--device", default=None, help="sd: 장치 인덱스 / arecord: hw:2")
    p.add_argument("--channel", default="left", choices=["left", "right"])
    p.add_argument("--gain", type=float, default=None, help="입력 게인 (기본: sd=1, arecord=5)")
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.55, help="기침 판정 점수 컷오프")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        for line in audio_source.SoundDeviceSource.list_devices():
            print(line)
        return

    if args.gain is None:
        args.gain = 5.0 if args.source == "arecord" else 1.0
    if args.device is not None and args.source in ("sd", "auto"):
        try:
            args.device = int(args.device)
        except ValueError:
            pass

    HUB = Hub(args)

    import uvicorn

    print("=" * 58)
    print("  Cough-ID 실시간 대시보드")
    print("  소스: %s   장치: %s   게인: %s" % (HUB.source.name, args.device, args.gain))
    print("  ->  http://localhost:%d" % args.port)
    print("=" * 58)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    main()

"""EventSender — 기침 클립을 서버 POST /events로 전송. 실패 시 디스크 큐 + 지수 백오프 재시도 (TC-05)."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

QUEUE_DIR = Path(__file__).parent / "queue"


class EventSender:
    def __init__(
        self,
        server_url: str,               # 예: http://<서버IP>:8000
        device_id: str = "rpi5-01",
        timeout: float = 5.0,
        max_backoff: float = 60.0,
        outbox_size: int = 32,
    ):
        self.endpoint = server_url.rstrip("/") + "/events"
        self.device_id = device_id
        self.timeout = timeout
        self.max_backoff = max_backoff
        QUEUE_DIR.mkdir(exist_ok=True)
        self._stop = threading.Event()
        # send()는 sounddevice 오디오 콜백 안에서 호출된다. 거기서 HTTP를 기다리면
        # 콜백이 막혀 마이크 버퍼가 넘치고 "input overflow"로 오디오가 유실된다
        # (2026-08-24 실기에서 확인). 그래서 send()는 큐에 넣기만 하고,
        # 실제 전송은 이 워커 스레드가 맡는다.
        self._outbox: "queue.Queue[tuple[bytes, dict]]" = queue.Queue(maxsize=outbox_size)
        self._dropped = 0
        self._worker = threading.Thread(target=self._send_loop, daemon=True)
        self._worker.start()
        self._thread = threading.Thread(target=self._retry_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def send(self, wav_bytes: bytes, peak_rms: float) -> None:
        """오디오 콜백에서 호출된다 — 절대 블로킹하지 않는다."""
        meta = {
            "device_id": self.device_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "peak_rms": round(peak_rms, 4),
        }
        try:
            self._outbox.put_nowait((wav_bytes, meta))
        except queue.Full:
            # 큐가 찼다는 건 검출이 전송보다 빠르다는 뜻이다. 여기서 기다리면
            # 오디오가 끊기므로 버린다. 대신 몇 개를 버렸는지 알린다.
            self._dropped += 1
            if self._dropped % 20 == 1:
                print(f"[sender] 전송 큐 포화 — 누적 {self._dropped}건 폐기 "
                      f"(검출 임계치를 올리거나 서버 처리를 늘려야 함)", flush=True)

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            try:
                wav_bytes, meta = self._outbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self._post(wav_bytes, meta):
                self._enqueue(wav_bytes, meta)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _post(self, wav_bytes: bytes, meta: dict) -> bool:
        try:
            r = requests.post(
                self.endpoint,
                files={"audio": ("cough.wav", wav_bytes, "audio/wav")},
                data={"meta": json.dumps(meta)},
                timeout=self.timeout,
            )
            ok = r.status_code < 300
            print(f"[sender] POST {r.status_code} {r.text[:120]}", flush=True)
            return ok
        except requests.RequestException as e:
            print(f"[sender] 전송 실패: {e}", flush=True)
            return False

    def _enqueue(self, wav_bytes: bytes, meta: dict) -> None:
        stem = QUEUE_DIR / uuid.uuid4().hex
        stem.with_suffix(".wav").write_bytes(wav_bytes)
        stem.with_suffix(".json").write_text(json.dumps(meta))
        print(f"[sender] 큐 적재: {stem.name}", flush=True)

    def _retry_loop(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            pending = sorted(QUEUE_DIR.glob("*.json"))
            if not pending:
                backoff = 2.0
                time.sleep(2)
                continue
            sent_any = False
            for meta_path in pending:
                wav_path = meta_path.with_suffix(".wav")
                if not wav_path.exists():
                    meta_path.unlink(missing_ok=True)
                    continue
                meta = json.loads(meta_path.read_text())
                if self._post(wav_path.read_bytes(), meta):
                    wav_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    sent_any = True
                else:
                    break  # 서버 아직 다운 → 백오프
            if sent_any:
                backoff = 2.0
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

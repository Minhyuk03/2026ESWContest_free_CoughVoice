"""EventSender — 기침 클립을 서버 POST /events로 전송. 실패 시 디스크 큐 + 지수 백오프 재시도 (TC-05).

생존 신호(POST /heartbeat)도 여기서 보낸다. 엣지는 기침이 있을 때만 말을 하므로,
조용한 밤과 장치가 죽은 상태가 서버에서 구분되지 않는 문제가 있었다. 주기적인 비트가
있어야 24시간 연속 동작을 검증할 수 있고, 기준선 계산이 가동 중단 구간을 '기침 0회'로
세는 것도 막을 수 있다.
"""
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
BAD_DIR = QUEUE_DIR / "bad"          # 반복 실패한 항목 격리

# 큐 상한. 파이 SD 여유가 1GB 안팎이고 이벤트 하나가 80KB이므로 무한정 쌓게 두면
# 디스크를 채운다 (2026-08-25 실제로 4,314파일 177MB까지 자랐고 여유가 977MB였다).
# 넘치면 **오래된 것부터** 버린다 — 최근 이벤트가 진단에 더 쓸모 있다.
QUEUE_MAX_EVENTS = 500
# 같은 항목을 이만큼 시도해도 서버가 거부하면 격리한다. 서버가 4xx/5xx로 거절하는
# 항목을 무한 재시도하면 그 뒤의 정상 항목까지 막힌다.
MAX_ATTEMPTS_PER_ITEM = 3


class EventSender:
    def __init__(
        self,
        server_url: str,               # 예: http://<서버IP>:8000
        device_id: str = "rpi5-01",
        timeout: float = 5.0,
        max_backoff: float = 60.0,
        outbox_size: int = 32,
        heartbeat_interval: float = 60.0,   # 0 이하면 생존 신호를 보내지 않는다
    ):
        base = server_url.rstrip("/")
        self.endpoint = base + "/events"
        self.heartbeat_endpoint = base + "/heartbeat"
        self.heartbeat_interval = heartbeat_interval
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
        if heartbeat_interval > 0:
            self._hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb.start()

    # ------------------------------------------------------------------
    def send(self, wav_bytes: bytes, peak_rms: float) -> None:
        """오디오 콜백에서 호출된다 — 절대 블로킹하지 않는다."""
        meta = {
            # 재전송 큐가 같은 클립을 다시 보낼 수 있으므로 이벤트마다 고유 ID를 붙인다.
            # 서버는 같은 ID를 다시 받으면 새 이벤트를 만들지 않는다(멱등성).
            "event_id": uuid.uuid4().hex,
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
            if self._post(wav_bytes, meta) != "ok":
                self._enqueue(wav_bytes, meta)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        """주기적으로 생존을 알린다.

        기침 전송과 스레드를 나눈 이유: 전송 워커는 서버가 죽어 있으면 백오프로 길게
        잠들 수 있는데, 그동안 생존 신호까지 멈추면 서버가 복구된 뒤에도 한참 오프라인으로
        보인다. 실패해도 재시도 큐에 쌓지 않는다 — 지나간 시점의 생존은 나중에 알려봐야
        의미가 없고, 큐만 불린다.
        """
        # 기동 직후 한 번 보내 서버가 즉시 온라인으로 인식하게 한다.
        first = True
        while not self._stop.is_set():
            if not first:
                if self._stop.wait(self.heartbeat_interval):
                    break
            first = False
            try:
                requests.post(self.heartbeat_endpoint,
                              json={"device_id": self.device_id},
                              timeout=self.timeout)
            except Exception:
                # 서버가 잠깐 없는 것은 정상 — 다음 주기에 다시 보낸다.
                # RequestException뿐 아니라 어떤 예외로도 이 스레드가 죽으면 안 된다.
                pass

    # ------------------------------------------------------------------
    def _post(self, wav_bytes: bytes, meta: dict) -> str:
        """전송 결과를 세 갈래로 구분한다: ok / server_error / unreachable.

        "서버에 닿지 못했다"와 "서버가 이 항목을 거절했다"는 대응이 달라야 한다.
        전자는 기다리면 풀리지만, 후자는 기다려도 그 항목은 영원히 안 된다.
        예전에는 둘 다 False였고 실패 시 루프를 break해서, 거절당하는 항목 하나가
        뒤의 정상 항목을 전부 막았다.
        """
        try:
            r = requests.post(
                self.endpoint,
                files={"audio": ("cough.wav", wav_bytes, "audio/wav")},
                data={"meta": json.dumps(meta)},
                timeout=self.timeout,
            )
            print(f"[sender] POST {r.status_code} {r.text[:120]}", flush=True)
            return "ok" if r.status_code < 300 else "server_error"
        except requests.RequestException as e:
            print(f"[sender] 전송 실패: {e}", flush=True)
            return "unreachable"

    def _enqueue(self, wav_bytes: bytes, meta: dict) -> None:
        stem = QUEUE_DIR / uuid.uuid4().hex
        stem.with_suffix(".wav").write_bytes(wav_bytes)
        stem.with_suffix(".json").write_text(json.dumps(meta))
        print(f"[sender] 큐 적재: {stem.name}", flush=True)
        self._trim_queue()

    def _trim_queue(self) -> None:
        """큐가 상한을 넘으면 오래된 것부터 버린다.

        상한이 없으면 서버가 오래 안 붙는 동안 SD 카드를 채운다. 실제로 그렇게 됐다.
        """
        items = sorted(QUEUE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        excess = len(items) - QUEUE_MAX_EVENTS
        if excess <= 0:
            return
        for meta_path in items[:excess]:
            meta_path.with_suffix(".wav").unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        print(f"[sender] 큐 상한({QUEUE_MAX_EVENTS}) 초과 — 오래된 {excess}건 폐기", flush=True)

    def _discard(self, meta_path: Path, reason: str, keep: bool = False) -> None:
        wav_path = meta_path.with_suffix(".wav")
        if keep:
            BAD_DIR.mkdir(exist_ok=True)
            for q in (meta_path, wav_path):
                if q.exists():
                    q.replace(BAD_DIR / q.name)
        else:
            wav_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        print(f"[sender] 큐 항목 제외({reason}): {meta_path.stem}", flush=True)

    def _retry_loop(self) -> None:
        """디스크 큐를 비운다.

        **이 스레드는 절대 죽으면 안 된다.** 예전에는 손상된 큐 파일 하나에
        `json.loads`가 예외를 던지면 데몬 스레드가 조용히 끝나버렸고, 그 뒤로 재시도가
        영원히 멈췄다. 2026-08-25에 0바이트 파일 12개 때문에 실제로 그렇게 됐고,
        이벤트가 4.5시간 늦게 도착하고 큐가 177MB까지 자랐다.
        그래서 항목 단위로도, 루프 전체로도 예외를 잡는다.
        """
        backoff = 2.0
        attempts: dict = {}
        self._trim_queue()          # 기동 시 이전 실행이 남긴 초과분을 정리한다
        while not self._stop.is_set():
            try:
                sent_any, reachable = self._drain_once(attempts)
            except Exception as e:  # 예상 못 한 오류에도 스레드를 유지한다
                print(f"[sender] 재시도 루프 오류(계속 진행): {e!r}", flush=True)
                sent_any, reachable = False, True
            if sent_any:
                backoff = 2.0
            elif not reachable:
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
            else:
                time.sleep(2)

    def _drain_once(self, attempts: dict) -> tuple:
        """큐를 한 바퀴 돈다. (하나라도 보냈나, 서버에 닿았나)를 돌려준다."""
        pending = sorted(QUEUE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not pending:
            return False, True
        sent_any = False
        for meta_path in pending:
            if self._stop.is_set():
                break
            wav_path = meta_path.with_suffix(".wav")
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                self._discard(meta_path, "오디오 없음")
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError, ValueError):
                # 전송 도중 프로세스가 죽으면 0바이트 파일이 남는다. 되살릴 수 없으므로
                # 버린다. 예전에는 여기서 예외가 스레드를 통째로 죽였다.
                self._discard(meta_path, "메타 손상")
                continue

            result = self._post(wav_path.read_bytes(), meta)
            if result == "ok":
                self._discard(meta_path, "전송 완료")
                attempts.pop(meta_path.name, None)
                sent_any = True
            elif result == "unreachable":
                return sent_any, False        # 서버가 없다 — 나머지는 다음 기회에
            else:
                n = attempts.get(meta_path.name, 0) + 1
                attempts[meta_path.name] = n
                if n >= MAX_ATTEMPTS_PER_ITEM:
                    self._discard(meta_path, f"{n}회 거절", keep=True)
                    attempts.pop(meta_path.name, None)
                # 서버는 살아 있으므로 다음 항목을 계속 시도한다
        return sent_any, True

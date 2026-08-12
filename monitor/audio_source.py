# -*- coding: utf-8 -*-
"""오디오 입력 소스 (Python 3.9 호환).

- SoundDeviceSource : Mac / USB 마이크 (sounddevice)
- ArecordSource     : RPi I2S MEMS 마이크 (arecord, S32_LE 48kHz 2ch -> 16kHz mono)
- SimSource         : 마이크 없이 합성 신호로 테스트

모든 소스는 float32 mono 16kHz 블록을 큐로 내보낸다.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from typing import List, Optional

import numpy as np

TARGET_SR = 16000


# --------------------------------------------------------------------------
# 공통 베이스
# --------------------------------------------------------------------------
class BaseSource(object):
    name = "base"

    def __init__(self, blocksize: int = 1024):
        self.blocksize = blocksize
        self.q = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._thread = None  # type: Optional[threading.Thread]
        self.error = None  # type: Optional[str]

    # ---- 하위 클래스 구현 ----
    def _run(self):
        raise NotImplementedError

    # ---- 공통 ----
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._safe_run, daemon=True)
        self._thread.start()

    def _safe_run(self):
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.error = "%s: %s" % (type(exc).__name__, exc)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _emit(self, block: np.ndarray):
        if self.q.full():
            try:
                self.q.get_nowait()  # 오래된 블록 버림 (실시간 우선)
            except queue.Empty:
                pass
        self.q.put(block)


# --------------------------------------------------------------------------
# 48kHz -> 16kHz 데시메이터 (numpy만 사용, 상태 유지)
# --------------------------------------------------------------------------
class Decimator3(object):
    """3:1 데시메이션 + 안티에일리어싱 FIR (cutoff ~7kHz @48kHz)."""

    def __init__(self, taps: int = 49):
        n = np.arange(taps) - (taps - 1) / 2.0
        fc = 7000.0 / 48000.0  # 정규화 cutoff
        h = np.sinc(2 * fc * n) * np.hamming(taps)
        h /= np.sum(h)
        self.h = h.astype(np.float32)
        self.tail = np.zeros(taps - 1, dtype=np.float32)
        self.offset = 0  # 다음 블록에서 첫 샘플을 뽑을 위치

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.tail, x.astype(np.float32)])
        y = np.convolve(buf, self.h, mode="valid")
        self.tail = buf[len(buf) - (len(self.h) - 1):]
        if len(y) == 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(self.offset, len(y), 3)
        if len(idx):
            self.offset = int(idx[-1]) + 3 - len(y)
        else:
            self.offset -= len(y)
        return y[idx].astype(np.float32)


# --------------------------------------------------------------------------
# sounddevice (Mac / USB)
# --------------------------------------------------------------------------
class SoundDeviceSource(BaseSource):
    name = "sounddevice"

    def __init__(self, device=None, blocksize: int = 1024, gain: float = 1.0):
        BaseSource.__init__(self, blocksize)
        self.device = device
        self.gain = gain

    def _run(self):
        import sounddevice as sd  # 지연 임포트

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            mono = indata[:, 0].astype(np.float32) * self.gain
            self._emit(mono.copy())

        with sd.InputStream(
            samplerate=TARGET_SR,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        ):
            while not self._stop.is_set():
                time.sleep(0.05)

    @staticmethod
    def list_devices() -> List[str]:
        try:
            import sounddevice as sd

            out = []
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    out.append("[%d] %s" % (i, d["name"]))
            return out
        except Exception as exc:  # noqa: BLE001
            return ["(sounddevice 사용 불가: %s)" % exc]


# --------------------------------------------------------------------------
# arecord (RPi I2S MEMS)
# --------------------------------------------------------------------------
class ArecordSource(BaseSource):
    """RPi5 + MS3625 I2S 마이크 전용.

    캡처 포맷: 2ch S32_LE 48kHz. 24bit left-justified 이므로 >> 8 시프트.
    L/R 핀을 GND에 물린 구성에서는 LEFT 채널만 유효하다.
    """

    name = "arecord"

    def __init__(
        self,
        device: str = "hw:2",
        channel: str = "left",
        gain: float = 5.0,
        blocksize: int = 1024,
        rate: int = 48000,
        channels: int = 2,
    ):
        BaseSource.__init__(self, blocksize)
        self.device = device
        self.channel = channel
        self.gain = gain
        self.rate = rate
        self.channels = channels
        self.proc = None  # type: Optional[subprocess.Popen]

    def _run(self):
        if shutil.which("arecord") is None:
            raise RuntimeError("arecord 를 찾을 수 없습니다 (alsa-utils 설치 필요)")

        cmd = [
            "arecord",
            "-D", self.device,
            "-c", str(self.channels),
            "-f", "S32_LE",
            "-r", str(self.rate),
            "-t", "raw",
            "-q",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        dec = Decimator3() if self.rate == 48000 else None
        ch_idx = 0 if self.channel == "left" else 1
        bytes_per_frame = 4 * self.channels
        read_frames = 2048
        nbytes = read_frames * bytes_per_frame

        assert self.proc.stdout is not None
        while not self._stop.is_set():
            raw = self.proc.stdout.read(nbytes)
            if not raw or len(raw) < bytes_per_frame:
                break
            usable = (len(raw) // bytes_per_frame) * bytes_per_frame
            data = np.frombuffer(raw[:usable], dtype="<i4").reshape(-1, self.channels)
            mono = (data[:, ch_idx] >> 8).astype(np.float32) / float(1 << 23)
            mono *= self.gain
            if dec is not None:
                mono = dec.process(mono)
            if len(mono):
                self._emit(mono)

        if self.proc.poll() is not None and self.proc.returncode not in (0, None):
            err = b""
            if self.proc.stderr is not None:
                err = self.proc.stderr.read()
            raise RuntimeError("arecord 종료 (code=%s) %s" % (self.proc.returncode, err.decode("utf-8", "ignore")[:300]))

    def stop(self):
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        BaseSource.stop(self)


# --------------------------------------------------------------------------
# 시뮬레이션 (마이크 없이 데모/검증)
# --------------------------------------------------------------------------
def synth_cough(sr: int = TARGET_SR, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """폭발적 onset + 빠른 감쇠 + 광대역 잡음 = 기침 유사 신호."""
    rng = rng or np.random.RandomState()
    dur = rng.uniform(0.28, 0.5)
    n = int(dur * sr)
    t = np.arange(n) / float(sr)
    noise = rng.randn(n).astype(np.float32)
    # 고역 강조 (차분)
    noise = np.concatenate([[0.0], np.diff(noise)]).astype(np.float32)
    attack = np.clip(t / 0.008, 0, 1)
    decay = np.exp(-t / rng.uniform(0.05, 0.09))
    env = attack * decay
    # 두 번째 작은 버스트 (기침 특유의 voiced tail)
    tail = np.zeros(n, dtype=np.float32)
    k = int(0.12 * sr)
    if k < n:
        tt = t[: n - k]
        tail[k:] = 0.25 * np.exp(-tt / 0.05) * np.sin(2 * np.pi * 190 * tt)
    return ((noise * env + tail) * 0.55).astype(np.float32)


def synth_speech(sr: int = TARGET_SR, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """유성음 하모닉 + 4Hz 음절 변조 = 대화 유사 신호."""
    rng = rng or np.random.RandomState()
    dur = rng.uniform(1.2, 2.2)
    n = int(dur * sr)
    t = np.arange(n) / float(sr)
    f0 = rng.uniform(105, 190)
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5 * t)
    sig = np.zeros(n, dtype=np.float32)
    for h, amp in [(1, 1.0), (2, 0.6), (3, 0.4), (4, 0.25), (5, 0.15), (6, 0.08)]:
        sig += amp * np.sin(2 * np.pi * f0 * h * vib * t)
    sig /= 2.5
    syl = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * rng.uniform(3.0, 5.0) * t))
    fade = np.clip(np.minimum(t / 0.08, (dur - t) / 0.12), 0, 1)
    sig = sig * syl * fade
    sig += 0.03 * rng.randn(n).astype(np.float32) * syl  # 약간의 마찰음
    return (sig * 0.4).astype(np.float32)


class SimSource(BaseSource):
    name = "sim"

    def __init__(self, blocksize: int = 1024, seed: int = 0, noise: float = 0.002):
        BaseSource.__init__(self, blocksize)
        self.rng = np.random.RandomState(seed)
        self.noise = noise

    def _run(self):
        pending = np.zeros(0, dtype=np.float32)
        next_event_at = time.time() + 1.5
        while not self._stop.is_set():
            now = time.time()
            if len(pending) < self.blocksize and now >= next_event_at:
                if self.rng.rand() < 0.5:
                    ev = synth_cough(rng=self.rng)
                else:
                    ev = synth_speech(rng=self.rng)
                pending = np.concatenate([pending, ev])
                next_event_at = now + float(self.rng.uniform(2.0, 3.5))

            if len(pending) >= self.blocksize:
                block = pending[: self.blocksize]
                pending = pending[self.blocksize:]
            else:
                block = np.zeros(self.blocksize, dtype=np.float32)
                if len(pending):
                    block[: len(pending)] = pending
                    pending = np.zeros(0, dtype=np.float32)

            block = block + self.noise * self.rng.randn(self.blocksize).astype(np.float32)
            self._emit(block.astype(np.float32))
            time.sleep(self.blocksize / float(TARGET_SR))


# --------------------------------------------------------------------------
def create_source(kind: str = "auto", **kwargs) -> BaseSource:
    if kind == "auto":
        try:
            import sounddevice  # noqa: F401
            kind = "sd"
        except Exception:  # noqa: BLE001
            kind = "arecord" if shutil.which("arecord") else "sim"

    if kind in ("sd", "sounddevice"):
        return SoundDeviceSource(
            device=kwargs.get("device"),
            blocksize=kwargs.get("blocksize", 1024),
            gain=kwargs.get("gain", 1.0),
        )
    if kind == "arecord":
        return ArecordSource(
            device=kwargs.get("device") or "hw:2",
            channel=kwargs.get("channel", "left"),
            gain=kwargs.get("gain", 5.0),
            blocksize=kwargs.get("blocksize", 1024),
        )
    if kind == "sim":
        return SimSource(blocksize=kwargs.get("blocksize", 1024))
    raise ValueError("알 수 없는 소스 종류: %s" % kind)

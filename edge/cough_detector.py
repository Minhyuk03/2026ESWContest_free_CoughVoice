"""CoughDetector — 1차 에너지 임계치 검출 (+ 2차 CNN은 후속 단계).

동작:
  - AudioCapture가 넘겨주는 100ms 청크의 RMS를 감시
  - 임계치 초과가 시작되면 "이벤트 후보" 시작, 아래로 내려가면 종료
  - 이벤트 길이가 기침 범위(0.1~1.5초)면 전후 여유 포함 2.5초 구간을 링버퍼에서 절단
  - on_cough(wav_bytes, peak_rms) 콜백 호출 → EventSender가 서버 전송

쿨다운을 2초로 둔 이유: 기침 직후의 들숨과 잔향이 임계치를 다시 넘어 **한 번의 기침이
두 건으로 기록**된다. 3초 클립 70개로 측정한 결과 쿨다운 1초에서 41%(29/70)가 중복
트리거였고 클립당 평균 1.47회였다. 알림 규칙이 이벤트 개수를 세므로 "1시간 10회"가
실제로는 7번쯤에 발동했다. 2초로 올리면 14%(10/70)로 줄어든다.

대가: 2초보다 빠른 연속 기침은 하나로 셈된다. 기침 발작처럼 몰아서 하는 경우
과소 계수되므로, 빈도를 절대값으로 해석하지 말 것.

2026-08-28 — **RMS를 전대역에서 재던 것이 오탐의 주된 통로였다.** 8/28 05:48·08:02
두 이벤트를 들어보니 기침이 아니었는데, 트리거 에너지의 거의 전부가 80Hz 아래에
있었다(80Hz 하이패스 후 100ms 최대 RMS 0.116→0.005, 0.102→0.004로 22~26배 감소).
I2S MEMS 마이크는 저역 드리프트·구조진동·공조 소음을 크게 받는데, 그 대역은 기침
판별에 아무 정보가 없으면서 RMS만 밀어 올린다. 실제 기침은 같은 필터에 거의 영향을
받지 않는다(0.173→0.156, 0.557→0.513).

그래서 **판정용 RMS만 80Hz 하이패스를 거친 신호에서 잰다.** 서버로 보내는 클립은
원본 그대로다(게이트·화자 식별은 영향 없음). 필터로 저역이 빠지면 같은 임계치가
상대적으로 높아지므로 임계치도 0.10 → 0.05로 함께 내린다. 라벨 기침 151개·TV 생활소음
20분 기준 실측:
    현행 (필터 없음, 0.10)  기침 트리거 100%   생활소음 774회/h
    HP80Hz + 0.10            72%              3회/h
    **HP80Hz + 0.05          95.4%            27회/h**   ← 채택
    HP80Hz + 0.03            97.4%            48회/h
문제의 두 이벤트는 HP 후 0.005/0.004라 어느 지점에서도 트리거되지 않는다.

2차 CNN(YAMNet/tflite) 판정은 classify() 자리에 끼워 넣도록 구조를 잡아둠(P4 후반).
"""
from __future__ import annotations

import io
import time
import wave

import numpy as np

from audio_capture import SAMPLE_RATE, AudioCapture


class HighPass:
    """2차 버터워스 하이패스 (biquad, 상태 유지).

    청크 단위로 들어오는 스트림에 걸어야 하므로 필터 상태를 호출 간에 이어간다.
    상태를 매 청크 초기화하면 경계마다 과도응답이 생겨 그것이 다시 오탐이 된다.
    scipy.signal.butter(2, fc/(fs/2), "high")와 같은 계수를 직접 계산한다 —
    파이 엣지에는 scipy가 없고(edge/requirements.txt) 이 하나 때문에 넣을 이유도 없다.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int = SAMPLE_RATE):
        w0 = 2.0 * np.pi * cutoff_hz / sample_rate
        cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
        alpha = sin_w0 / (2.0 * (2 ** -0.5))        # Q = 1/sqrt(2) → 버터워스
        b = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2])
        a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
        self.b = b / a[0]
        self.a = a / a[0]
        self._z1 = 0.0
        self._z2 = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """전치 직접형 II. 짧은 청크(1600 샘플)라 파이썬 루프로 충분하다."""
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        z1, z2 = self._z1, self._z2
        out = np.empty_like(x, dtype=np.float64)
        for i, xn in enumerate(x):
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            out[i] = yn
        self._z1, self._z2 = z1, z2
        return out.astype(np.float32)


class CoughDetector:
    def __init__(
        self,
        capture: AudioCapture,
        rms_threshold: float = 0.05,   # 하이패스 적용 기준값 (위 docstring 실측 참조)
        min_dur: float = 0.08,         # 기침 최소 길이(초)
        max_dur: float = 1.5,          # 이보다 길면 말소리/소음으로 간주
        clip_seconds: float = 2.5,     # 서버로 보낼 절단 길이
        cooldown: float = 2.0,         # 연속 트리거 방지 (아래 주석 참조)
        hp_hz: float = 80.0,           # 판정용 RMS에만 거는 하이패스. 0이면 끔
    ):
        self.capture = capture
        self.rms_threshold = rms_threshold
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.clip_seconds = clip_seconds
        self.cooldown = cooldown
        self.hp_hz = hp_hz
        # 판정용 RMS 전용 필터. 링버퍼(서버로 보낼 클립)는 원본 그대로 둔다.
        self._hp = HighPass(hp_hz) if hp_hz > 0 else None
        # 필터 상태가 0에서 출발하면 첫 청크에 계단 응답이 실려 RMS가 치솟는다.
        # 실제로 배포 직후 peak_rms=0.123짜리 가짜 트리거가 1건 찍혔다(2026-08-28).
        # 상태가 자리 잡을 때까지(0.5초) 판정을 미룬다. 링버퍼는 그동안에도 채워진다.
        self._warmup_left = int(0.5 * SAMPLE_RATE) if self._hp is not None else 0
        self.on_cough = None  # callable(wav_bytes: bytes, peak_rms: float)

        self._active = False
        self._event_start = 0.0
        self._peak = 0.0
        self._last_fire = 0.0
        capture.on_chunk = self._on_chunk

    # ------------------------------------------------------------------
    def _on_chunk(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        measured = self._hp(chunk) if self._hp is not None else chunk
        if self._warmup_left > 0:
            self._warmup_left -= len(chunk)
            return
        rms = float(np.sqrt(np.mean(measured**2)))
        now = time.monotonic()

        if not self._active:
            if rms >= self.rms_threshold and now - self._last_fire > self.cooldown:
                self._active = True
                self._event_start = now
                self._peak = rms
        else:
            self._peak = max(self._peak, rms)
            if rms < self.rms_threshold:
                self._finish(now)
            elif now - self._event_start > self.max_dur:
                self._active = False  # 너무 길다 → 기침 아님

    def _finish(self, now: float) -> None:
        self._active = False
        dur = now - self._event_start
        if not (self.min_dur <= dur <= self.max_dur):
            return
        # 여유가 링버퍼에 쌓이도록 잠깐 뒤에 절단해도 되지만, 단순화를 위해 즉시 절단
        clip = self.capture.ring.read_last(self.clip_seconds)
        if len(clip) == 0 or not self.classify(clip):
            return
        self._last_fire = now
        if self.on_cough:
            self.on_cough(to_wav_bytes(clip), self._peak)

    # ------------------------------------------------------------------
    def classify(self, clip: np.ndarray) -> bool:
        """2차 판정 자리. 현재는 통과(True). 추후 YAMNet/tflite로 교체."""
        return True


def to_wav_bytes(mono_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 mono → 16bit PCM WAV 바이트."""
    pcm = (np.clip(mono_f32, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()

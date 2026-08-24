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

2차 CNN(YAMNet/tflite) 판정은 classify() 자리에 끼워 넣도록 구조를 잡아둠(P4 후반).
"""
from __future__ import annotations

import io
import time
import wave

import numpy as np

from audio_capture import SAMPLE_RATE, AudioCapture


class CoughDetector:
    def __init__(
        self,
        capture: AudioCapture,
        rms_threshold: float = 0.08,   # 환경 소음에 맞게 캘리브레이션 필요
        min_dur: float = 0.08,         # 기침 최소 길이(초)
        max_dur: float = 1.5,          # 이보다 길면 말소리/소음으로 간주
        clip_seconds: float = 2.5,     # 서버로 보낼 절단 길이
        cooldown: float = 2.0,         # 연속 트리거 방지 (아래 주석 참조)
    ):
        self.capture = capture
        self.rms_threshold = rms_threshold
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.clip_seconds = clip_seconds
        self.cooldown = cooldown
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
        rms = float(np.sqrt(np.mean(chunk**2)))
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

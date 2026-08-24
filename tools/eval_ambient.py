#!/usr/bin/env python3
"""장시간 생활소음 녹음에서 시간당 오탐 횟수를 측정한다.

3초짜리 통제 샘플로 잰 "오탐 0%"는 연속 운용을 대변하지 못한다. 상용 제품
(Hyfe: 시간당 1.03회)과 비교 가능한 형식의 수치를 내려면 실제 환경을 길게
녹음해 세어야 한다.

실제 파이프라인을 그대로 재현한다:
    파이 CoughDetector(에너지) 트리거 → 서버 PANNs 게이트 판정
두 단계를 각각 세므로, 파이가 서버로 보내는 트래픽 양과 최종 오탐을 함께 알 수 있다.

**녹음 중 실제 기침이 있었다면 오탐이 아니다.** --exclude 로 해당 구간(초)을 빼라.

사용 예:
    python3 eval_ambient.py ~/Downloads/cough_data/tv_ambient_30min.wav
    python3 eval_ambient.py rec.wav --exclude 412 977
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

EDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "edge")
sys.path.insert(0, EDGE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import cough_detector                                    # noqa: E402
from audio_capture import SAMPLE_RATE, AudioCapture      # noqa: E402
from cough_detector import CoughDetector                 # noqa: E402
from app.ml.cough_gate import gate                       # noqa: E402

MIC_BIT_SHIFT = 8      # 24bit가 32bit 슬롯에 left-justified — arecord 원본은 시프트 전이다
MIC_FULL_SCALE = float(2 ** 23)
DECIM = 3              # 48kHz → 16kHz


class VirtualClock:
    def __init__(self): self.t = 1000.0
    def monotonic(self): return self.t


def load_long(path: str, gain: float, chunk_frames: int = 48000 * 10):
    """긴 스테레오 32bit 녹음을 청크로 읽어 16kHz mono float32로 만든다."""
    out = []
    with wave.open(path, "rb") as w:
        n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if width != 4:
            raise SystemExit(f"32bit 녹음이 아닙니다 (sampwidth={width})")
        total = w.getnframes()
        print(f"  {total/rate/60:.1f}분 · {rate}Hz · {n_ch}ch", flush=True)
        while True:
            raw = w.readframes(chunk_frames)
            if not raw:
                break
            a = np.frombuffer(raw, dtype=np.int32)
            if n_ch > 1:
                a = a.reshape(-1, n_ch)[:, 0]
            a = (a >> MIC_BIT_SHIFT).astype(np.float32) / MIC_FULL_SCALE
            out.append(np.clip(a[::DECIM] * gain, -1.0, 1.0))
    return np.concatenate(out), rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--gain", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--exclude", type=float, nargs="*", default=[],
                    help="실제 기침이 난 시각(초). ±3초 구간을 집계에서 제외한다")
    args = ap.parse_args()

    print(f"녹음 로드: {args.wav}")
    x, _ = load_long(args.wav, args.gain)
    dur_h = len(x) / SAMPLE_RATE / 3600
    print(f"  16kHz 변환 후 {len(x)/SAMPLE_RATE/60:.1f}분\n")

    clock = VirtualClock()
    orig = cough_detector.time
    cough_detector.time = clock
    try:
        cap = AudioCapture(source="file", wav_path=args.wav, chunk_ms=100)
        det = CoughDetector(cap, rms_threshold=args.threshold)
        events = []
        det.on_cough = lambda wav_bytes, peak: events.append((clock.t - 1000.0, wav_bytes))

        n = cap.chunk_samples
        step = n / SAMPLE_RATE
        for i in range(0, len(x), n):
            cap._emit(x[i:i + n])
            clock.t += step
    finally:
        cough_detector.time = orig

    def excluded(t):
        return any(abs(t - e) <= 3.0 for e in args.exclude)

    kept = [(t, b) for t, b in events if not excluded(t)]
    print(f"■ 1단계 — 파이 에너지 검출기 (임계치 {args.threshold})")
    print(f"  트리거 {len(events)}회"
          + (f" (실제 기침 제외 후 {len(kept)}회)" if args.exclude else ""))
    print(f"  → 시간당 {len(kept)/dur_h:.1f}회 서버로 전송됨\n")

    tmp = os.path.expanduser("~/.cache/coughid/_ambient_tmp.wav")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    scored = []
    for t, b in kept:
        with open(tmp, "wb") as f:
            f.write(b)
        scored.append((t, gate.score(tmp)))

    n_pass = sum(1 for _, s in scored if s >= gate.threshold)
    print(f"■ 2단계 — 서버 PANNs 게이트 (현재 임계치 {gate.threshold})")
    print(f"  통과 {n_pass}회 / {len(scored)}회  →  **시간당 오탐 {n_pass/dur_h:.1f}회**")
    print(f"  참고: Hyfe(상용) 시간당 1.03회\n")

    # 임계치를 올리면 오탐은 줄지만 기침 검출률도 떨어진다. 우리 기침 60개 기준
    # 검출률(eval_gate.py 측정치)과 나란히 놓고 운용점을 고른다.
    known_recall = {0.005: 98, 0.01: 98, 0.02: 97, 0.05: 97,
                    0.1: 95, 0.2: 92, 0.3: 90, 0.5: None, 1.0: None}
    print(f"  {'임계치':>8}{'시간당 오탐':>12}{'기침 검출률':>13}")
    for th in sorted(known_recall):
        fp = sum(1 for _, s in scored if s >= th) / dur_h
        rec = known_recall[th]
        rec_s = f"{rec}%" if rec is not None else "미측정"
        print(f"  {th:>8.3f}{fp:>11.1f}회{rec_s:>13}")

    top = sorted(scored, key=lambda x: -x[1])[:10]
    print("\n점수가 높았던 구간 (직접 들어볼 것 — TV 속 기침일 수 있다):")
    for t, s in top:
        print(f"  {int(t)//60:02d}:{int(t)%60:02d}  점수 {s:.3f}")


if __name__ == "__main__":
    main()

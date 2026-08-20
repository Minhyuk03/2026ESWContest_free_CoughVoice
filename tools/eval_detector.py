#!/usr/bin/env python3
"""P4 검출기 오프라인 평가 — 수집한 WAV를 실시간처럼 흘려 CoughDetector 트리거를 센다.

식별(P3) 앞단에서 "기침이 아닌 소리"를 얼마나 걸러내는지 측정한다. 검출기가 통과시킨
소리는 무조건 화자 매칭을 거치므로, 여기서 새는 오탐은 그대로 오식별이 된다.

edge/cough_detector.py의 **실제 코드 경로**를 그대로 쓰되 시계만 가상으로 돌린다.
로직을 재구현하면 실제 동작과 어긋나므로 그렇게 하지 않는다.

사용 예:
    python3 eval_detector.py --data ~/Downloads/cough_data
    python3 eval_detector.py --data ~/Downloads/cough_data --sweep
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import wave
from collections import defaultdict

import numpy as np

EDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "edge")
sys.path.insert(0, EDGE)

import audio_capture              # noqa: E402
import cough_detector             # noqa: E402
from audio_capture import SAMPLE_RATE, AudioCapture  # noqa: E402
from cough_detector import CoughDetector             # noqa: E402

# collect_cough.py는 24bit 정렬(>>8)로 저장하므로 유효 풀스케일은 2^23이다.
# audio_capture._run_file은 2^31로 나누고 있어 이 파일들을 256배 작게 읽는다(별도 이슈).
COLLECT_FULL_SCALE = 2 ** 23
MIC_DECIM = 3        # 48kHz → 16kHz


class VirtualClock:
    """청크 길이만큼만 흐르는 시계 — 파일을 실시간 대기 없이 재생하기 위함."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def load_as_edge_input(path: str, gain: float) -> np.ndarray:
    """수집 WAV를 엣지 마이크 경로와 동일한 스케일·샘플레이트의 float32로 변환."""
    with wave.open(path, "rb") as w:
        rate, width, nch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int32 if width == 4 else np.int16).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch)[:, 0]
    x = x / (COLLECT_FULL_SCALE if width == 4 else 2 ** 15)
    if rate != SAMPLE_RATE:
        x = x[::rate // SAMPLE_RATE] if rate % SAMPLE_RATE == 0 else x[::MIC_DECIM]
    x = np.clip(x * gain, -1.0, 1.0).astype(np.float32)
    return x


def run_one(path: str, threshold: float, gain: float, chunk_ms: int = 100) -> tuple[bool, float]:
    """WAV 하나를 흘려보내고 (트리거 여부, 최대 청크 RMS)를 반환."""
    clock = VirtualClock()
    orig_time = cough_detector.time
    cough_detector.time = clock                     # 실제 로직은 그대로, 시계만 교체
    try:
        cap = AudioCapture(source="file", wav_path=path, chunk_ms=chunk_ms)
        det = CoughDetector(cap, rms_threshold=threshold)
        fired = []
        det.on_cough = lambda wav, peak: fired.append(peak)

        x = load_as_edge_input(path, gain)
        n = cap.chunk_samples
        step = n / SAMPLE_RATE
        max_rms = 0.0
        for i in range(0, len(x), n):
            chunk = x[i:i + n]
            max_rms = max(max_rms, float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)
            cap._emit(chunk)
            clock.t += step
        # 마지막 청크가 임계치 위에서 끝났을 때를 위해 무음을 한 번 더 흘린다
        cap._emit(np.zeros(n, dtype=np.float32))
        clock.t += step
        return bool(fired), max_rms
    finally:
        cough_detector.time = orig_time


def load_rows(data_dir: str):
    rows = []
    with open(os.path.join(data_dir, "metadata.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["path"] = os.path.join(data_dir, "wav", r["filename"])
            if os.path.exists(r["path"]):
                rows.append(r)
    return rows


COUGH_TYPES = {"dry", "natural", "throat"}


def evaluate(rows, threshold: float, gain: float):
    hit = defaultdict(int)
    total = defaultdict(int)
    rms_by_group = defaultdict(list)
    for r in rows:
        group = r["type"] if r["type"] in COUGH_TYPES else f"neg/{r['type']}"
        fired, max_rms = run_one(r["path"], threshold, gain)
        total[group] += 1
        hit[group] += int(fired)
        rms_by_group[group].append(max_rms)
    return hit, total, rms_by_group


def report(hit, total, rms_by_group, threshold, gain):
    print(f"\n■ 임계치 {threshold}  게인 {gain}x")
    print(f"  {'구분':<16}{'트리거':>10}{'비율':>9}   최대RMS(평균/최고)")
    cough_h = cough_t = neg_h = neg_t = 0
    for g in sorted(total, key=lambda k: (k.startswith("neg"), k)):
        a = np.array(rms_by_group[g])
        rate = hit[g] / total[g] * 100
        print(f"  {g:<16}{hit[g]:>5}/{total[g]:<4}{rate:>7.0f}%   "
              f"{a.mean():.3f} / {a.max():.3f}")
        if g.startswith("neg"):
            neg_h += hit[g]; neg_t += total[g]
        else:
            cough_h += hit[g]; cough_t += total[g]
    if cough_t and neg_t:
        print(f"  → 기침 검출률 {cough_h/cough_t*100:.0f}% ({cough_h}/{cough_t}) · "
              f"비기침 오탐률 {neg_h/neg_t*100:.0f}% ({neg_h}/{neg_t})")
    return (cough_h, cough_t, neg_h, neg_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/cough_data"))
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--gain", type=float, default=5.0)
    ap.add_argument("--sweep", action="store_true", help="임계치를 훑어 최적점을 찾는다")
    args = ap.parse_args()

    rows = load_rows(args.data)
    print(f"데이터 {len(rows)}개: {args.data}")
    print("기침 = dry/natural/throat · 비기침 = neg/speech, neg/noise")

    if not args.sweep:
        report(*evaluate(rows, args.threshold, args.gain), args.threshold, args.gain)
        return

    print("\n임계치 스윕 — 기침은 최대한 잡고 비기침은 최대한 버리는 지점을 찾는다")
    print(f"  {'임계치':>8}{'기침 검출률':>14}{'비기침 오탐률':>15}")
    best = None
    for t in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
        hit, total, _ = evaluate(rows, t, args.gain)
        ch = sum(v for k, v in hit.items() if k in COUGH_TYPES)
        ct = sum(v for k, v in total.items() if k in COUGH_TYPES)
        nh = sum(v for k, v in hit.items() if k.startswith("neg"))
        nt = sum(v for k, v in total.items() if k.startswith("neg"))
        recall, fpr = ch / ct, nh / nt
        print(f"  {t:>8.2f}{recall*100:>12.0f}%{fpr*100:>14.0f}%")
        score = recall - fpr
        if best is None or score > best[1]:
            best = (t, score, recall, fpr)
    print(f"\n  최적 임계치 {best[0]:.2f} — 검출률 {best[2]*100:.0f}%, 오탐률 {best[3]*100:.0f}%")
    if best[3] > 0.2:
        print("  ⚠ 오탐률이 20%를 넘는다. 에너지+지속시간만으로는 분리가 안 된다는 뜻이고,")
        print("    CoughDetector.classify()에 2차 판정(스펙트럼 특징 등)이 필요하다.")


if __name__ == "__main__":
    main()

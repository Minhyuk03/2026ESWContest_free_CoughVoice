# -*- coding: utf-8 -*-
"""합성 신호로 분류기 정확도 확인. 실행: python test_classifier.py"""

from __future__ import annotations

import numpy as np

from audio_source import TARGET_SR, synth_cough, synth_speech
from classifier import Config, StreamAnalyzer

SR = TARGET_SR


def run_case(sig, noise=0.002, seed=0):
    rng = np.random.RandomState(seed)
    pad = np.zeros(int(1.2 * SR), dtype=np.float32)
    x = np.concatenate([pad, sig, pad]).astype(np.float32)
    x = x + noise * rng.randn(len(x)).astype(np.float32)
    an = StreamAnalyzer(Config())
    evs = []
    for i in range(0, len(x), 1024):
        _, e = an.push(x[i:i + 1024])
        evs.extend(e)
    return evs


def main():
    rng = np.random.RandomState(42)
    rows = []
    ok = 0
    total = 0

    for i in range(12):
        evs = run_case(synth_cough(rng=rng), seed=i)
        ev = max(evs, key=lambda e: e["peak_db"]) if evs else None
        got = ev["label"] if ev else "none"
        total += 1
        ok += (got == "cough")
        rows.append(("cough", got, ev))

    for i in range(12):
        evs = run_case(synth_speech(rng=rng), seed=100 + i)
        ev = max(evs, key=lambda e: e["duration"]) if evs else None
        got = ev["label"] if ev else "none"
        total += 1
        ok += (got == "speech")
        rows.append(("speech", got, ev))

    print("%-8s %-8s %-6s %-7s %-7s %-8s %-8s %s" %
          ("정답", "판정", "점수", "길이s", "어택ms", "센트로이드", "플랫니스", "봉우리"))
    print("-" * 78)
    for exp, got, ev in rows:
        mark = "O" if exp == got else "X"
        if ev is None:
            print("%-8s %-8s  (이벤트 미검출)  %s" % (exp, got, mark))
            continue
        print("%-8s %-8s %-6.2f %-7.2f %-7.0f %-8.0f %-8.3f %-3d %s" % (
            exp, got, ev["score"], ev["duration"], ev["attack_time"] * 1000,
            ev["centroid"], ev["flatness"], ev["n_peaks"], mark))
    print("-" * 78)
    print("정확도: %d/%d = %.1f%%" % (ok, total, 100.0 * ok / total))


if __name__ == "__main__":
    main()

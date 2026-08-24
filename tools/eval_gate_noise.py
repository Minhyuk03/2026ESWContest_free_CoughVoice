#!/usr/bin/env python3
"""소음 환경에서의 기침 게이트 강건성 측정.

조용한 방에서 녹음한 샘플로만 잰 성능은 실제 거실(TV·대화·생활소음)을 대변하지
못한다. 별도 녹음 없이 확인하기 위해, 우리가 가진 기침에 우리가 가진 네거티브를
SNR을 조절해 섞어 검출률이 어디서 무너지는지 본다.

한계: 섞는 소음이 같은 방에서 녹음한 말소리·생활잡음(박수·문닫기·키보드)이다.
TV의 연속적인 음악·효과음은 포함되지 않으므로, 이 결과는 근사치다.

사용 예:
    python3 eval_gate_noise.py --data ~/Downloads/cough_data
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from app.ml.cough_gate import gate  # noqa: E402
from app.ml.features import read_wav  # noqa: E402

COUGH_TYPES = {"dry", "natural", "throat"}
SNRS = [20, 10, 5, 0, -5]


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def mix(cough, noise, snr_db):
    """기침 대비 소음을 목표 SNR로 맞춰 더한다. 길이는 짧은 쪽에 맞춘다."""
    n = min(len(cough), len(noise))
    c, v = cough[:n], noise[:n]
    scale = rms(c) / (rms(v) * (10 ** (snr_db / 20)))
    out = c + v * scale
    peak = float(np.abs(out).max())
    return (out / peak * 0.99).astype(np.float32) if peak > 1 else out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/cough_data"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/coughid/noise_mix"))
    args = ap.parse_args()

    rows = []
    with open(os.path.join(args.data, "metadata.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["path"] = os.path.join(args.data, "wav", r["filename"])
            if os.path.exists(r["path"]):
                rows.append(r)

    coughs = [r for r in rows if r["type"] in COUGH_TYPES]
    speech = [r for r in rows if r["type"] == "speech"]
    noises = [r for r in rows if r["type"] == "noise"]
    print(f"기침 {len(coughs)}개 · 말소리 {len(speech)}개 · 생활잡음 {len(noises)}개\n")

    os.makedirs(args.out, exist_ok=True)
    import wave

    def write(path, x, rate):
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(4); w.setframerate(rate)
            w.writeframes((np.clip(x, -1, 1) * (2 ** 23)).astype("<i4").tobytes())

    rng = np.random.default_rng(0)
    print(f"{'조건':<18}" + "".join(f"{s:>8}dB" for s in SNRS))
    for label, bank in [("+ 말소리", speech), ("+ 생활잡음", noises)]:
        line = f"{label:<18}"
        for snr in SNRS:
            hit = 0
            for r in coughs:
                c, rate = read_wav(r["path"])
                nz, _ = read_wav(bank[rng.integers(len(bank))]["path"])
                tmp = os.path.join(args.out, "mix.wav")
                write(tmp, mix(c, nz, snr), rate)
                hit += int(gate.check(tmp)[0])
            line += f"{hit/len(coughs)*100:>9.0f}%"
        print(line, flush=True)

    print(f"\n기준: 소음 없는 원본에서 98% (60개 중 59개)")
    print("SNR 0dB = 기침과 소음의 크기가 같음 · -5dB = 소음이 더 큼")


if __name__ == "__main__":
    main()

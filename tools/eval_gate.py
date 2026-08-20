#!/usr/bin/env python3
"""기침 게이트(PANNs) 오프라인 평가 — 기침 검출률과 비기침 오탐률, 임계치 곡선.

엣지 검출기(tools/eval_detector.py)가 못 거른 비기침을 서버 게이트가 얼마나 막는지 본다.
임계치는 이 곡선으로 정한다 — 감으로 정하지 말 것.

사용 예:
    python3 eval_gate.py --data ~/Downloads/cough_data
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from app.ml.cough_gate import gate  # noqa: E402

COUGH_TYPES = {"dry", "natural", "throat"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/cough_data"))
    args = ap.parse_args()

    rows = []
    with open(os.path.join(args.data, "metadata.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["path"] = os.path.join(args.data, "wav", r["filename"])
            if os.path.exists(r["path"]):
                rows.append(r)

    print(f"데이터 {len(rows)}개 채점 중...")
    scored = [(r, gate.score(r["path"])) for r in rows]

    print(f"\n{'종류':<10}{'n':>4}{'최저':>10}{'중앙':>10}{'최고':>10}")
    by_type = {}
    for r, s in scored:
        by_type.setdefault(r["type"], []).append(s)
    for t in sorted(by_type, key=lambda k: (k not in COUGH_TYPES, k)):
        v = np.array(by_type[t])
        mark = "기침" if t in COUGH_TYPES else "비기침"
        print(f"{t:<10}{len(v):>4}{v.min():>10.4f}{np.median(v):>10.4f}{v.max():>10.4f}   {mark}")

    pos = [s for r, s in scored if r["type"] in COUGH_TYPES]
    neg = [s for r, s in scored if r["type"] not in COUGH_TYPES]
    if not pos or not neg:
        sys.exit("\n기침/비기침 양쪽 샘플이 모두 있어야 곡선을 그릴 수 있습니다.")

    print(f"\n{'임계치':>10}{'기침 검출률':>13}{'비기침 오탐률':>14}")
    for th in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        tp = sum(s >= th for s in pos) / len(pos)
        fp = sum(s >= th for s in neg) / len(neg)
        cur = "  ← 현재 설정" if abs(th - gate.threshold) < 1e-9 else ""
        print(f"{th:>10.3f}{tp*100:>12.0f}%{fp*100:>13.0f}%{cur}")

    lo, hi = max(neg), min(pos)
    print(f"\n비기침 최고 {lo:.4f} / 기침 최저 {hi:.4f}")
    if hi > lo:
        print(f"완전 분리 가능 — 그 사이 아무 값이나 임계치로 쓸 수 있다 (권장: {(lo*hi)**0.5:.4f})")
    else:
        print("겹침 있음 — 아래 샘플을 직접 들어보고 녹음 불량인지 확인할 것")
        for r, s in sorted(scored, key=lambda x: x[1]):
            if r["type"] in COUGH_TYPES and s <= lo:
                print(f"  기침인데 낮음: {r['filename']}  {s:.4f}")


if __name__ == "__main__":
    main()

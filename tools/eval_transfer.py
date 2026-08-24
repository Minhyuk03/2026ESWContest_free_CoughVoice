#!/usr/bin/env python3
"""2단계 — Coswara로 얻은 보정이 우리 녹음 환경으로 전이되는지 확인한다.

1단계(train_cough_projection.py)는 Coswara 자체 도메인에서 EER 24.9%까지 낮췄다.
그러나 Coswara는 참가자가 각자 자기 기기로 브라우저 녹음한 데이터고, 우리는 고정된
I2S MEMS 마이크 하나로 50cm에서 녹음한다. 도메인이 달라 전이가 안 될 수 있으며,
그 여부가 이 프로젝트의 화자 식별 채택 여부를 가른다.

검증 구성 (s01 기준 verification):
  등록   : s01 ses01 (8/18)
  동일인 : s01 ses02 (8/20)  — 등록과 다른 날
  타인   : x01 (미등록) + s02  — 모두 s01이 아님

출처: Coswara (LEAP Lab, IISc Bangalore) — CC BY 4.0
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
import torch._dynamo  # noqa: F401  — speechbrain 적재 후 첫 import 시 죽는 문제 회피

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from app.ml.identifier import SpeakerIdentifier  # noqa: E402
from train_cough_projection import Projection, eer  # noqa: E402

DATA = os.path.expanduser("~/Downloads/cough_data")
PROJ = os.path.expanduser("~/.cache/coughid/projection.npz")


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "metadata.csv"), encoding="utf-8")))

    def paths(**f):
        return [os.path.join(DATA, "wav", r["filename"]) for r in rows
                if all(r[k] == v for k, v in f.items())]

    ident = SpeakerIdentifier()
    print("임베딩 추출 중...", flush=True)
    enroll = np.stack([ident.embed(p) for p in paths(speaker="s01", session="1")])
    genuine_x = np.stack([ident.embed(p) for p in paths(speaker="s01", session="2")])
    impostor_x = np.stack([ident.embed(p) for p in paths(speaker="x01")
                           + paths(speaker="s02")])
    print(f"  등록 {len(enroll)}개 · 동일인 {len(genuine_x)}개 · 타인 {len(impostor_x)}개\n")

    z = np.load(PROJ, allow_pickle=True)
    mu = z["mu"]
    model = Projection()
    model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files
                           if k.startswith("w_")})
    model.eval()

    def norm(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

    def score(transform):
        E, G, I = (transform(A) for A in (enroll, genuine_x, impostor_x))
        ref = norm(E.mean(0, keepdims=True))
        return (norm(G) @ ref.T).ravel(), (norm(I) @ ref.T).ravel()

    def projected(A):
        with torch.no_grad():
            return model(torch.from_numpy((A - mu).astype(np.float32))).numpy()

    print(f"{'방법':<26}{'EER':>8}{'동일인':>9}{'타인':>9}{'격차':>9}")
    results = {}
    for label, tf in [("원본 코사인", lambda A: A),
                      ("Coswara 중심화", lambda A: A - mu),
                      ("Coswara 투영층", projected)]:
        g, i = score(tf)
        e, t = eer(g, i)
        results[label] = e
        print(f"{label:<26}{e*100:>7.2f}%{g.mean():>+9.3f}{i.mean():>+9.3f}"
              f"{g.mean()-i.mean():>+9.3f}")

    best = min(results, key=results.get)
    print(f"\n최고: {best} — EER {results[best]*100:.2f}%")
    print(f"참고: Coswara 자체 도메인 24.90% · 선행 연구 13.39% · 우연 50%")
    if results[best] > 0.35:
        print("\n→ 전이 실패. 우리 녹음 환경에서는 사용 불가 수준이다.")


if __name__ == "__main__":
    main()

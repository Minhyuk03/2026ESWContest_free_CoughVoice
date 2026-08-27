#!/usr/bin/env python3
"""백본별 중심 벡터를 Coswara에서 계산한다 — 운영에서 쓸 수 있는 **외부** 평균.

왜 필요한가:
    중심화(평균 차감)는 이 프로젝트에서 가장 효과가 큰 보정이다(2026-08-27 WavLM
    통제 데이터: 원본 37.2% → 중심화 25.9%). 그런데 평가 스크립트가 쓰던 평균은
    **평가셋 자신의 평균**이라 두 가지 문제가 있다.

      1. 낙관적이다 — 시험 데이터의 통계를 미리 본 셈이다.
      2. **운영에서 쓸 수 없다** — 서버는 기침 1건을 받는 시점에 평가셋이 없다.

    그래서 학습·평가·운영 어디서나 동일한 고정 벡터가 필요하고, 그것을 화자 980명의
    공개 데이터(Coswara)에서 뽑는다. 투영층의 mu가 ECAPA에 대해 하던 역할과 같다.

주의:
    Coswara는 참가자 1명당 한 세션뿐이라(2026-08-27 확인) **세션 불변성 학습에는
    쓸 수 없다.** 여기서 쓰는 것은 "기침 임베딩의 전역 평균" 하나뿐이며, 그것은
    세션 라벨을 필요로 하지 않으므로 이 한계와 무관하다.

사용:
    tools/compute_center.py --backbone wavlm
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch._dynamo  # noqa: F401  — speechbrain 적재 후 첫 import 시 죽는 문제 회피
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from app.ml.backbone import get_backbone            # noqa: E402
from app.ml.features import (TARGET_RATE, crop_peaks, normalize_rms,  # noqa: E402
                             read_wav)

DATA = os.path.expanduser("~/datasets/Coswara/coughs")
FILES = ("cough-heavy.wav", "cough-shallow.wav")


def out_path(backend: str) -> str:
    return os.path.expanduser(f"~/.cache/coughid/center_{backend}.npz")


def embed_file(backbone, path: str, n_crops: int) -> list:
    x, rate = read_wav(path)
    out = []
    for seg in crop_peaks(x, rate, n_crops):
        t = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0)
        if rate != TARGET_RATE:
            t = torchaudio.functional.resample(t, rate, TARGET_RATE)
        wav = torch.from_numpy(
            np.ascontiguousarray(normalize_rms(t.squeeze(0).numpy()))).unsqueeze(0)
        e = backbone.encode(wav)
        out.append(e / (np.linalg.norm(e) + 1e-9))   # 중심 벡터는 정규화된 분포에서 뽑는다
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--backbone", default="wavlm", choices=("ecapa", "wavlm"))
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="참가자 수 제한(디버그용)")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*", "*"))
                  if os.path.isdir(d))
    if args.limit:
        dirs = dirs[:args.limit]
    print(f"참가자 {len(dirs)}명 · 백본 {args.backbone}", flush=True)

    backbone = get_backbone(args.backbone)
    embs, skipped = [], 0
    for i, d in enumerate(dirs, 1):
        for fn in FILES:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                continue
            try:
                embs.extend(embed_file(backbone, p, args.crops))
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"  건너뜀 {os.path.basename(d)}/{fn}: {exc}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(dirs)}  (임베딩 {len(embs)}개)", flush=True)

    X = np.stack(embs).astype(np.float32)
    mu = X.mean(axis=0)
    path = out_path(args.backbone)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mu=mu, n=len(X), backend=args.backbone)
    print(f"\n임베딩 {len(X)}개 (건너뜀 {skipped}) → 중심 벡터 {mu.shape}")
    print(f"  |mu| = {np.linalg.norm(mu):.4f}   저장: {path}")


if __name__ == "__main__":
    main()

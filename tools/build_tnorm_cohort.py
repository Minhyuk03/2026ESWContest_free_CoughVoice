#!/usr/bin/env python3
"""T-norm 코호트를 만든다 — 등록자가 아닌 사람들의 엣지 클립 임베딩 묶음.

코호트는 "등록본이 아무에게나 얼마나 후한가"를 재는 자다. 그래서 두 조건을 지켜야 한다:

  1. **등록자 본인 클립이 들어가면 안 된다.** μ가 본인 점수에 끌려 올라가 본인이 손해를 본다
  2. **성능을 재는 시험셋과 겹치면 안 된다.** 겹치면 누수다 — 실제로 hwang 시험 블록을
     코호트에 넣었더니 정확도가 100%로 나왔다가, 빼고 다시 재니 89.7%였다(2026-08-28)

기본 입력은 8/26 라벨 세션 아카이브(hwang·choi 80건)다. 둘 다 DB 등록 화자가 아니고
전부 엣지 클립이라 채널이 운영과 같다. 통제 녹음(collect_cough.py)은 마이크 경로가 달라
쓰지 않는다 — 8/25 실측에서 통제 녹음으로 등록하니 동일인 0.419 / 타인 0.636으로
순서가 뒤집힌 적이 있다.

사용:
    ~/.venvs/coughid/bin/python tools/build_tnorm_cohort.py
    ~/.venvs/coughid/bin/python tools/build_tnorm_cohort.py --archive <경로> --out <경로>
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import torch  # noqa: F401  — speechbrain 적재 순서 고정
import torch._dynamo  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from app.ml.identifier import SpeakerIdentifier  # noqa: E402
from app.ml.tnorm import COHORT_PATH, MIN_COHORT  # noqa: E402

DEFAULT_ARCHIVE = os.path.expanduser("~/Downloads/coughid_label_session_20260826")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE,
                    help="cough_id.db 와 audio_store/ 를 가진 아카이브 디렉터리")
    ap.add_argument("--out", default=COHORT_PATH)
    ap.add_argument("--backend", default="wavlm")
    args = ap.parse_args()

    db = os.path.join(args.archive, "cough_id.db")
    if not os.path.isfile(db):
        raise SystemExit(f"아카이브에 cough_id.db 가 없다: {db}")

    con = sqlite3.connect(db)
    paths = []
    for (ap_,) in con.execute("select audio_path from cough_events order by id"):
        p = os.path.join(args.archive, ap_)
        if os.path.exists(p):
            paths.append(p)
    if len(paths) < MIN_COHORT:
        raise SystemExit(f"클립이 {len(paths)}건뿐이다 (최소 {MIN_COHORT}건 필요)")

    ident = SpeakerIdentifier(backend=args.backend)
    print(f"아카이브 {args.archive}\n클립 {len(paths)}건 · 백본 {args.backend} · {ident.embed_dim}차원")
    embs = []
    for i, p in enumerate(paths, 1):
        print(f"\r  임베딩 {i}/{len(paths)}", end="", flush=True)
        v = ident.embed(p, project=False)      # 투영 이중 적용 방지 — 운영 match()와 같은 조건
        n = float(np.linalg.norm(v))
        embs.append(v if n < 1e-12 else (v / n).astype(np.float32))
    print()

    a = np.stack(embs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, a)
    print(f"저장: {args.out}  shape={a.shape}")
    print("\n※ 백본을 바꾸면 차원이 달라져 이 파일은 무효가 된다. 다시 만들 것.")
    print("※ 여기 들어간 화자를 나중에 등록하면 코호트를 다시 만들어야 한다 "
          "(등록자가 코호트에 있으면 본인이 손해를 본다).")


if __name__ == "__main__":
    main()

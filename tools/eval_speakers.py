#!/usr/bin/env python3
"""화자 3명·다중 세션 기준 재평가 — 본인확인(verification)과 다자식별(identification).

기존 eval_transfer.py는 `x01`을 타인으로 하드코딩했으나, x01은 s01과 **같은 사람**이다
(8/24 확인). 여기서는 x01을 s01의 한 세션으로 되돌려 놓는다.

세션 구성 (모두 다른 날):
  s01 : ses01 8/18 · ses02 8/20 · x01→ses04 8/24 · ses03 8/25   (4일)
  s02 : ses01 8/24 · ses02 8/25                                  (2일)
  s03 : ses01 8/24                                               (1일 — 등록 불가, 타인 전용)

**등록 세션과 시험 세션은 반드시 다른 날이어야 한다.** 같은 날 녹음을 쪼개면 목소리가
아니라 그날의 마이크 위치·방 울림을 외운 것을 정확도로 착각한다.

사용:
    python3 eval_speakers.py [--data ~/Downloads/cough_data] [--no-cache]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch._dynamo  # noqa: F401  — speechbrain 적재 후 첫 import 시 죽는 문제 회피

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from app.ml.identifier import SpeakerIdentifier  # noqa: E402
from train_cough_projection import Projection  # noqa: E402

PROJ = os.path.expanduser("~/.cache/coughid/projection.npz")


def cache_path(backend):
    """**백본별로 캐시를 나눈다.** 한 파일에 섞으면 백본을 바꿔도 옛 임베딩이 재사용돼
    바뀐 줄 모르고 옛 수치를 다시 보게 된다."""
    tag = "" if backend == "ecapa" else f"_{backend}"
    return os.path.expanduser(f"~/.cache/coughid/our_embeddings{tag}.npz")

# x01은 s01과 동일인이다. s01의 기존 세션 번호와 겹치지 않게 4번으로 넣는다.
ALIAS = {("x01", "1"): ("s01", "4")}


def load_rows(data_dir):
    """metadata.csv를 읽어 (화자, 세션, 경로)로 정규화한다. neg는 화자가 아니므로 제외."""
    out = []
    with open(os.path.join(data_dir, "metadata.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["speaker"] == "neg":
                continue
            spk, ses = ALIAS.get((r["speaker"], r["session"]), (r["speaker"], r["session"]))
            path = os.path.join(data_dir, "wav", r["filename"])
            if os.path.exists(path):
                out.append((spk, ses, r["type"], path))
    return out


def embeddings(rows, backend="ecapa", use_cache=True):
    """원본 임베딩(투영 전)을 추출한다. 파일 경로를 키로 캐시한다."""
    path = cache_path(backend)
    cache = {}
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        cache = {k: z[k] for k in z.files}

    missing = [p for _, _, _, p in rows if p not in cache]
    if missing:
        print(f"임베딩 추출 {len(missing)}개 (캐시 {len(cache)}개, 백본 {backend})...", flush=True)
        ident = SpeakerIdentifier(backend=backend)
        for i, p in enumerate(missing, 1):
            cache[p] = ident.embed(p, project=False)
            if i % 20 == 0:
                print(f"  {i}/{len(missing)}", flush=True)
        np.savez(path, **cache)
    return cache


def norm(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def eer_curve(genuine, impostor):
    """EER과 그때의 임계치. 모든 점수를 후보 임계치로 훑는다."""
    g, i = np.asarray(genuine), np.asarray(impostor)
    best = None
    for t in np.unique(np.concatenate([g, i])):
        frr = float((g < t).mean())          # 동일인을 거부
        far = float((i >= t).mean())         # 타인을 수락
        if best is None or abs(frr - far) < abs(best[2] - best[3]):
            best = ((frr + far) / 2, float(t), frr, far)
    return best  # (eer, threshold, frr, far)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/cough_data"))
    ap.add_argument("--backbone", default="wavlm", choices=("ecapa", "wavlm"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.data)
    emb = embeddings(rows, backend=args.backbone, use_cache=not args.no_cache)
    print(f"\n백본: {args.backbone}")

    by_ses = defaultdict(list)      # (화자, 세션) -> 임베딩 목록
    for spk, ses, _typ, path in rows:
        by_ses[(spk, ses)].append(emb[path])
    by_ses = {k: np.stack(v) for k, v in by_ses.items()}

    speakers = sorted({s for s, _ in by_ses})
    sessions = defaultdict(list)
    for spk, ses in sorted(by_ses):
        sessions[spk].append(ses)

    print("\n=== 데이터 구성 ===")
    for spk in speakers:
        parts = [f"ses{s}({len(by_ses[(spk, s)])})" for s in sessions[spk]]
        print(f"  {spk}: {' · '.join(parts)}  = {sum(len(by_ses[(spk,s)]) for s in sessions[spk])}개")

    # 변환 — Coswara 투영층·중심화는 **ECAPA 192차원 위에서 만들어진 것**이라
    # 다른 백본에는 적용할 수 없다. WavLM에서는 평가셋 중심화로 대체한다
    # (외부 평균이 아니므로 낙관적일 수 있다는 점을 감안해 읽을 것).
    if args.backbone == "ecapa":
        z = np.load(PROJ, allow_pickle=True)
        mu = z["mu"]
        model = Projection()
        model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files
                               if k.startswith("w_")})
        model.eval()

        def projected(A):
            with torch.no_grad():
                return model(torch.from_numpy((A - mu).astype(np.float32))).numpy()

        transforms = [("원본 코사인", lambda A: A),
                      ("Coswara 중심화", lambda A: A - mu),
                      ("Coswara 투영층", projected)]
    else:
        # 외부(Coswara) 중심화가 운영에서 실제로 쓰는 보정이다. 평가셋 중심화는
        # 상한을 가늠하는 참고치일 뿐 — 시험 데이터의 통계를 미리 본 값이다.
        from app.ml.centering import Centering
        ext = Centering("wavlm")
        mu_eval = np.concatenate(list(by_ses.values())).mean(axis=0)
        transforms = [("원본 코사인", lambda A: A)]
        if ext.available:
            transforms.append(("Coswara 중심화", ext.apply))
            projected = ext.apply
        else:
            projected = lambda A: A - mu_eval
        transforms.append(("평가셋 중심화(참고)", lambda A: A - mu_eval))

    # ---------- 화자 내 일관성 (실시간 전이 가능성의 선행 지표) ----------
    print("\n=== 화자 내 일관성 — 세션 간 상호 유사도 ===")
    print("  8/25 실측(ECAPA 투영층): 통제 녹음 +0.324 / 실사용 엣지 클립 +0.110")
    P = {k: norm(projected(v)) for k, v in by_ses.items()}
    for spk in speakers:
        ses_list = sessions[spk]
        within, cross = [], []
        for a in ses_list:
            S = P[(spk, a)] @ P[(spk, a)].T
            within += S[np.triu_indices_from(S, k=1)].tolist()
        for a, b in itertools.combinations(ses_list, 2):
            cross += (P[(spk, a)] @ P[(spk, b)].T).ravel().tolist()
        w = f"{np.mean(within):+.3f}" if within else "  n/a"
        c = f"{np.mean(cross):+.3f}" if cross else "  n/a"
        print(f"  {spk}  세션내 {w}   세션간 {c}")
    others = []
    for a, b in itertools.combinations(sorted(by_ses), 2):
        if a[0] != b[0]:
            others += (P[a] @ P[b].T).ravel().tolist()
    print(f"  타인 간                    {np.mean(others):+.3f}")

    # ---------- 본인 확인 ----------
    enrollable = [s for s in speakers if len(sessions[s]) >= 2]
    print(f"\n=== 본인 확인 (verification) — 등록 가능 화자: {', '.join(enrollable)} ===")
    print("  각 화자의 한 세션으로 등록 → 그 화자의 다른 세션=동일인, 나머지 화자 전체=타인")

    print(f"\n{'방법':<20}{'EER':>8}{'임계치':>9}{'동일인':>9}{'타인':>9}{'격차':>9}")
    pooled = {}
    for label, tf in transforms:
        T = {k: norm(tf(v)) for k, v in by_ses.items()}
        g_all, i_all = [], []
        for spk in enrollable:
            for enroll_ses in sessions[spk]:
                ref = T[(spk, enroll_ses)].mean(0, keepdims=True)
                ref = ref / (np.linalg.norm(ref) + 1e-9)
                for (o_spk, o_ses), M in T.items():
                    if o_spk == spk and o_ses == enroll_ses:
                        continue            # 등록에 쓴 세션은 시험에서 제외
                    s = (M @ ref.T).ravel().tolist()
                    (g_all if o_spk == spk else i_all).extend(s)
        e, t, _, _ = eer_curve(g_all, i_all)
        pooled[label] = (e, t, g_all, i_all)
        print(f"{label:<20}{e*100:>7.2f}%{t:>9.3f}{np.mean(g_all):>+9.3f}"
              f"{np.mean(i_all):>+9.3f}{np.mean(g_all)-np.mean(i_all):>+9.3f}")
    print(f"  (동일인 트라이얼 {len(pooled['원본 코사인'][2])} · 타인 {len(pooled['원본 코사인'][3])})")

    # **"(참고)"가 붙은 변환은 선택에서 제외한다.** 평가셋 중심화는 시험 데이터의 평균을
    # 미리 본 값이라 운영에서 재현할 수 없다. 이후 운용 곡선·다자 식별은 실제로 배포
    # 가능한 설정에서 나와야 의미가 있다.
    deployable = [k for k in pooled if "(참고)" not in k]
    best_label = min(deployable, key=lambda k: pooled[k][0])
    print(f"\n최고(배포 가능): {best_label} — EER {pooled[best_label][0]*100:.2f}%")
    ref = [k for k in pooled if "(참고)" in k]
    if ref:
        r = min(ref, key=lambda k: pooled[k][0])
        print(f"  참고: {r} {pooled[r][0]*100:.2f}% — 시험 데이터 평균을 쓴 값이라 운영 불가")

    # 화자별 분해
    print(f"\n=== 화자별 EER ({best_label}) ===")
    tf = dict(transforms)[best_label]
    T = {k: norm(tf(v)) for k, v in by_ses.items()}
    for spk in enrollable:
        g, i = [], []
        for enroll_ses in sessions[spk]:
            ref = T[(spk, enroll_ses)].mean(0, keepdims=True)
            ref = ref / (np.linalg.norm(ref) + 1e-9)
            for (o_spk, o_ses), M in T.items():
                if o_spk == spk and o_ses == enroll_ses:
                    continue
                s = (M @ ref.T).ravel().tolist()
                (g if o_spk == spk else i).extend(s)
        e, t, _, _ = eer_curve(g, i)
        print(f"  {spk}  EER {e*100:5.2f}%  (동일인 {len(g)} / 타인 {len(i)})  "
              f"동일인 {np.mean(g):+.3f} · 타인 {np.mean(i):+.3f}")

    # ---------- 운용 곡선 ----------
    e, t, g_all, i_all = pooled[best_label]
    g_all, i_all = np.array(g_all), np.array(i_all)
    print(f"\n=== 운용 곡선 ({best_label}) ===")
    print(f"{'임계치':>7}{'재현율':>9}{'FAR':>9}{'정밀도':>9}")
    # **임계치 후보는 점수 분포에서 뽑는다.** 고정 격자(0.20~0.60)는 ECAPA 스케일에
    # 맞춘 것이라, 유사도가 전반적으로 높은 백본에서는 곡선이 통째로 범위 밖으로 나간다.
    lo = float(min(g_all.min(), i_all.min()))
    hi = float(max(g_all.max(), i_all.max()))
    grid = sorted(set(np.round(np.linspace(lo, hi, 11), 3)))
    for th in grid:
        tp = int((g_all >= th).sum())
        fp = int((i_all >= th).sum())
        rec = tp / len(g_all)
        far = fp / len(i_all)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        print(f"{th:>7.2f}{rec*100:>8.1f}%{far*100:>8.1f}%{prec*100:>8.1f}%")

    # ---------- 다자 식별 ----------
    print(f"\n=== 다자 식별 (identification) — 등록 {len(enrollable)}명 ({best_label}) ===")
    print("  등록: 각 화자의 첫 세션 / 시험: 등록에 쓰지 않은 모든 기침")
    refs = {}
    for spk in enrollable:
        r = T[(spk, sessions[spk][0])].mean(0, keepdims=True)
        refs[spk] = (r / (np.linalg.norm(r) + 1e-9)).ravel()
    names = list(refs)
    R = np.stack([refs[n] for n in names])

    correct = total = 0
    unreg_correct = unreg_total = 0
    for (o_spk, o_ses), M in T.items():
        if o_spk in refs and o_ses == sessions[o_spk][0]:
            continue
        scores = M @ R.T
        pred = [names[j] for j in scores.argmax(1)]
        if o_spk in refs:
            total += len(pred)
            correct += sum(p == o_spk for p in pred)
        else:
            unreg_total += len(pred)
            unreg_correct += int((scores.max(1) < t).sum())   # 임계치 미만 = unknown 처리
    print(f"  등록 화자 정확도  {correct}/{total} = {correct/total*100:.1f}%")
    if unreg_total:
        print(f"  미등록자 거부율   {unreg_correct}/{unreg_total} = "
              f"{unreg_correct/unreg_total*100:.1f}%  (임계치 {t:.3f})")


if __name__ == "__main__":
    main()

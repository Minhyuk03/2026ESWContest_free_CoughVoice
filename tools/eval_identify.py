#!/usr/bin/env python3
"""P3 오프라인 평가 — 등록/검증 세션을 나눠 코사인 유사도 분포와 임계치 곡선을 뽑는다.

**등록 세션과 검증 세션은 반드시 다른 날이어야 한다.** 같은 날 녹음을 쪼개 평가하면
모델이 목소리가 아니라 그날의 마이크 위치·방 울림을 외운 것을 정확도로 착각하게 된다.

사용 예:
    python3 eval_identify.py --data ~/Downloads/cough_data
    python3 eval_identify.py --data ~/Downloads/cough_data --ablation
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from app.ml.identifier import SpeakerIdentifier, bytes_to_embedding  # noqa: E402

NEG_SPEAKER = "neg"   # 기침이 아닌 소리 — 화자 식별 대상이 아니다


def load_metadata(data_dir):
    rows = []
    with open(os.path.join(data_dir, "metadata.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["path"] = os.path.join(data_dir, "wav", r["filename"])
            if os.path.exists(r["path"]):
                rows.append(r)
    return rows


def summarize(name, sims):
    if not sims:
        return f"  {name:<24} (샘플 없음)"
    a = np.array(sims)
    return (f"  {name:<24} n={len(a):<3} "
            f"평균={a.mean():.3f}  최저={a.min():.3f}  최고={a.max():.3f}  "
            f"표준편차={a.std():.3f}")


def evaluate(rows, enroll_ses, test_ses, prep, verbose=True):
    """등록 세션으로 registry를 만들고, 등록에 쓰이지 않은 모든 기침을 시험한다.

    트라이얼 구성 원칙:
      - 등록 화자는 `s`로 시작하는 ID만. **`x`(외부인)는 절대 등록하지 않는다** —
        미등록자를 unknown으로 거르는지 보려면 registry 밖에 있어야 한다.
      - 등록에 사용한 (화자, 세션) 조합은 시험에서 제외한다. 같은 녹음을 등록과
        검증에 함께 쓰면 그날의 마이크 위치를 외운 것이 정확도로 둔갑한다.
      - 등록 화자의 다른 세션 = genuine, 미등록 화자 = impostor.

    등록·검증 양쪽에 동일한 prep을 적용한다 — 전처리가 어긋나면 유사도가 무의미해진다.
    """
    ident = SpeakerIdentifier()

    enroll = defaultdict(list)
    for r in rows:
        if r["speaker"].startswith("s") and int(r["session"]) == enroll_ses:
            enroll[r["speaker"]].append(r["path"])
    if not enroll:
        sys.exit(f"세션 {enroll_ses}에 등록용 기침 샘플이 없습니다.")

    speaker_ids = {name: i + 1 for i, name in enumerate(sorted(enroll))}
    registry = []
    for name, paths in sorted(enroll.items()):
        blob, n = ident.enroll(paths, **prep)
        registry.append((speaker_ids[name], blob))
        if verbose:
            print(f"  등록: {name} → id={speaker_ids[name]}, 샘플 {n}개")

    outsiders = sorted({r["speaker"] for r in rows
                        if r["speaker"] not in speaker_ids and r["speaker"] != NEG_SPEAKER})
    if verbose:
        print(f"  미등록(외부인): {', '.join(outsiders) if outsiders else '없음'}")

    genuine, impostor = [], []
    by_type = defaultdict(list)
    trials = []
    for r in rows:
        if r["speaker"] == NEG_SPEAKER:
            continue                        # 기침이 아니므로 화자 판정 대상이 아니다
        enrolled = r["speaker"] in speaker_ids
        if enrolled and int(r["session"]) == enroll_ses:
            continue                        # 등록에 쓴 녹음은 시험하지 않는다

        emb = ident.embed(r["path"], project=False, **prep)
        res = ident.match(emb, registry)
        sim = res.similarity if res.similarity is not None else -1.0
        best_id = _argmax_id(emb, registry)
        best_name = next((n for n, i in speaker_ids.items() if i == best_id), None)
        trials.append((r, sim, best_name, enrolled))
        if enrolled:
            genuine.append(sim)
            by_type[r["type"]].append(sim)
        else:
            impostor.append(sim)            # 미등록 화자 = unknown이 정답
    return genuine, impostor, by_type, trials, speaker_ids, ident, registry


def _argmax_id(emb, registry):
    best_id, best = None, -2.0
    for pid, blob in registry:
        s = float(np.dot(emb, bytes_to_embedding(blob)))
        if s > best:
            best_id, best = pid, s
    return best_id


def channel_bias_check(ident, rows, registry, prep):
    """비음성 네거티브(박수·문 닫기 등)와의 유사도로 채널 편향을 진단한다.

    이 소리들은 사람 목소리가 아니므로 유사도가 낮아야 정상이다. 여기에 높은 점수가
    나오면 임베딩이 화자가 아니라 **녹음 환경(무음·잔향)** 을 보고 있다는 뜻이고,
    그 상태의 높은 동일인 유사도는 성능이 아니라 착시다.
    """
    negs = defaultdict(list)
    for r in rows:
        if r["speaker"] == NEG_SPEAKER:
            emb = ident.embed(r["path"], project=False, **prep)
            negs[r["type"]].append(_best_sim(emb, registry))
    if not negs:
        return None
    print("\n채널 편향 진단 (낮을수록 좋음 — 목소리가 아닌 소리)")
    worst = -1.0
    for t in sorted(negs):
        print(summarize(f"neg/{t}", negs[t]))
        worst = max(worst, float(np.max(negs[t])))
    return worst


def _best_sim(emb, registry):
    return max(float(np.dot(emb, bytes_to_embedding(blob))) for _, blob in registry)


def threshold_table(genuine, impostor):
    print("\n임계치 곡선")
    print("  임계치   FRR(동일인 거부)   FAR(타인 수락)")
    rows = []
    for t in np.arange(0.20, 0.96, 0.05):
        frr = float(np.mean(np.array(genuine) < t)) if genuine else float("nan")
        far = float(np.mean(np.array(impostor) >= t)) if impostor else float("nan")
        rows.append((t, frr, far))
        far_s = "  —  (타인 샘플 없음)" if not impostor else f"{far*100:8.1f}%"
        print(f"  {t:.2f}  {frr*100:12.1f}%   {far_s}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/cough_data"))
    ap.add_argument("--enroll-session", type=int, default=1)
    ap.add_argument("--test-session", type=int, default=2)
    ap.add_argument("--ablation", action="store_true",
                    help="크롭·정규화 유무에 따른 유사도 변화를 비교")
    args = ap.parse_args()

    rows = load_metadata(args.data)
    print(f"데이터 {len(rows)}개 로드: {args.data}")
    print(f"등록 세션 {args.enroll_session} → 검증 세션 {args.test_session}\n")

    variants = [("크롭+정규화 (기본)", dict(crop=True, normalize=True))]
    if args.ablation:
        variants += [
            ("크롭만",              dict(crop=True, normalize=False)),
            ("정규화만",            dict(crop=False, normalize=True)),
            ("원본 3초 그대로",     dict(crop=False, normalize=False)),
        ]

    last = None
    for label, prep in variants:
        print("=" * 62)
        print(f"■ {label}")
        print("=" * 62)
        genuine, impostor, by_type, trials, speaker_ids, ident, registry = evaluate(
            rows, args.enroll_session, args.test_session, prep)

        print("\n동일 화자 유사도 (등록과 다른 날 녹음)")
        print(summarize("전체", genuine))
        for t in sorted(by_type):
            print(summarize(f"  └ {t}", by_type[t]))

        if impostor:
            print("\n타인 유사도 (미등록 화자 — unknown이 정답)")
            print(summarize("전체", impostor))
        else:
            print("\n타인 유사도: 미등록 화자 샘플이 없어 측정 불가.")
            print("  → 등록 화자가 1명뿐이면 FAR을 알 수 없고, 임계치를 확정할 수 없다.")
            print("  → s02 이상 추가 수집이 임계치 결정의 선행 조건이다.")

        worst_neg = channel_bias_check(ident, rows, registry, prep)
        if worst_neg is not None and genuine:
            gap = float(np.mean(genuine)) - worst_neg
            print(f"\n  분리 격차(동일인 평균 − 비음성 최고) = {gap:+.3f}"
                  f"{'  ← 겹침, 전처리 재검토 필요' if gap <= 0 else ''}")

        threshold_table(genuine, impostor)
        last = (genuine, impostor)
        print()

    genuine, impostor = last
    if genuine and impostor:
        gaps = min(genuine) - max(impostor)
        print(f"분리 여유(min 동일인 − max 타인) = {gaps:+.3f}  "
              f"{'양수면 완전 분리 가능' if gaps > 0 else '겹침 있음 — 오판 불가피'}")


if __name__ == "__main__":
    main()

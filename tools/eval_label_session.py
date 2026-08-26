#!/usr/bin/env python3
"""라벨 세션 → 실사용 EER. `label_session.py`가 남긴 라벨 구간과 서버 이벤트를 맞춰 잰다.

왜 필요한가:
    지금까지 화자 식별 수치(EER 16.7%)는 전부 `collect_cough.py`의 통제 녹음
    (고정 거리·조용한 방·연속 녹음)에서 나왔다. 실사용 엣지 클립은 화자 내 일관성이
    +0.110으로 통제 조건(+0.324)의 3분의 1이라, 통제 수치가 실시간 성능을 대변하지
    못한다. 이 스크립트가 실사용 조건에서 EER을 처음으로 낸다.

**동일인 쌍은 반드시 서로 다른 블록에서 만든다.** 같은 블록 안에서 쌍을 만들면
목소리가 아니라 그 몇 분간의 마이크 위치·자세·옷 스치는 소리를 맞히게 되고,
EER이 실제보다 크게 좋아 보인다. 이 프로젝트가 반복해서 빠진 함정이 정확히 그것이라
(s01 ses03 vs s02 ses02가 16분 간격이라 다른 사람인데 같은 사람 수준으로 나왔다),
여기서는 블록 간 쌍만 동일인으로 인정한다. 화자당 블록이 1개뿐이면 경고하고 제외한다.

사용:
    python3 tools/eval_label_session.py --labels label_session_*.json
    python3 tools/eval_label_session.py --labels a.json --server http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import itertools
import json
import os
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import torch
import torch._dynamo  # noqa: F401  — speechbrain 적재 후 첫 import 시 죽는 문제 회피

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from app.ml.identifier import SpeakerIdentifier  # noqa: E402
from train_cough_projection import Projection  # noqa: E402

PROJ = os.path.expanduser("~/.cache/coughid/projection.npz")
CACHE = os.path.expanduser("~/.cache/coughid/label_session_embeddings.npz")
AUDIO_DIR = os.path.expanduser("~/.cache/coughid/label_session_wav")
DEFAULT_SERVER = os.environ.get("COUGHID_SERVER", "http://127.0.0.1:8000")


def parse_ts(s: str) -> dt.datetime:
    """ISO 문자열 → tz-aware UTC. 서버는 Z, 라벨 파일은 +09:00으로 준다."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def load_blocks(paths):
    """라벨 파일들 → [(화자, 시작, 종료, 출처)]. 블록에 고유 번호를 매긴다."""
    blocks = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        for b in doc.get("blocks", []):
            blocks.append({
                "speaker": b["speaker"],
                "start": parse_ts(b["start"]),
                "end": parse_ts(b["end"]),
                "source": os.path.basename(p),
                "id": len(blocks),
            })
    return blocks


def fetch_events(server, limit):
    url = f"{server.rstrip('/')}/events?limit={limit}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def match(events, blocks, margin):
    """이벤트를 라벨 구간에 배정한다. 어느 구간에도 없으면 버린다(세션 밖 잡음)."""
    out, unmatched = [], 0
    slack = dt.timedelta(seconds=margin)
    for e in events:
        t = parse_ts(e["captured_at"])
        hit = [b for b in blocks if b["start"] - slack <= t <= b["end"] + slack]
        if not hit:
            unmatched += 1
            continue
        # 구간이 겹치면 시작이 늦은 쪽(더 좁은 쪽)을 택한다
        b = max(hit, key=lambda b: b["start"])
        out.append((b["speaker"], b["id"], e["id"]))
    return out, unmatched


def download(server, event_id):
    path = os.path.join(AUDIO_DIR, f"{event_id}.wav")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"{server.rstrip('/')}/events/{event_id}/audio"
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(path, "wb") as f:
            f.write(r.read())
    except Exception:
        return None
    return path


def embeddings(pairs, server, use_cache=True):
    """이벤트 id → 원본 ECAPA 임베딩(투영 전). id를 키로 캐시한다."""
    cache = {}
    if use_cache and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: z[k] for k in z.files}

    missing = [ev for _, _, ev in pairs if str(ev) not in cache]
    if missing:
        print(f"오디오 내려받기·임베딩 추출 {len(missing)}개 (캐시 {len(cache)}개)...", flush=True)
        ident = SpeakerIdentifier()
        failed = 0
        for i, ev in enumerate(missing, 1):
            wav = download(server, ev)
            if wav is None:
                failed += 1
                continue
            cache[str(ev)] = ident.embed(wav, project=False)
            if i % 20 == 0:
                print(f"  {i}/{len(missing)}", flush=True)
        np.savez(CACHE, **cache)
        if failed:
            print(f"  ⚠ 오디오 없음 {failed}개 — 보존기간(7일)이 지났거나 삭제된 이벤트")
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
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="label_session_*.json (글롭 가능)")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--limit", type=int, default=5000, help="서버에서 가져올 이벤트 수")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="구간 경계 여유(초). 클립 시각이 살짝 밀릴 때만 쓴다")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    os.makedirs(AUDIO_DIR, exist_ok=True)

    paths = sorted({p for pat in args.labels for p in glob.glob(pat)} or set(args.labels))
    blocks = load_blocks(paths)
    if not blocks:
        sys.exit("라벨 블록이 없습니다.")

    events = fetch_events(args.server, args.limit)
    pairs, unmatched = match(events, blocks, args.margin)

    print(f"\n라벨 파일 {len(paths)}개 · 블록 {len(blocks)}개 · 서버 이벤트 {len(events)}건")
    print(f"구간에 배정 {len(pairs)}건 / 구간 밖 {unmatched}건")

    print("\n=== 블록 구성 ===")
    per_block = defaultdict(int)
    for _, bid, _ in pairs:
        per_block[bid] += 1
    for b in blocks:
        kst = b["start"].astimezone(dt.timezone(dt.timedelta(hours=9)))
        print(f"  블록{b['id']:<3}{b['speaker']:<10}{kst.strftime('%m/%d %H:%M:%S')}  {per_block[b['id']]:>4}건")

    emb = embeddings(pairs, args.server, use_cache=not args.no_cache)

    by_block = defaultdict(list)     # (화자, 블록) -> 임베딩 목록
    for spk, bid, ev in pairs:
        if str(ev) in emb:
            by_block[(spk, bid)].append(emb[str(ev)])
    by_block = {k: np.stack(v) for k, v in by_block.items() if v}

    speakers = sorted({s for s, _ in by_block})
    blocks_of = defaultdict(list)
    for spk, bid in sorted(by_block):
        blocks_of[spk].append(bid)

    if len(speakers) < 2:
        sys.exit("\n화자가 2명 미만입니다. 타인 대조군이 없어 EER을 낼 수 없습니다.")

    usable = [s for s in speakers if len(blocks_of[s]) >= 2]
    if not usable:
        print("\n⚠ 블록이 2개 이상인 화자가 없습니다. 동일인 쌍을 블록 간으로 만들 수 없어")
        print("  EER은 건너뜁니다. 같은 화자를 서로 다른 시간대에 두 번 이상 녹음하세요.")

    # 변환 3종 — eval_speakers.py와 동일
    z = np.load(PROJ, allow_pickle=True)
    mu = z["mu"]
    model = Projection()
    model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("w_")})
    model.eval()

    def projected(A):
        with torch.no_grad():
            return model(torch.from_numpy((A - mu).astype(np.float32))).numpy()

    transforms = [("원본 코사인", lambda A: A),
                  ("Coswara 중심화", lambda A: A - mu),
                  ("Coswara 투영층", projected)]

    # ---------- 화자 내 일관성 ----------
    print("\n=== 화자 내 일관성 (투영층) ===")
    print("  기준: 통제 녹음 +0.324 / 8/25 실사용 엣지 클립 +0.110")
    P = {k: norm(projected(v)) for k, v in by_block.items()}
    for spk in speakers:
        within, cross = [], []
        for b in blocks_of[spk]:
            S = P[(spk, b)] @ P[(spk, b)].T
            within += S[np.triu_indices_from(S, k=1)].tolist()
        for a, b in itertools.combinations(blocks_of[spk], 2):
            cross += (P[(spk, a)] @ P[(spk, b)].T).ravel().tolist()
        w = f"{np.mean(within):+.3f}" if within else "  n/a"
        c = f"{np.mean(cross):+.3f}" if cross else "  n/a"
        print(f"  {spk:<10} 블록내 {w}   블록간 {c}")
    others = []
    for a, b in itertools.combinations(sorted(by_block), 2):
        if a[0] != b[0]:
            others += (P[a] @ P[b].T).ravel().tolist()
    print(f"  {'타인 간':<10}        {np.mean(others):+.3f}")

    if not usable:
        return

    # ---------- 본인 확인 (블록 간 동일인 쌍만) ----------
    print(f"\n=== 본인 확인 (verification) — 블록 2개 이상인 화자: {', '.join(usable)} ===")
    print("  동일인 = 같은 화자의 서로 다른 블록 / 타인 = 다른 화자")
    print(f"\n{'방법':<20}{'EER':>8}{'임계치':>9}{'동일인':>9}{'타인':>9}{'격차':>9}")

    pooled = {}
    for label, tf in transforms:
        T = {k: norm(tf(v)) for k, v in by_block.items()}
        gen, imp = [], []
        for spk in usable:
            for a, b in itertools.combinations(blocks_of[spk], 2):
                gen += (T[(spk, a)] @ T[(spk, b)].T).ravel().tolist()
        for a, b in itertools.combinations(sorted(by_block), 2):
            if a[0] != b[0]:
                imp += (T[a] @ T[b].T).ravel().tolist()
        e, thr, _, _ = eer_curve(gen, imp)
        pooled[label] = (e, thr, float(np.mean(gen)), float(np.mean(imp)), gen, imp)
        print(f"{label:<20}{e*100:7.2f}%{thr:9.3f}{np.mean(gen):+9.3f}{np.mean(imp):+9.3f}"
              f"{np.mean(gen)-np.mean(imp):+9.3f}")

    best = min(pooled, key=lambda k: pooled[k][0])
    print(f"\n최고: {best} — EER {pooled[best][0]*100:.2f}%")
    print(f"  (동일인 쌍 {len(pooled[best][4])}개 / 타인 쌍 {len(pooled[best][5])}개)")
    print("\n참고: 통제 녹음 기준 EER 16.7%. 이 값이 그보다 크게 나쁘면")
    print("      통제 조건 수치는 시연 근거로 쓸 수 없다는 뜻이다.")


if __name__ == "__main__":
    main()

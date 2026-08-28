#!/usr/bin/env python3
"""백본 교체 실험 — 같은 실사용 클립·같은 프로토콜에서 임베더만 바꿔 EER을 비교한다.

배경 (2026-08-27):
    VoxCeleb 사전학습 ECAPA는 실사용 엣지 클립에서 EER 46.25%로 우연 수준이다.
    블록 *내부* 유사도(+0.23~0.25)만 높고 블록 *간*은 타인 수준(+0.18)으로 떨어져,
    임베딩이 화자가 아니라 그 몇 분간의 마이크 위치·자세를 인코딩한다는 것이
    `tools/eval_label_session.py`로 확인됐다.

    남은 저비용 카드가 "더 강한 사전학습 임베더로 교체(추론만, 학습 없음)"라서
    이 스크립트를 만들었다. 판정 기준은 실험 전에 못박는다:
        **블록 간 EER이 30% 밑으로 내려가지 않으면 이 경로는 중단.**

`eval_label_session.py`와의 차이:
    - 서버가 필요 없다. 아카이브(DB + audio_store + 라벨 JSON)만으로 재현된다.
    - 백본을 `--backbone`으로 고른다. 오디오 전처리는 양쪽 동일하게 유지해
      차이가 백본에서만 오도록 한다.

프로토콜은 그대로다 — **동일인 쌍은 반드시 서로 다른 블록에서 만든다.**
같은 블록 안에서 쌍을 만들면 채널 아티팩트를 화자 변별력으로 착각한다.

사용:
    tools/eval_backbone.py --archive ~/Downloads/coughid_label_session_20260826 \
        --backbone wavlm
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import torch
import torch._dynamo  # noqa: F401  — speechbrain 적재 후 첫 import 시 죽는 문제 회피

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from app.ml.features import crop_active, normalize_rms, read_wav  # noqa: E402

CACHE_DIR = os.path.expanduser("~/.cache/coughid")
WAVLM_MODEL = "microsoft/wavlm-base-plus-sv"
CROP_S = 1.2          # features.CROP_S 와 동일 — ECAPA 경로와 맞춘다
TARGET_RATE = 16000


# ---------------------------------------------------------------- 데이터

def parse_ts(s: str) -> dt.datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)   # DB는 naive=UTC 규약
    return d.astimezone(dt.timezone.utc)


def load_events(archive: str):
    """아카이브 DB에서 (id, 촬영시각, wav경로)를 읽는다."""
    db = os.path.join(archive, "cough_id.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, captured_at, audio_path FROM cough_events ORDER BY id").fetchall()
    con.close()
    out = []
    for ev, ts, path in rows:
        wav = path if os.path.isabs(path) else os.path.join(archive, path)
        if os.path.exists(wav):
            out.append((ev, parse_ts(ts), wav))
    return out


KST = dt.timezone(dt.timedelta(hours=9))


def load_blocks(paths, offset=0):
    """라벨 파일들 → 블록 목록. **날짜(KST)를 함께 들고 다닌다** — 동일인 쌍을
    "같은 날 다른 블록"과 "다른 날"로 갈라 보기 위해서다. 이 프로젝트의 실패는
    항상 날짜 경계에서 났으므로 두 수치를 섞으면 안 된다."""
    blocks = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        for b in doc.get("blocks", []):
            start = parse_ts(b["start"])
            blocks.append({"speaker": b["speaker"], "start": start,
                           "end": parse_ts(b["end"]),
                           "date": start.astimezone(KST).strftime("%m/%d"),
                           "id": offset + len(blocks)})
    return blocks


def assign(events, blocks):
    """이벤트를 라벨 구간에 배정. 구간 밖은 버린다(세션 밖 잡음)."""
    out, dropped = [], 0
    for ev, t, wav in events:
        hit = [b for b in blocks if b["start"] <= t <= b["end"]]
        if not hit:
            dropped += 1
            continue
        b = max(hit, key=lambda b: b["start"])
        out.append((b["speaker"], b["id"], ev, wav, b["date"]))
    return out, dropped


# ---------------------------------------------------------------- 백본

def load_audio(wav: str, crop: bool) -> np.ndarray:
    """양 백본이 공유하는 전처리. 엣지 클립은 이미 16kHz/16bit라 리샘플이 없다."""
    x, rate = read_wav(wav)
    if rate != TARGET_RATE:
        raise ValueError(f"{wav}: {rate}Hz — 엣지 클립은 16kHz여야 한다")
    if crop:
        x = crop_active(x, rate, crop_s=CROP_S)
    return normalize_rms(x)


def make_backbone(name: str, model_id=None):
    """공용 백본을 그대로 쓴다 — 평가와 운영이 다른 임베더를 쓰는 사고를 막기 위함."""
    from app.ml.backbone import BACKENDS as IMPL, get_backbone
    if model_id and name == "wavlm":
        return IMPL["wavlm"](model_id=model_id)       # 체크포인트 비교용
    return get_backbone(name)


def embed_all(backbone, xs):
    out = []
    for i, x in enumerate(xs, 1):
        t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        out.append(backbone.encode(t))
        if i % 20 == 0:
            print(f"  {i}/{len(xs)}", flush=True)
    return np.stack(out)


BACKENDS = ("ecapa", "wavlm")


# ---------------------------------------------------------------- 평가

def norm(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def eer_curve(genuine, impostor):
    g, i = np.asarray(genuine), np.asarray(impostor)
    best = None
    for t in np.unique(np.concatenate([g, i])):
        frr = float((g < t).mean())
        far = float((i >= t).mean())
        if best is None or abs(frr - far) < abs(best[2] - best[3]):
            best = ((frr + far) / 2, float(t), frr, far)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, nargs="+",
                    help="cough_id.db + audio_store/ + 라벨 JSON 이 든 폴더 (여러 개 가능)")
    ap.add_argument("--labels", default=None, help="라벨 JSON (기본: 아카이브 안에서 탐색)")
    ap.add_argument("--backbone", default="wavlm", choices=sorted(BACKENDS))
    ap.add_argument("--model", default=None,
                    help="wavlm 백본의 체크포인트 교체 (기본 microsoft/wavlm-base-plus-sv)")
    ap.add_argument("--enroll", type=int, default=0, metavar="N",
                    help="등록 템플릿 프로토콜: 등록 블록의 N개 평균 vs 타 블록 단일 클립. "
                         "0이면 단일-단일 쌍(기본)")
    ap.add_argument("--no-crop", action="store_true",
                    help="1.2초 크롭 없이 2.5초 클립 전체를 넣는다")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    # 아카이브를 여러 개 받는다. **이벤트 id는 아카이브마다 1부터 다시 매겨지므로**
    # 캐시 키·블록 id를 아카이브별로 분리하지 않으면 서로 다른 녹음이 덮어써진다.
    archives = [os.path.expanduser(a) for a in args.archive]
    pairs, dropped, n_events, blocks = [], 0, 0, []
    for ai, archive in enumerate(archives):
        labels = [args.labels] if (args.labels and len(archives) == 1) else [
            os.path.join(archive, f) for f in sorted(os.listdir(archive))
            if f.startswith("label_session_") and f.endswith(".json")]
        if not labels:
            sys.exit(f"라벨 JSON을 찾지 못했습니다: {archive}")
        bl = load_blocks(labels, offset=len(blocks))
        ev = load_events(archive)
        pr, dr = assign(ev, bl)
        blocks += bl
        n_events += len(ev)
        dropped += dr
        pairs += [(spk, bid, f"{ai}:{e}", wav, date) for spk, bid, e, wav, date in pr]
    crop = not args.no_crop

    print(f"\n백본 {args.backbone} · 크롭 {'1.2초' if crop else '없음(2.5초 전체)'}")
    print(f"아카이브 {len(archives)}개 · 이벤트 {n_events}건 → 구간 배정 {len(pairs)}건 "
          f"/ 구간 밖 {dropped}건")

    model_tag = ""
    if args.model:
        model_tag = "_" + args.model.split("/")[-1].replace("-", "")
    tag = f"{args.backbone}{model_tag}{'' if crop else '_full'}"
    cache_path = os.path.join(CACHE_DIR, f"backbone_{tag}.npz")
    cache = {}
    if not args.no_cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cache = {k: z[k] for k in z.files}

    todo = [(ev, wav) for _, _, ev, wav, _ in pairs if str(ev) not in cache]
    if todo:
        print(f"임베딩 추출 {len(todo)}개 (캐시 {len(cache)}개)...", flush=True)
        backbone = make_backbone(args.backbone, args.model)
        xs = [load_audio(wav, crop) for _, wav in todo]
        embs = embed_all(backbone, xs)
        for (ev, _), e in zip(todo, embs):
            cache[str(ev)] = e
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, **cache)

    by_block = defaultdict(list)
    block_date = {}
    for spk, bid, ev, _, date in pairs:
        if str(ev) in cache:
            by_block[(spk, bid)].append(cache[str(ev)])
            block_date[(spk, bid)] = date
    by_block = {k: np.stack(v) for k, v in by_block.items() if v}

    speakers = sorted({s for s, _ in by_block})
    blocks_of = defaultdict(list)
    for spk, bid in sorted(by_block):
        blocks_of[spk].append(bid)
    usable = [s for s in speakers if len(blocks_of[s]) >= 2]

    print("\n=== 블록 구성 ===")
    for (spk, bid), A in sorted(by_block.items(), key=lambda kv: kv[0][1]):
        print(f"  블록{bid:<3}{spk:<10}{block_date[(spk, bid)]}  {len(A):>4}건  "
              f"({A.shape[1]}차원)")
    n_days = {s: len({block_date[(s, b)] for b in bs}) for s, bs in
              [(s, [b for sp, b in by_block if sp == s]) for s in
               sorted({s for s, _ in by_block})]}
    multi_day = [s for s, n in n_days.items() if n >= 2]
    print(f"  날짜가 2일 이상인 화자: {', '.join(multi_day) if multi_day else '없음'}"
          + ("" if multi_day else "  ← 날짜 간 EER을 낼 수 없다"))
    if not usable:
        sys.exit("블록이 2개 이상인 화자가 없습니다 — 블록 간 동일인 쌍을 만들 수 없습니다.")

    # 외부(Coswara) 중심화 = 운영에서 실제로 쓰는 보정. 평가셋 중심화는 참고치다
    # (시험 데이터의 평균을 미리 본 값이라 낙관적이고, 서버에서는 재현 불가능하다).
    from app.ml.centering import Centering
    ext = Centering(args.backbone)
    mu = np.concatenate(list(by_block.values())).mean(axis=0)
    transforms = [("원본 코사인", lambda A: A)]
    if ext.available:
        transforms.append(("Coswara 중심화", ext.apply))
    transforms.append(("평가셋 중심화(참고)", lambda A: A - mu))

    print("\n=== 화자 내 일관성 ===")
    print("  블록 내부가 높고 블록 간이 타인 수준이면 = 화자가 아니라 채널을 재고 있는 것")
    P = {k: norm(v) for k, v in by_block.items()}
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

    print(f"\n=== 본인 확인 (블록 간 동일인 쌍만) — 화자: {', '.join(usable)} ===")
    print(f"\n{'방법':<20}{'EER':>8}{'임계치':>9}{'동일인':>9}{'타인':>9}{'격차':>9}")
    results = {}
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
        results[label] = e
        print(f"{label:<20}{e*100:7.2f}%{thr:9.3f}{np.mean(gen):+9.3f}{np.mean(imp):+9.3f}"
              f"{np.mean(gen)-np.mean(imp):+9.3f}")
        if label == "원본 코사인":
            n_gen, n_imp = len(gen), len(imp)

    if args.enroll:
        print(f"\n=== 등록 템플릿 프로토콜 (등록 {args.enroll}개 평균 vs 타 블록 단일 클립) ===")
        print("  실제 운용 구성이다 — 등록은 여러 개, 판정은 클립 1개.")
        print(f"\n{'방법':<20}{'EER':>8}{'임계치':>9}{'동일인':>9}{'타인':>9}{'격차':>9}")
        for label, tf in transforms:
            T = {k: norm(tf(v)) for k, v in by_block.items()}
            gen, imp = [], []
            for (espk, ebid), E in T.items():
                k = min(args.enroll, len(E))
                ref = E[:k].mean(axis=0)
                ref = ref / (np.linalg.norm(ref) + 1e-9)
                for (tspk, tbid), X in T.items():
                    if tbid == ebid:
                        continue          # 같은 블록은 채널이 같아 의미 없다
                    (gen if tspk == espk else imp).extend((X @ ref).tolist())
            e, thr, _, _ = eer_curve(gen, imp)
            print(f"{label:<20}{e*100:7.2f}%{thr:9.3f}{np.mean(gen):+9.3f}"
                  f"{np.mean(imp):+9.3f}{np.mean(gen)-np.mean(imp):+9.3f}")
            results[label + f" (등록{args.enroll})"] = e
        print(f"  (동일인 {len(gen)} / 타인 {len(imp)})")

    # "(참고)"는 평가셋 평균을 쓴 값이라 운영에서 재현할 수 없다 — 선택에서 뺀다.
    deployable = [k for k in results if "(참고)" not in k]
    best_label = min(deployable, key=lambda k: results[k])
    best = results[best_label]
    print(f"\n최고(배포 가능): {best_label} — EER {best*100:.2f}%  "
          f"(동일인 쌍 {n_gen} / 타인 쌍 {n_imp})")
    ref = [k for k in results if "(참고)" in k]
    if ref:
        r = min(ref, key=lambda k: results[k])
        print(f"  참고: {r} {results[r]*100:.2f}% — 시험 데이터 평균을 쓴 값이라 운영 불가")
    print("\n기준선: ECAPA 실사용 49.9%(원본)·46.3%(중심화) · 우연 50% · 통제 녹음 WavLM 30.5%")


if __name__ == "__main__":
    main()

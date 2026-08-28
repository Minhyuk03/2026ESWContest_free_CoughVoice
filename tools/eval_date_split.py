#!/usr/bin/env python3
"""날짜 분리 조건에서 s01 식별 성능을 잰다 (열린 집합 재현율 + 닫힌 집합 2지선다 + 발작 단위).

**이 프로젝트에서 좋게 나온 수치는 예외 없이 세션 내부였다.** 등록 직후 같은 자리에서
검증하면 0.86~0.87이 나오지만 그건 목소리가 아니라 그 몇 분간의 마이크 위치·자세를
맞힌 것이다. 그래서 이 도구는 **등록일과 시험일이 다른 것을 강제**한다 —
등록에 쓴 이벤트(enrolled=1)는 시험에서 무조건 빼고, 시험 클립이 등록 클립과 같은 날
(KST 기준)이면 경고한다.

읽는 곳 (전부 오프라인, 운영 DB에 아무것도 쓰지 않는다):
    운영 DB   server/cough_id.db + server/audio_store/   — s01 등록본과 오늘 시험 클립
    아카이브  ~/Downloads/coughid_label_session_20260826/ — 타인(hwang·choi) 엣지 클립 88건
              label_session_20260826_2236.json 이 블록 정답지다

내는 것:
    ① 열린 집합 — s01 템플릿 대 오늘 클립 유사도 분포. 어제 잰 타인 FAR과 짝지어
      임계치별 재현율/FAR을 나란히 놓는다. **이 경로는 2026-08-27 기준 운용 지점이 없다**
      (FAR 5%를 만드는 0.88이 본인 클립조차 전부 밑돈다). 확인용으로만 낸다
    ② 닫힌 집합 2지선다 — 등록 2인(s01 · hwang) 중 누구를 고르는지. 양방향으로 잰다
    ③ 발작 단위 + 마진 보류 — 1/3/5/10개 묶음 정확도, 마진별 커버리지·정확도
    ④ choi(미등록)를 넣었을 때 보류로 빠지는 비율

사용:
    # 오늘(KST) 들어온 s01 클립 전부를 시험셋으로
    ~/.venvs/coughid/bin/python tools/eval_date_split.py --since 2026-08-28T14:00

    # 이벤트 id를 직접 지정
    ~/.venvs/coughid/bin/python tools/eval_date_split.py --event-ids 137-156
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys

import numpy as np
import torch  # noqa: F401  — speechbrain 적재 순서 고정
import torch._dynamo  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "server"))
from app.ml.identifier import SpeakerIdentifier, bytes_to_embedding  # noqa: E402

LIVE_DB = os.path.join(REPO, "server", "cough_id.db")
LIVE_ROOT = os.path.join(REPO, "server")
ARCHIVE = os.path.expanduser("~/Downloads/coughid_label_session_20260826")
KST = dt.timezone(dt.timedelta(hours=9))

# 2026-08-27 실측. 등록=8/27 엣지 8개 / 시험=8/26 엣지 80건(hwang 40·choi 40), 다른 날.
# 타인이 2명뿐이라 잠정치다 — choi 하나 때문에 0.75에서 15%↔60%로 튄다.
FAR_20260827 = {0.75: 0.375, 0.80: 0.275, 0.85: 0.125, 0.88: 0.050}


def l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


def parse_ids(spec: str) -> list[int]:
    """"137-156,160" → [137..156, 160]"""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_live(args):
    """운영 DB에서 s01 등록본과 시험 클립을 읽는다."""
    con = sqlite3.connect(LIVE_DB)
    row = con.execute("select alias, embedding_ref, sample_count from persons where id=1").fetchone()
    if row is None or not row[1]:
        raise SystemExit("persons id=1(s01)에 등록본이 없다. 먼저 enroll-from-events 를 할 것")
    alias, blob, n_enroll = row
    tmpl = l2(bytes_to_embedding(blob))

    enrolled_days = {
        dt.datetime.fromisoformat(r[0]).replace(tzinfo=dt.timezone.utc).astimezone(KST).date()
        for r in con.execute("select captured_at from cough_events where enrolled=1 and person_id=1")
    }

    q = ("select id, captured_at, audio_path, cough_score, peak_rms, enrolled "
         "from cough_events order by id")
    rows = con.execute(q).fetchall()
    tests = []
    for eid, cap, ap, score, prms, enrolled in rows:
        when = dt.datetime.fromisoformat(cap).replace(tzinfo=dt.timezone.utc).astimezone(KST)
        if enrolled:
            continue                      # 등록에 쓴 것은 시험에서 반드시 제외
        if args.event_ids and eid not in args.event_ids:
            continue
        if args.since and when < args.since:
            continue
        if args.until and when > args.until:
            continue
        path = os.path.join(LIVE_ROOT, ap)
        if not os.path.exists(path):
            print(f"  ⚠ 이벤트 {eid}: 오디오 없음 — 건너뜀")
            continue
        tests.append(dict(id=eid, when=when, path=path, score=score, peak_rms=prms))
    return alias, tmpl, n_enroll, enrolled_days, tests


def load_archive():
    """아카이브에서 hwang·choi 블록별 클립 경로를 읽는다."""
    labels = json.load(open(os.path.join(ARCHIVE, "label_session_20260826_2236.json")))
    blocks = [dict(speaker=b["speaker"],
                   start=dt.datetime.fromisoformat(b["start"]),
                   end=dt.datetime.fromisoformat(b["end"]),
                   clips=[]) for b in labels["blocks"]]
    con = sqlite3.connect(os.path.join(ARCHIVE, "cough_id.db"))
    for eid, cap, ap in con.execute("select id, captured_at, audio_path from cough_events order by id"):
        when = dt.datetime.fromisoformat(cap).replace(tzinfo=dt.timezone.utc)
        for b in blocks:
            if b["start"] <= when <= b["end"]:
                p = os.path.join(ARCHIVE, ap)
                if os.path.exists(p):
                    b["clips"].append(dict(id=eid, when=when, path=p))
                break
    return blocks


def split_blocks(items, gap_s=60.0):
    """시간 간격으로 블록을 나눈다. 휴식을 사이에 둔 두 묶음은 다른 블록이다.

    발작(연속 기침) 묶음을 만들 때 휴식 구간을 건너뛰면 안 된다 — 그 구간을 건너뛴
    묶음은 실제로 존재하지 않는 발작이고, 자세가 바뀐 전후를 평균해 버려서
    "자세 변화에 견디는가"라는 질문 자체를 지워버린다.
    """
    blocks, cur = [], []
    for it in items:
        if cur and (it["when"] - cur[-1]["when"]).total_seconds() > gap_s:
            blocks.append(cur); cur = []
        cur.append(it)
    if cur:
        blocks.append(cur)
    return blocks


def embed_all(ident, items, tag):
    """project=False — 투영층 이중 적용을 막는다(운영 match()와 같은 조건)."""
    out = []
    for i, it in enumerate(items, 1):
        print(f"\r  {tag} 임베딩 {i}/{len(items)}", end="", flush=True)
        out.append(l2(ident.embed(it["path"], project=False)))
    print()
    return np.stack(out) if out else np.zeros((0, ident.embed_dim), np.float32)


def windows(A, B, spans, k):
    """블록 경계를 넘지 않는 크기 k 슬라이딩 묶음. spans = [(start, end), ...] 인덱스 구간."""
    out = []
    for s0, s1 in spans:
        for i in range(s0, s1 - k + 1):
            out.append((A[i:i + k].mean(), B[i:i + k].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None, help="시험 클립 시작 시각 (KST, 예 2026-08-28T14:00)")
    ap.add_argument("--until", type=str, default=None)
    ap.add_argument("--event-ids", type=str, default=None, help="예 137-156,160")
    ap.add_argument("--bouts", type=str, default="1,3,5,10")
    ap.add_argument("--margins", type=str, default="0.02,0.05,0.10,0.15,0.20")
    args = ap.parse_args()
    args.since = dt.datetime.fromisoformat(args.since).replace(tzinfo=KST) if args.since else None
    args.until = dt.datetime.fromisoformat(args.until).replace(tzinfo=KST) if args.until else None
    args.event_ids = set(parse_ids(args.event_ids)) if args.event_ids else None
    sizes = [int(x) for x in args.bouts.split(",")]
    margins = [float(x) for x in args.margins.split(",")]

    alias, tmpl, n_enroll, enrolled_days, tests = load_live(args)
    print(f"\n등록: {alias} (id=1) · 샘플 {n_enroll}개 · 등록일(KST) "
          f"{', '.join(str(d) for d in sorted(enrolled_days))}")
    if not tests:
        raise SystemExit("시험 클립이 없다. --since / --event-ids 를 확인할 것")
    test_days = {t["when"].date() for t in tests}
    print(f"시험: {len(tests)}건 · 날짜(KST) {', '.join(str(d) for d in sorted(test_days))}")
    same = test_days & enrolled_days
    if same:
        print(f"  ⚠⚠ 등록일과 같은 날짜의 시험 클립이 있다 ({same}). **날짜 분리가 아니다** — "
              f"여기서 나오는 수치는 성능이 아니다")
    print(f"  게이트 점수 {min(t['score'] or 0 for t in tests):.3f}~{max(t['score'] or 0 for t in tests):.3f}"
          f" / peak_rms {min(t['peak_rms'] or 0 for t in tests):.3f}~{max(t['peak_rms'] or 0 for t in tests):.3f}")

    ident = SpeakerIdentifier(backend="wavlm")
    print(f"백본 {ident.backbone.name if hasattr(ident.backbone,'name') else 'wavlm'} · {ident.embed_dim}차원")
    if tmpl.size != ident.embed_dim:
        raise SystemExit(f"등록본 차원 {tmpl.size} ≠ 임베더 {ident.embed_dim}. 재등록이 필요하다")

    test_blocks = split_blocks(tests)
    spans_test, off = [], 0
    for b in test_blocks:
        spans_test.append((off, off + len(b))); off += len(b)
    print(f"  시험 블록 {len(test_blocks)}개: " + " / ".join(
        f"{len(b)}건 {b[0]['when']:%H:%M:%S}~{b[-1]['when']:%H:%M:%S}" for b in test_blocks))
    if len(test_blocks) > 1:
        gaps = [(test_blocks[i + 1][0]["when"] - test_blocks[i][-1]["when"]).total_seconds()
                for i in range(len(test_blocks) - 1)]
        print(f"  블록 간 휴식: " + " / ".join(f"{g/60:.1f}분" for g in gaps))

    E_test = embed_all(ident, tests, "시험")
    sim_self = E_test @ tmpl

    # ── ① 열린 집합 ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("① 열린 집합 — s01 템플릿 대 시험 클립 (날짜 분리)")
    print("=" * 72)
    print(f"  유사도  최소 {sim_self.min():.3f} / 중앙 {np.median(sim_self):.3f} / "
          f"평균 {sim_self.mean():.3f} / 최대 {sim_self.max():.3f}")
    print(f"\n  {'임계치':>7}{'재현율(오늘)':>14}{'FAR(8/27 타인 80건)':>22}")
    for th in sorted(FAR_20260827):
        rec = float((sim_self >= th).mean())
        print(f"  {th:>7.2f}{rec:>13.1%}{FAR_20260827[th]:>21.1%}")
    print("  ※ FAR은 타인 2명(hwang·choi) 기준 잠정치다. 재현율과 FAR이 같은 임계치에서")
    print("     동시에 쓸 만한 지점이 없으면 열린 집합(미등록 거부)은 주장하지 말 것")

    # ── ② 닫힌 집합 2지선다 ────────────────────────────────────────────────
    blocks = load_archive()
    hw = [b for b in blocks if b["speaker"] == "hwang"]
    ch = [b for b in blocks if b["speaker"] == "choi"]
    print("\n" + "=" * 72)
    print("② 닫힌 집합 2지선다 — 등록 2인(s01 · hwang) 중 택1")
    print("=" * 72)
    print(f"  아카이브 블록: " + " / ".join(
        f"{b['speaker']}#{i} {len(b['clips'])}건" for i, b in enumerate(blocks)))

    E_hw = [embed_all(ident, b["clips"], f"hwang#{i}") for i, b in enumerate(hw)]
    tmpl_hw = l2(E_hw[0].mean(0))          # hwang 블록 1개로 등록 템플릿
    test_hw = E_hw[1]                      # 나머지 블록이 hwang 시험셋 (블록 분리)

    a_self, b_self = sim_self, E_test @ tmpl_hw            # s01 클립: 정답 s01
    a_hw, b_hw = test_hw @ tmpl_hw, test_hw @ tmpl         # hwang 클립: 정답 hwang
    spans_hw = [(0, len(a_hw))]
    print(f"\n  s01 클립 {len(a_self)}건  : s01 {a_self.mean():.3f} vs hwang {b_self.mean():.3f}"
          f"  → 정답 선택 {float((a_self > b_self).mean()):.1%}")
    print(f"  hwang 클립 {len(a_hw)}건: hwang {a_hw.mean():.3f} vs s01 {b_hw.mean():.3f}"
          f"  → 정답 선택 {float((a_hw > b_hw).mean()):.1%}")
    both = np.concatenate([(a_self > b_self), (a_hw > b_hw)])
    print(f"  양방향 합산 기침 1개 정확도: {float(both.mean()):.1%}  (8/27 블록분리 기준 84.4%)")
    if len(test_blocks) > 1:
        print("  s01 블록별 (평균이 만든 착시인지 확인):")
        for bi, (s0, s1) in enumerate(spans_test):
            w = (a_self[s0:s1] > b_self[s0:s1])
            print(f"    블록{bi + 1} {s1 - s0}건 → {float(w.mean()):.1%} "
                  f"(s01 {a_self[s0:s1].mean():.3f} vs hwang {b_self[s0:s1].mean():.3f})")

    # ── ③ 발작 단위 + 마진 보류 ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("③ 발작 단위 묶음 + 마진 보류")
    print("=" * 72)
    print(f"  {'묶음':>5}{'s01 정확도':>12}{'hwang 정확도':>14}{'합산':>9}{'표본':>8}")
    pooled = {}
    for k in sizes:
        rows = []
        for name, A, B, sp in (("s01", a_self, b_self, spans_test),
                               ("hwang", a_hw, b_hw, spans_hw)):
            g = windows(A, B, sp, k)      # 블록 경계를 넘지 않는다
            if not g:
                rows.append((name, None, 0, []))
                continue
            rows.append((name, sum(1 for x, y in g if x > y) / len(g), len(g), g))
        allg = rows[0][3] + rows[1][3]
        pooled[k] = allg
        acc = sum(1 for x, y in allg if x > y) / len(allg) if allg else None
        f = lambda v: f"{v:.1%}" if v is not None else "표본부족"
        print(f"  {k:>5}{f(rows[0][1]):>12}{f(rows[1][1]):>14}{f(acc):>9}{len(allg):>8}")
    print("  8/27 블록분리 기준: 1개 84.4% · 3개 90.8% · 5개 95.4% · 10개 99.2%")

    print(f"\n  마진 보류 (두 등록본 점수차가 마진 미만이면 판정 보류)")
    for k in sizes:
        allg = pooled.get(k) or []
        if not allg:
            continue
        print(f"    묶음 {k}개 (표본 {len(allg)})")
        for m in margins:
            kept = [(x, y) for x, y in allg if abs(x - y) >= m]
            cov = len(kept) / len(allg)
            acc = (sum(1 for x, y in kept if x > y) / len(kept)) if kept else float("nan")
            print(f"      마진 {m:.2f} → 커버리지 {cov:>6.1%} · 정확도 "
                  + (f"{acc:.1%}" if kept else "  —  "))
    print("  8/27 기준: 5개 묶음·마진 0.05 → 커버리지 80.6% / 정확도 99.4%")

    # ── ④ choi = 미등록자 ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("④ choi(미등록) — 등록 2인 중 하나로 억지 배정되는지, 보류로 빠지는지")
    print("=" * 72)
    E_ch = np.concatenate([embed_all(ident, b["clips"], f"choi#{i}") for i, b in enumerate(ch)])
    c_s01, c_hw = E_ch @ tmpl, E_ch @ tmpl_hw
    print(f"  choi {len(E_ch)}건: s01 {c_s01.mean():.3f} / hwang {c_hw.mean():.3f}")
    print(f"  {'마진':>7}{'보류율':>10}   (보류되지 않으면 미등록자가 등록 2인 중 하나로 찍힌다)")
    for m in margins:
        print(f"  {m:>7.2f}{float((np.abs(c_s01 - c_hw) < m).mean()):>9.1%}")
    for th in sorted(FAR_20260827):
        print(f"  임계치 {th:.2f}에서 choi가 둘 중 하나를 넘김: "
              f"{float((np.maximum(c_s01, c_hw) >= th).mean()):.1%}")

    print("\n※ 좋은 수치가 나왔다면 위 '등록일/시험일' 줄부터 다시 볼 것.")
    print("  이 프로젝트에서 좋게 나온 수치는 예외 없이 세션 내부였다.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""실시간 라벨 세션 — 엣지가 평소대로 도는 상태에서 '누가 언제 기침했는지'를 기록한다.

왜 필요한가:
    지금까지 화자 식별 평가는 전부 `collect_cough.py`로 모은 통제 녹음(50cm 고정·조용한 방)
    으로만 했다. 그런데 실사용 엣지 클립은 화자 내 일관성이 +0.149로 통제 조건(+0.34)의
    절반이라, 통제 조건 수치가 실시간 성능을 대변하지 못한다.
    실시간 클립에 정답 라벨을 붙여야 실사용 EER을 처음으로 잴 수 있다.

**collect_cough.py를 쓰면 안 된다.** 그 스크립트는 arecord로 마이크를 점유하고,
그동안 엣지 서비스는 아무것도 듣지 못한다(2026-08-25 확인: 수집 세션 15:37~15:40,
15:53~15:55 구간에 엣지 이벤트가 0건이고 그 사이 공백에만 찍혔다).
이 스크립트는 녹음하지 않는다. 시각만 적는다. 소리는 평소처럼 엣지가 잡는다.

**화자를 번갈아 두 번씩 한다.** 각자 한 번씩만 하면 화자 차이와 녹음 세션 차이가
분리되지 않는다. 지금까지 이 프로젝트가 반복해서 빠진 함정이 정확히 그것이다
(s01 ses03과 s02 ses02는 다른 사람인데 16분 간격이라 유사도가 같은 사람 수준으로 나왔다).

사용:
    python3 tools/label_session.py            # 맥북에서 서버가 돌 때
    python3 tools/label_session.py --server http://<서버IP>:8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

# 맥미니(192.168.219.134)는 2026-08-27부로 구성에서 제외했다. 서버는 맥북에서 돈다.
DEFAULT_SERVER = os.environ.get("COUGHID_SERVER", "http://127.0.0.1:8000")


def get(server: str, path: str, timeout: float = 10.0):
    with urllib.request.urlopen(f"{server}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def count_since(server: str, since_iso: str) -> int:
    """블록 시작 이후 도착한 이벤트 수. 서버 시각이 아니라 captured_at 기준."""
    try:
        rows = get(server, "/events?limit=200")
    except Exception:
        return -1
    since = datetime.fromisoformat(since_iso)
    n = 0
    for e in rows:
        t = datetime.fromisoformat(e["captured_at"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if t >= since:
            n += 1
    return n


class Ticker(threading.Thread):
    """블록 진행 중 도착 건수를 계속 보여 준다.

    엣지가 죽었거나 마이크를 놓친 것을 20회 다 기침한 뒤에 알게 되면 세션을 통째로
    다시 해야 한다. 실시간으로 보여 주면 두세 번 기침해 보고 바로 알 수 있다.
    """

    def __init__(self, server: str, since: str):
        super().__init__(daemon=True)
        self.server, self.since, self.stop_flag = server, since, threading.Event()

    def run(self):
        while not self.stop_flag.wait(3.0):
            n = count_since(self.server, self.since)
            elapsed = int((datetime.now(timezone.utc)
                           - datetime.fromisoformat(self.since).astimezone(timezone.utc))
                          .total_seconds())
            msg = (f"    도착 {n}건 · {elapsed//60}분 {elapsed%60}초 경과"
                   if n >= 0 else "    (서버 응답 없음)")
            print(f"\r{msg:<60}", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--out", default=None, help="라벨 파일 경로 (기본: 실행 폴더에 자동 생성)")
    args = ap.parse_args()

    print(f"서버: {args.server}")
    try:
        get(args.server, "/health")
        ov = get(args.server, "/stats/overview")
    except Exception as e:
        print(f"서버에 연결할 수 없습니다: {e}")
        return 1
    print(f"  엣지 온라인: {'예' if ov.get('device_online') else '아니오 ← 확인 필요'}")
    print(f"  오늘 누적: {ov.get('today_cough_count')}건")

    if not ov.get("device_online"):
        print("\n엣지가 최근 5분 안에 이벤트를 보낸 적이 없습니다.")
        print("파이에서 서비스 상태를 확인하세요:  systemctl --user status coughid-edge")
        if input("그래도 진행할까요? [y/N] ").strip().lower() != "y":
            return 1

    print("""
────────────────────────────────────────────────────────────
 진행 방법
   · 한 블록당 기침 20회, 4~5초 간격으로 천천히
     (엣지 쿨다운이 2초라 그보다 빠르면 한 건으로 합쳐집니다)
   · 평소 지내는 자리에서 자연스럽게. 거리를 일부러 맞추지 마세요
   · 블록 사이에 2~3분 쉬고, 화자를 번갈아 두 번씩 하세요
       예)  s01 → s02 → s01 → s02
   · 한 블록이 끝나면 Enter, 전부 끝나면 화자 이름을 비우고 Enter
────────────────────────────────────────────────────────────
""")

    blocks = []
    while True:
        label = f"블록 {len(blocks) + 1}"
        speaker = input(f"{label} — 화자 ID (예: s01, 끝내려면 빈칸): ").strip()
        if not speaker:
            break
        input(f"  준비되면 Enter → 시작 ({speaker})")
        start = now_iso()
        print(f"  시작 {start[11:19]} — 기침 시작하세요. 끝나면 Enter")
        tick = Ticker(args.server, start)
        tick.start()
        input()
        tick.stop_flag.set()
        tick.join(timeout=4.0)
        end = now_iso()
        # 마지막 이벤트가 서버에 저장될 시간을 준다 (게이트+식별이 CPU로 수 초 걸린다)
        print("\r    마무리 대기 중...".ljust(60), end="", flush=True)
        time.sleep(8)
        n = count_since(args.server, start)
        print(f"\r  종료 {end[11:19]} — 이 블록에서 {n}건 도착".ljust(60))
        blocks.append({"speaker": speaker, "start": start, "end": end, "events_seen": n})
        if n < 8:
            print("    ⚠ 도착 건수가 적습니다. 더 크게·천천히 하거나 마이크 쪽으로 가까이 가세요")

    if not blocks:
        print("기록된 블록이 없습니다.")
        return 1

    speakers = {b["speaker"] for b in blocks}
    out = args.out or f"label_session_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"server": args.server, "created_at": now_iso(), "blocks": blocks},
                  f, ensure_ascii=False, indent=2)

    print(f"\n라벨 파일 저장: {out}")
    print(f"{'블록':<6}{'화자':<8}{'시작':<10}{'종료':<10}{'건수':>5}")
    for i, b in enumerate(blocks, 1):
        print(f"{i:<6}{b['speaker']:<8}{b['start'][11:19]:<10}{b['end'][11:19]:<10}"
              f"{b['events_seen']:>5}")

    # 설계 조건을 만족했는지 알려 준다. 부족한 채로 분석하면 또 세션 효과를 본다.
    print()
    if len(speakers) < 2:
        print("⚠ 화자가 1명뿐입니다. 타인 대조군이 없어 EER을 낼 수 없습니다.")
    counts = {s: sum(1 for b in blocks if b["speaker"] == s) for s in speakers}
    if any(v < 2 for v in counts.values()):
        few = [s for s, v in counts.items() if v < 2]
        print(f"⚠ 블록이 1개뿐인 화자: {', '.join(few)}")
        print("  같은 화자가 서로 다른 시간대에 두 번 이상 있어야 화자 차이와 세션 차이가 분리됩니다.")
    if len(speakers) >= 2 and all(v >= 2 for v in counts.values()):
        print("✓ 화자 2명 이상 · 각자 2블록 이상 — 분석 조건을 만족합니다.")
    total = sum(b["events_seen"] for b in blocks)
    print(f"\n총 {total}건. 이 파일을 개발 장비로 옮겨 분석하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

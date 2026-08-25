#!/usr/bin/env python3
"""24시간 연속 동작 관측 — 체크리스트의 "크래시 0회 검증" 근거를 만든다.

무엇을 재는가:
  - 서버 가용성: /health 응답 여부와 왕복 시간
  - 엣지 생존: /stats/overview의 device_online (최근 5분 내 이벤트 수신 여부)
  - 이벤트 유입 연속성: 누적 건수가 늘어나는가, 끊긴 구간은 없는가
  - 수집→저장 지연: 이벤트의 received_at − captured_at
  - 서버 재시작 횟수: launchd 로그의 uvicorn 기동 줄 개수 (맥미니에서 실행할 때만)

**맥미니에서 돌리는 것을 권장한다.** 맥북은 뚜껑을 덮거나 네트워크를 벗어나면
관측이 끊기는데, 그러면 "서버가 죽은 것"과 "관측자가 죽은 것"을 구분할 수 없다.

주의 — 지연시간에 대하여:
    received_at − captured_at은 **엣지→서버 구간만**이다. NFR-03의 "전체 지연 3초"는
    여기에 대시보드 폴링(0~3초)이 더 붙는다. 또 두 시각은 서로 다른 기계의 벽시계라
    NTP 오차가 섞인다. 음수가 나오면 시계가 어긋난 것이다.

사용:
    python3 tools/soak_monitor.py --hours 24                  # 관측 시작
    nohup python3 tools/soak_monitor.py --hours 24 &          # 터미널 닫아도 유지
    python3 tools/soak_monitor.py --report soak_YYYY....csv   # 결과 요약
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_SERVER = os.environ.get("COUGHID_SERVER", "http://192.168.219.134:8000")
LAUNCHD_LOG = os.path.expanduser("~/Library/Logs/coughid/server.log")
FIELDS = ["ts", "health_ok", "latency_ms", "device_online", "online_source", "today_count",
          "active_alerts", "total_events", "max_lag_s", "restarts", "error"]


def get(server: str, path: str, timeout: float = 10.0):
    with urllib.request.urlopen(f"{server}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def count_restarts() -> int:
    """launchd 로그에서 uvicorn 기동 줄을 센다. KeepAlive가 죽은 프로세스를 되살리면
    여기 줄이 하나 늘어나므로, 이 값이 관측 중 증가하면 크래시가 있었다는 뜻이다."""
    if not os.path.exists(LAUNCHD_LOG):
        return -1                      # 이 기계에 로그가 없다(맥북에서 실행 중)
    n = 0
    try:
        with open(LAUNCHD_LOG, errors="ignore") as f:
            for line in f:
                if "Started server process" in line or "Uvicorn running on" in line:
                    n += 1
    except OSError:
        return -1
    return n


def sample(server: str) -> dict:
    row = {k: "" for k in FIELDS}
    row["ts"] = datetime.now(timezone.utc).astimezone().isoformat()
    row["restarts"] = count_restarts()
    t0 = time.time()
    try:
        get(server, "/health", timeout=10)
        row["latency_ms"] = round((time.time() - t0) * 1000, 1)
        row["health_ok"] = 1
    except Exception as e:
        row["health_ok"] = 0
        row["error"] = f"health: {type(e).__name__}"
        return row

    try:
        ov = get(server, "/stats/overview", timeout=15)
        row["device_online"] = 1 if ov.get("device_online") else 0
        # 판정 근거를 함께 남긴다. "last_event"면 그 값은 생존이 아니라
        # "최근에 진짜 기침이 있었나"라서 오프라인 표시를 그대로 믿으면 안 된다.
        row["online_source"] = ov.get("device_online_source", "last_event")
        row["today_count"] = ov.get("today_cough_count", "")
        row["active_alerts"] = ov.get("active_alerts", "")
    except Exception as e:
        row["error"] = f"overview: {type(e).__name__}"

    try:
        ev = get(server, "/events?limit=20", timeout=15)
        row["total_events"] = len(ev)
        lags = []
        for e in ev:
            try:
                c = datetime.fromisoformat(e["captured_at"])
                r = datetime.fromisoformat(e["received_at"])
                lags.append((r - c).total_seconds())
            except Exception:
                pass
        if lags:
            row["max_lag_s"] = round(max(lags), 2)
    except Exception as e:
        row["error"] = (row["error"] + " | " if row["error"] else "") + f"events: {type(e).__name__}"
    return row


def run(server: str, hours: float, interval: int, out: str) -> int:
    end = time.time() + hours * 3600
    new = not os.path.exists(out)
    print(f"관측 시작 — {server}\n  기록: {out}\n  간격 {interval}초 · {hours}시간")
    base_restarts = count_restarts()
    print(f"  시작 시점 서버 기동 횟수: "
          f"{base_restarts if base_restarts >= 0 else '측정 불가(맥북에서 실행 중)'}\n")
    fails = 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        while time.time() < end:
            row = sample(server)
            w.writerow(row)
            f.flush()                      # 중간에 끊겨도 여기까지는 남는다
            if not row["health_ok"]:
                fails += 1
                print(f"  [{row['ts'][11:19]}] 응답 없음 ({row['error']})", flush=True)
            elif row["device_online"] == 0 and row["online_source"] == "heartbeat":
                print(f"  [{row['ts'][11:19]}] 서버 정상 · **엣지 오프라인**(하트비트 끊김)",
                      flush=True)
            time.sleep(interval)
    print(f"\n관측 종료. 실패 {fails}회. 요약: "
          f"python3 {os.path.basename(__file__)} --report {out}")
    return 0


def report(path: str) -> int:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("기록이 비어 있습니다.")
        return 1
    n = len(rows)
    ok = sum(1 for r in rows if r["health_ok"] == "1")
    hb = [r for r in rows if r.get("online_source") == "heartbeat"]
    edge_on = sum(1 for r in hb if r["device_online"] == "1")
    lat = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
    lags = [float(r["max_lag_s"]) for r in rows if r["max_lag_s"]]
    restarts = [int(r["restarts"]) for r in rows if r["restarts"] not in ("", "-1")]

    start = datetime.fromisoformat(rows[0]["ts"])
    stop = datetime.fromisoformat(rows[-1]["ts"])
    span_h = (stop - start).total_seconds() / 3600

    print(f"=== 관측 요약 ===")
    print(f"  기간        {start:%m-%d %H:%M} ~ {stop:%m-%d %H:%M}  ({span_h:.1f}시간, {n}회 측정)")
    print(f"  서버 가용   {ok}/{n} = {ok/n*100:.2f}%")
    if hb:
        print(f"  엣지 온라인 {edge_on}/{len(hb)} = {edge_on/len(hb)*100:.2f}%  (하트비트 기준)")
    else:
        print(f"  엣지 온라인: 측정 불가 — 관측 구간에 하트비트가 없었다"
              f"(device_online이 '최근 기침 있었나'를 뜻하던 시기)")
    if lat:
        lat.sort()
        print(f"  /health 응답 중앙값 {lat[len(lat)//2]:.1f}ms · 최대 {lat[-1]:.1f}ms")
    if lags:
        lags.sort()
        print(f"  수집→저장 지연 중앙값 {lags[len(lags)//2]:.2f}s · 최대 {lags[-1]:.2f}s")
        print(f"    (엣지→서버 구간만. 대시보드 폴링 0~3초가 추가로 붙는다)")
    if restarts:
        delta = max(restarts) - min(restarts)
        print(f"  서버 기동 횟수 증가 {delta}회 — "
              + ("**크래시 0회**" if delta == 0 else f"**관측 중 {delta}회 재시작됨**"))
    else:
        print(f"  서버 재시작 여부: 측정 불가 (맥미니의 launchd 로그가 있는 기계에서 실행해야 함)")

    down = [r for r in rows if r["health_ok"] != "1"]
    if down:
        print(f"\n  응답 없던 시각 {len(down)}건:")
        for r in down[:10]:
            print(f"    {r['ts'][5:19]}  {r['error']}")
        if len(down) > 10:
            print(f"    ... 외 {len(down)-10}건")
    else:
        print(f"\n  응답 실패 0건 — 관측 구간 내내 서버가 살아 있었다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--interval", type=int, default=60, help="측정 간격(초)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None, help="기록 파일을 읽어 요약만 출력")
    args = ap.parse_args()
    if args.report:
        return report(args.report)
    out = args.out or f"soak_{datetime.now():%Y%m%d_%H%M}.csv"
    return run(args.server, args.hours, args.interval, out)


if __name__ == "__main__":
    sys.exit(main())

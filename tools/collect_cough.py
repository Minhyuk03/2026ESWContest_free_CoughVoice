#!/usr/bin/env python3
"""
Cough-ID 기침 샘플 수집 스크립트 (라즈베리파이에서 실행)

I2S 마이크(hw:2, 2ch S32_LE 48kHz)에서 왼쪽 채널만 뽑아
32bit mono WAV로 저장하고 메타데이터 CSV에 자동 기록합니다.

사용 예:
    python3 collect_cough.py --speaker s01 --session 1 --distance 100 --type dry --count 20
    python3 collect_cough.py --speaker s01 --session 1 --distance 200 --type natural --count 10
    python3 collect_cough.py --speaker neg --session 1 --distance 100 --type speech --count 10

주요 옵션:
    --speaker   화자 ID (s01, s02, ... / 외부인은 x01 / 네거티브는 neg)
    --session   세션 번호. 반드시 다른 날 녹음할 것 (1, 2, ...)
    --distance  마이크까지 거리 cm (100 또는 200)
    --type      dry | natural | throat | speech | laugh | noise
    --count     녹음 횟수
    --device    ALSA 장치 (기본 hw:2)
    --outdir    저장 폴더 (기본 ~/cough_data)
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
import wave

try:
    import numpy as np
except ImportError:
    sys.exit("numpy 필요: sudo apt install -y python3-numpy")

RATE = 48000
FULL_SCALE = 2 ** 23          # 24bit
QUIET_THRESHOLD = 0.005       # 0.5% 미만이면 너무 조용
CLIP_THRESHOLD = 0.90         # 90% 초과면 클리핑 위험

VALID_TYPES = ["dry", "natural", "throat", "speech", "laugh", "noise"]


POP_TRIM_S = 0.1  # 장치 오픈 직후 팝 노이즈 회피용 트림(초)


def record_raw(device, seconds):
    """arecord로 raw S32_LE 스테레오를 받아 왼쪽 채널(24bit 정렬)을 반환.
    장치 오픈 직후 팝 노이즈를 피하려고 여유분을 더 녹음한 뒤 앞부분을 잘라낸다."""
    capture_s = int(seconds) + 1
    proc = subprocess.run(
        ["arecord", "-D", device, "-c", "2", "-r", str(RATE),
         "-f", "S32_LE", "-d", str(capture_s), "-t", "raw", "-q"],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"arecord 실패:\n{proc.stderr.decode(errors='replace')}")
    stereo = np.frombuffer(proc.stdout, dtype=np.int32).reshape(-1, 2)
    mono = stereo[:, 0] >> 8
    trim = int(RATE * POP_TRIM_S)
    target = int(RATE * seconds)
    return mono[trim:trim + target]   # 왼쪽 채널, left-justified 보정 + 팝 트림


def inspect(mono):
    peak = int(np.abs(mono).max())
    rms = float((mono.astype(np.float64) ** 2).mean() ** 0.5)
    ratio = peak / FULL_SCALE
    if ratio < QUIET_THRESHOLD:
        verdict = "TOO_QUIET"
    elif ratio > CLIP_THRESHOLD:
        verdict = "CLIPPED"
    else:
        verdict = "OK"
    return peak, rms, ratio, verdict


def save_wav(path, mono):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(4)
        w.setframerate(RATE)
        w.writeframes(mono.astype("<i4").tobytes())


def countdown(n=3):
    for i in range(n, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1)
    print("  녹음!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--session", required=True, type=int)
    ap.add_argument("--distance", required=True, type=int, help="cm")
    ap.add_argument("--type", required=True, choices=VALID_TYPES)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--device", default="hw:2")
    ap.add_argument("--outdir", default=os.path.expanduser("~/cough_data"))
    args = ap.parse_args()

    wav_dir = os.path.join(args.outdir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    meta_path = os.path.join(args.outdir, "metadata.csv")
    new_meta = not os.path.exists(meta_path)

    print("=" * 58)
    print(f"  화자 {args.speaker} | 세션 {args.session} | "
          f"{args.distance}cm | {args.type} | {args.count}회")
    print(f"  장치 {args.device} | {args.duration}초/클립")
    print("=" * 58)
    print("\n각 클립마다 카운트다운 후 녹음합니다.")
    print("'녹음!' 이 뜨면 한 번만 기침하세요. Ctrl+C 로 중단.\n")

    meta_file = open(meta_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(meta_file)
    if new_meta:
        writer.writerow(["filename", "speaker", "session", "distance_cm",
                         "type", "index", "peak", "rms", "peak_ratio",
                         "duration_s", "rate", "recorded_at"])

    saved = 0
    idx = 0
    try:
        while saved < args.count:
            idx += 1
            print(f"[{saved + 1}/{args.count}]", end="")
            countdown()

            mono = record_raw(args.device, args.duration)
            peak, rms, ratio, verdict = inspect(mono)

            if verdict == "TOO_QUIET":
                print(f"  → 너무 조용함 (peak {ratio*100:.2f}%). "
                      f"더 가까이서 다시 하세요. 저장 안 함.\n")
                continue
            if verdict == "CLIPPED":
                print(f"  → 클리핑 위험 (peak {ratio*100:.2f}%). "
                      f"조금 떨어져서 다시 하세요. 저장 안 함.\n")
                continue

            saved += 1
            fname = (f"{args.speaker}_ses{args.session:02d}"
                     f"_{args.distance}cm_{args.type}_{saved:03d}.wav")
            fpath = os.path.join(wav_dir, fname)
            save_wav(fpath, mono)

            writer.writerow([fname, args.speaker, args.session, args.distance,
                             args.type, saved, peak, round(rms, 1),
                             round(ratio, 5), args.duration, RATE,
                             dt.datetime.now().isoformat(timespec="seconds")])
            meta_file.flush()
            print(f"  → 저장 {fname}  (peak {ratio*100:.2f}%)\n")

    except KeyboardInterrupt:
        print("\n\n중단됨.")
    finally:
        meta_file.close()

    print("=" * 58)
    print(f"저장 완료: {saved}개  ({idx}회 시도)")
    print(f"  WAV : {wav_dir}")
    print(f"  META: {meta_path}")
    print("=" * 58)


if __name__ == "__main__":
    main()

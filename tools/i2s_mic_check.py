#!/usr/bin/env python3
"""
I2S MEMS 마이크(INMP441/MS3625) 진단 스크립트 - 라즈베리파이에서 실행

사용법:
    python3 i2s_mic_check.py            # 기본 hw:2
    python3 i2s_mic_check.py hw:0       # 카드 번호 지정

배선이 맞는지, 어느 채널에 신호가 들어오는지, 게인을 얼마로 잡아야 하는지 알려줍니다.
실행 중에 마이크에 대고 말하거나 손뼉을 치세요.
"""
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy 필요: sudo apt install python3-numpy")

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "hw:2"
RATE = 48000
SECONDS = 3
FULL_SCALE = 2 ** 23  # 24-bit


def record_raw():
    cmd = [
        "arecord", "-D", DEVICE,
        "-c", "2", "-r", str(RATE), "-f", "S32_LE",
        "-d", str(SECONDS), "-t", "raw",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        sys.exit(
            f"arecord 실패 (device={DEVICE})\n"
            f"{proc.stderr.decode(errors='replace')}\n"
            "→ 'arecord -l' 로 카드 번호를 확인하고 인자로 넘기세요."
        )
    return proc.stdout


def analyse(name, ch):
    peak = int(np.abs(ch).max())
    rms = float(np.sqrt((ch.astype(np.float64) ** 2).mean()))
    dc = float(ch.mean())
    unique = len(np.unique(ch[:2000]))

    print(f"\n[{name}]")
    print(f"  peak   = {peak:>12,}   ({peak / FULL_SCALE * 100:6.2f}% of full scale)")
    print(f"  rms    = {rms:>12,.0f}")
    print(f"  dc     = {dc:>12,.0f}")
    print(f"  unique = {unique:>12,}   (첫 2000샘플의 고유값 수)")

    if peak == 0:
        print("  판정: 완전 무신호. 이 채널에는 마이크가 없거나 SD/클럭 미연결.")
        return None
    if unique <= 2:
        print("  판정: 값이 고정됨. 클럭(SCK/WS)은 도는데 데이터가 안 옴 → SD 선 또는 L/R 핀 확인.")
        return None
    if peak < FULL_SCALE * 0.0005:
        print("  판정: 신호가 매우 약함. 배선은 됐을 수 있으나 음향 홀이 막혔는지 확인.")
    else:
        print("  판정: 정상 신호 감지됨.")

    headroom = FULL_SCALE * 0.7 / max(peak, 1)
    print(f"  권장 게인 ≈ {headroom:.1f}x  (피크를 full scale의 70%로 맞출 때)")
    return peak


def main():
    print(f"device={DEVICE}  rate={RATE}  {SECONDS}초 녹음")
    print("지금 마이크에 대고 소리를 내세요...")

    raw = record_raw()
    data = np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)

    # 24-bit 데이터가 32-bit 슬롯에 left-justified 되어 있으므로 8bit 시프트
    left = data[:, 0] >> 8
    right = data[:, 1] >> 8

    lp = analyse("LEFT  (L/R -> GND)", left)
    rp = analyse("RIGHT (L/R -> 3.3V)", right)

    print("\n" + "=" * 52)
    if lp and not rp:
        print("결과: 왼쪽 채널만 동작. 마이크 1개 구성이면 정상입니다.")
    elif rp and not lp:
        print("결과: 오른쪽 채널만 동작. L/R 핀이 3.3V에 물려 있습니다.")
    elif lp and rp:
        print("결과: 양쪽 다 신호 있음. 마이크 2개 구성이면 정상입니다.")
    else:
        print("결과: 신호 없음. 아래 순서로 확인하세요.")
        print("  1. 'arecord -l' 에 카드가 잡히는가 → 아니면 config.txt 오버레이 문제")
        print("  2. VDD가 3.3V(pin 1/17)에 연결됐는가")
        print("  3. SCK=pin12, WS=pin35, SD=pin38 인가")
        print("  4. L/R 핀이 GND에 고정됐는가 (floating 금지)")
        print("  5. 마이크 음향 홀이 기판에 눌려 막히지 않았는가")


if __name__ == "__main__":
    main()

"""엣지 메인 — 캡처 → 검출 → 전송 파이프라인 조립.

사용 예:
  마이크:      python main.py --server http://192.168.0.10:8000
  I2S 마이크:  python main.py --server http://... --gain 5
  파일 테스트: python main.py --server http://localhost:8000 --wav test_cough.wav
"""
from __future__ import annotations

import argparse
import time

from audio_capture import AudioCapture
from cough_detector import CoughDetector
from event_sender import EventSender


def main() -> None:
    p = argparse.ArgumentParser(description="Cough-ID edge")
    p.add_argument("--server", required=True, help="서버 URL (http://IP:8000)")
    p.add_argument("--wav", help="WAV 파일 시뮬레이션 모드 (마이크 없이 테스트)")
    p.add_argument(
        "--device", default=None,
        help="sounddevice 장치 index/이름. I2S MEMS 마이크는 'plughw:CARD=sndrpigooglevoi' 권장"
        " (USB 오디오 연결 시 카드 번호가 바뀌므로 hw:N 대신 이름 사용)",
    )
    p.add_argument("--gain", type=float, default=5.0, help="입력 게인 (I2S MEMS 실측 잠정치 ~5x, 거리별로 재측정 필요)")
    p.add_argument("--threshold", type=float, default=0.08, help="RMS 검출 임계치")
    p.add_argument("--device-id", default="rpi5-01")
    p.add_argument("--heartbeat", type=float, default=60.0,
                   help="생존 신호 간격(초). 0이면 보내지 않는다")
    args = p.parse_args()

    capture = AudioCapture(
        source="file" if args.wav else "mic",
        wav_path=args.wav,
        device=int(args.device) if args.device and args.device.isdigit() else args.device,
        gain=args.gain,
    )
    detector = CoughDetector(capture, rms_threshold=args.threshold)
    sender = EventSender(args.server, device_id=args.device_id,
                         heartbeat_interval=args.heartbeat)

    detector.on_cough = lambda wav, peak: (
        print(f"[main] 기침 검출! peak_rms={peak:.3f} → 전송", flush=True),
        sender.send(wav, peak),
    )

    capture.start()
    print(f"[main] 시작 — source={'file' if args.wav else 'mic'}, "
          f"threshold={args.threshold}, server={args.server}", flush=True)
    try:
        if args.wav:
            time.sleep(1)
            while capture._thread and capture._thread.is_alive():
                time.sleep(0.5)
            time.sleep(3)  # 파일 종료 후 전송 마무리 대기
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        sender.stop()
        print("[main] 종료", flush=True)


if __name__ == "__main__":
    main()

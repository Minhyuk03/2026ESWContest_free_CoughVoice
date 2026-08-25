"""엣지 메인 — 캡처 → 검출 → 전송 파이프라인 조립.

사용 예:
  마이크:      python main.py --server http://192.168.0.10:8000
  I2S 마이크:  python main.py --server http://... --gain 5
  파일 테스트: python main.py --server http://localhost:8000 --wav test_cough.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from audio_capture import AudioCapture
from cough_detector import CoughDetector
from event_sender import EventSender

# 마이크 모드에서 이 시간 동안 오디오 청크가 하나도 없으면 캡처가 멎은 것으로 보고
# 프로세스를 종료한다(systemd Restart=always가 재기동). 정상은 100ms마다 들어온다.
# 스트림 열림·첫 콜백까지의 여유로 시작 직후 잠깐은 봐준다.
AUDIO_STALL_LIMIT_S = 15.0


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
    p.add_argument("--device-token", default=os.environ.get("COUGHID_DEVICE_TOKEN", ""),
                   help="서버 COUGHID_DEVICE_TOKEN과 같은 값. 서버가 요구하면 필요하다 "
                        "(기본값은 환경변수에서 읽는다)")
    args = p.parse_args()

    capture = AudioCapture(
        source="file" if args.wav else "mic",
        wav_path=args.wav,
        device=int(args.device) if args.device and args.device.isdigit() else args.device,
        gain=args.gain,
    )
    detector = CoughDetector(capture, rms_threshold=args.threshold)
    sender = EventSender(args.server, device_id=args.device_id,
                         heartbeat_interval=args.heartbeat,
                         device_token=args.device_token)

    detector.on_cough = lambda wav, peak: (
        print(f"[main] 기침 검출! peak_rms={peak:.3f} → 전송", flush=True),
        sender.send(wav, peak),
    )

    capture.start()
    print(f"[main] 시작 — source={'file' if args.wav else 'mic'}, "
          f"threshold={args.threshold}, server={args.server}", flush=True)
    stalled = False
    try:
        if args.wav:
            time.sleep(1)
            while capture._thread and capture._thread.is_alive():
                time.sleep(0.5)
            time.sleep(3)  # 파일 종료 후 전송 마무리 대기
        else:
            # 마이크 모드 워치독. 캡처는 데몬 스레드라 스트림 열기 실패·콜백 예외로
            # 조용히 죽어도 이 루프는 계속 돌고, 하트비트는 별도 스레드라 서버엔
            # online으로 보인다. 그러면 마이크가 죽었는데도 기침이 영영 0건이 된다.
            # (1) 스레드 종료, (2) 스트림은 살아 있으나 오디오가 끊긴 정지 둘 다 잡는다.
            while True:
                time.sleep(1)
                if capture._thread is None or not capture._thread.is_alive():
                    print("[main] 캡처 스레드 종료 감지 — 재기동을 위해 프로세스를 내린다",
                          flush=True)
                    stalled = True
                    break
                since = capture.seconds_since_audio()
                if since is not None and since > AUDIO_STALL_LIMIT_S:
                    print(f"[main] {since:.0f}s간 오디오 없음 — 마이크 정지로 보고 프로세스를 "
                          "내린다(systemd가 재기동)", flush=True)
                    stalled = True
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        sender.stop()
        print("[main] 종료", flush=True)
    if stalled:
        sys.exit(1)   # 비정상 종료로 알려 systemd가 재기동하게 한다


if __name__ == "__main__":
    main()

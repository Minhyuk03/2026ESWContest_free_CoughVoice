# monitor — 실시간 소리 모니터

마이크 입력을 실시간으로 분석해 **기침 / 대화 / 기타 소음**을 웹 대시보드에서 바로 확인하는 도구.
Python 서버가 마이크를 읽고 WebSocket으로 브라우저에 밀어준다. Mac과 RPi5 둘 다에서 동작한다.

`edge/`·`server/`와는 독립적으로 돌아가는 **검출 임계값 튜닝 · 시연용** 도구다.
여기서 찾은 임계값을 `edge/cough_detector.py`에 반영하는 흐름을 상정한다.

## 설치

```bash
cd monitor
pip install -r requirements.txt
```

RPi에서 I2S 마이크만 쓸 거면 `sounddevice`는 빼도 된다 (`arecord`만 있으면 됨).

## 실행

```bash
# 1) 마이크 없이 데모 — 합성 기침/음성이 번갈아 재생됨
python server.py --source sim

# 2) Mac 내장 / USB 마이크
python server.py --list-devices          # 장치 번호 확인
python server.py --source sd --device 1

# 3) RPi5 + MS3625 I2S 마이크
python server.py --source arecord --device hw:2 --gain 5
```

브라우저에서 `http://localhost:8000` 접속. RPi에서 띄웠다면 Mac에서 `http://mh.local:8000`.

## 화면 구성

| 영역 | 내용 |
|---|---|
| 판정 카드 | 방금 감지한 소리의 라벨(기침/대화/기타)과 기침 점수 |
| 입력 레벨 | 실시간 dB, 감지 임계값 위치(흰 선), 센트로이드·플랫니스·ZCR |
| 파형 | 최근 12초 레벨 곡선, 파란 음영 = 이벤트로 잡힌 구간 |
| 스펙트로그램 | 60Hz–7.8kHz, 48밴드 로그 스케일 |
| 누적 감지 | 기침/대화/기타 카운터 |
| 이벤트 근거 | 점수를 구성한 6개 특징의 기여도 막대 |
| 임계값 조정 | 판정 점수·감지 민감도·종료 유예를 실시간 변경 |

## 판별 원리 (규칙 기반)

32ms 창 / 16ms 홉으로 프레임 특징을 뽑고, 적응형 노이즈 플로어 + 히스테리시스 상태머신으로
소리 이벤트를 잘라낸 뒤 이벤트 단위로 점수를 매긴다.

| 특징 | 가중치 | 기침 | 대화 |
|---|---|---|---|
| 지속시간 | 0.18 | 0.15–0.7s | 보통 1s 이상 |
| onset 속도 | 0.17 | 피크까지 20ms 이내 (폭발음) | 완만 |
| 에너지 형태 | 0.13 | 앞쪽 피크 + 빠른 감쇠, 크레스트↑ | 평탄 |
| 주파수 중심 | 0.20 | 2.6kHz 이상 (광대역) | 300–800Hz |
| 광대역성(플랫니스) | 0.18 | 0.2 이상 (무성 잡음) | 0.03 이하 (하모닉) |
| 단일 봉우리 | 0.14 | 1개 | 음절 변조로 여러 개 |

가중합 ≥ `cough_threshold`(기본 0.55) → **기침**.
아니면서 지속시간이 길거나 음절 변조가 있으면 → **대화**, 나머지는 **기타 소음**.

첫 0.65초는 배경 소음 측정용 워밍업 구간이라 이벤트를 잡지 않는다.
배경이 계속 임계값 위로 올라오면 자동으로 노이즈 플로어를 재보정한다.

## 검증

```bash
python test_classifier.py
```

합성 기침 12개 / 합성 음성 12개에 대해 **24/24 (100%)**.
합성 신호는 실제보다 분리가 쉬우므로, 실제 마이크에서는 대시보드의 슬라이더로
`감지 민감도`와 `판정 점수`를 조정하면서 튜닝하는 것을 권장한다.

## 파일

```
monitor/
├── audio_source.py     입력 소스 (sounddevice / arecord I2S / 시뮬레이션) + 48k→16k 데시메이터
├── classifier.py       프레임 특징 추출, 이벤트 상태머신, 규칙 기반 스코어링
├── server.py           FastAPI + WebSocket 브로드캐스트
├── static/index.html   대시보드 UI (단일 파일, 의존성 없음)
└── test_classifier.py  합성 신호 검증
```

## 나중에 모델로 교체할 때

`classifier.py`의 `StreamAnalyzer._finish_event()`가 이벤트 특징을 모아 라벨/점수를 내는
유일한 지점이다. 여기서 이벤트 구간 오디오를 ECAPA-TDNN이나 YAMNet에 넘기고
같은 형태의 dict(`label`, `score`, `parts`)를 반환하면 서버·UI는 그대로 쓸 수 있다.

## RPi 참고 (기존 세팅)

- 배선: VDD→pin1(3.3V), GND→pin6, SCK→pin12, WS→pin35, SD→pin38, L/R→GND
- overlay `googlevoicehat-soundcard`, 장치는 **card 2 = hw:2**
- 캡처 2ch S32_LE 48kHz, 24bit left-justified → `>> 8` 시프트 (코드에 반영됨)
- L/R을 GND에 물렸으므로 **LEFT 채널만 유효** → `--channel left` (기본값)
- 실측 게인 약 5배 → `--gain 5` (기본값)

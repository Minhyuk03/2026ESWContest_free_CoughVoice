# 하드웨어 — I2S MEMS 마이크 (RPi5)

최종 검증: 2026-08-05
대상: MS3625 I2S MEMS 마이크 (INMP441 호환) × 1, Raspberry Pi 5

---

## 배선표

| RPi5 핀 | 마이크 핀 | 비고 |
|---|---|---|
| pin 1 (3.3V) | VDD | **5V 절대 금지.** pin 2/4는 5V이며 연결 시 즉시 손상 |
| pin 6 (GND) | GND | |
| pin 12 (GPIO18) | SCK (BCLK) | |
| pin 35 (GPIO19) | WS (LRCLK) | |
| pin 38 (GPIO20) | SD (DOUT) | |
| — | L/R | **GND에 로컬 점프.** 띄워두면(floating) 채널 불안정으로 신호가 잡히지 않음 |

pin 1 위치: USB-C 전원 포트에 가장 가까운 헤더 끝, 보드 안쪽(중앙에 가까운) 열의 첫 번째. 바깥쪽 열 첫 번째가 pin 2(5V)이므로 혼동 주의.

## 마이크 모듈 핀아웃

```
윗줄:   SD   VDD  GND
아랫줄:  L/R  WS   SCK
```

## 마이크 2개로 확장할 때

`SCK` / `WS` / `SD`를 그대로 병렬 연결하고, 두 번째 마이크의 `L/R`만 **3.3V**에 물리면 오른쪽 채널로 잡힌다.

RPi5는 I2S 버스가 하나뿐이고 INMP441 계열은 TDM을 지원하지 않으므로 **최대 2개**. 3개 이상이 필요하면 ReSpeaker USB Mic Array 같은 별도 보드가 필요하다.

## 전원

마이크는 라즈베리파이 3.3V 레일에서 직접 공급받는다. 별도 전원 불필요 (개당 약 1.4mA). 오히려 별도 전원을 쓰면 그라운드 기준이 갈려 I2S 클럭 타이밍이 흔들릴 수 있다.

---

## OS 설정

`/boot/firmware/config.txt`

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

`dtparam=audio=on` 은 주석 처리. 저장 후 재부팅.

확인:

```bash
arecord -l
# card 2: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: ...
```

> 마이크를 연결하지 않아도 카드는 잡혀야 정상이다. 카드가 안 보이면 배선이 아니라 오버레이 문제다.

---

## 캡처 파라미터

| 항목 | 값 |
|---|---|
| ALSA 장치 | **`hw:2`** — card 0 아님. USB 오디오 연결 시 번호가 바뀔 수 있어 `plughw:CARD=sndrpigooglevoi` 권장 |
| 포맷 | `S32_LE`, 2ch, 48000 Hz |
| 유효 채널 | **LEFT만 사용한다.** RIGHT에도 같은 소리가 실린다 — 아래 2026-08-28 재측정 참조 |
| 비트 정렬 | 24bit가 32bit 슬롯에 left-justified → **`>> 8` 시프트 필수** |
| 실측 게인 | 약 5x (초기 추정 20~30보다 훨씬 낮음) — **잠정치, 실사용 거리에서 재측정 필요** |

시프트를 빠뜨리면 값이 256배로 보여 클리핑처럼 오인하기 쉽다. 반대로 원시 진폭이 매우 작게 보이는 것은 정상이므로 배선 문제로 오판하지 말 것.

### 최소 캡처 예시

```python
import subprocess
import numpy as np

raw = subprocess.run(
    ["arecord", "-D", "hw:2", "-c", "2", "-r", "48000",
     "-f", "S32_LE", "-d", "3", "-t", "raw"],
    capture_output=True).stdout

stereo = np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)
mono = stereo[:, 0] >> 8      # LEFT 채널, 24bit 정렬 보정
```

---

## 검증

`tools/i2s_mic_check.py` 실행.

```bash
python3 tools/i2s_mic_check.py hw:2
```

정상 출력 예 (2026-08-05 실측):

```
LEFT : peak=1,121,519 (13.37%)  rms=298,968  uniq=1452   -> 정상 신호
RIGHT: peak=0 (0.00%)  rms=0  uniq=1                     -> 마이크 1개 구성
```

### ⚠ 2026-08-28 재측정 — **RIGHT는 더 이상 0이 아니다**

거실 69.2분 연속 녹음(`arecord -D hw:2 -f S32_LE -r 48000 -c 2`)을 분석한 결과, RIGHT 채널이
LEFT와 사실상 같은 음향 신호를 싣고 있었다.

| 구간 | corr(L, R) | R/L rms 비 |
|---|---|---|
| 큰 소리 (HP80 후 RMS 상위) | 0.76 | 1.05 |
| 조용한 구간 | 0.42 | 0.98 |

위 2026-08-05 출력(`RIGHT: peak=0, uniq=1`)은 재현되지 않는다. 언제·왜 바뀌었는지는 확인하지
못했다 — 배선 변경 가능성과 측정 경로 차이(당시 `i2s_mic_check.py` vs 이번 `arecord` 원시 스트림)를
모두 배제하지 못했다.

**파이프라인 영향은 없다.** `audio_capture._run_mic`과 모든 평가 도구가 `indata[:, 0]`(LEFT)만
쓴다. 다만 **"RIGHT가 0인지"로 배선 정상 여부를 판정하면 안 된다** — 그 진단은 더 이상 유효하지 않다.

### 증상별 원인

| 증상 | 원인 |
|---|---|
| `arecord -l`에 카드 없음 | `config.txt` 오버레이 미설정 |
| 양 채널 peak=0 | `SD` 선 또는 클럭(`SCK`/`WS`) 미연결 |
| 값이 고정 (uniq ≤ 2) | 클럭은 도는데 데이터 없음 → `SD` 또는 `L/R` 확인 |
| 신호가 매우 약함 | 음향 홀이 기판에 눌려 막힘 |
| 마이크가 뜨거움 | **즉시 전원 차단.** VDD가 5V에 연결됨 |

---

## 접속 정보

```bash
ssh mh@mh.local
```

- 호스트명은 `raspberrypi`가 아니라 **`mh`**
- OS: Debian trixie / 커널 6.18.34 / Python 3.13 / 사용자 `mh`
- 개발 중 아이폰 핫스팟(`172.20.10.x`) 사용 시 Mac도 같은 핫스팟에 연결되어야 함

---

## 주의

- 헤더 핀 번호는 육안 확인이 아니라 **동작으로 검증**된 상태다. 재배선 시 위 배선표를 그대로 따를 것
- 만능기판은 구멍끼리 연결되어 있지 않다. 패드 간 연결은 직접 납땜해야 한다
- 듀폰 하우징을 패드에 얹기만 하면 접촉 불량이 발생한다. I2S는 MHz 단위 클럭이라 순간적인 접촉 불량에도 데이터가 깨진다. 반드시 납땜할 것
- I2S 배선은 15cm 이내로 짧게, `SCK`/`WS`는 GND와 꼬아서 노이즈를 줄인다

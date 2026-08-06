# Cough ID — 기침 소리 기반 화자 식별 시스템

> 제 24회 임베디드 소프트웨어 경진대회 출품작
> 기숙사·요양시설 등 다인 거주 환경에서 **기침 소리만으로 누가 기침했는지 식별**하고, 이상 징후(빈발 기침)를 관리자·보호자에게 실시간 알리는 시스템. 무조작·비착용 — 거주자는 아무것도 하지 않아도 된다.

## 1. 시스템 구성

```
[거주자 기침] → 엣지(라즈베리파이 5) → 서버(FastAPI) → 웹 대시보드(React)
                기침 검출·구간 절단      화자 식별·알림 평가    실시간 모니터링
```

| 계층 | 역할 | 기술 |
|------|------|------|
| 엣지 (`edge/`) | 오디오 상시 캡처 → 기침 검출(에너지+CNN) → 2~3초 절단 → 서버 전송(실패 시 재시도 큐) | Python, sounddevice, tflite |
| 서버 (`server/`) | 이벤트 수신 → log-mel 특징 추출 → ECAPA-TDNN 임베딩 → 코사인 매칭 → 알림 규칙 평가 → WebSocket 푸시 | FastAPI, SQLAlchemy, SpeechBrain |
| 대시보드 (`dashboard/`) | 실시간 피드, 이력 조회, 화자 등록, 알림 규칙 설정 (7개 화면) | React + Vite, Recharts |
| 모니터 (`monitor/`) | 마이크 입력을 실시간 분석해 **기침/대화/기타**를 화면에서 즉시 확인 — 검출 임계값 튜닝·시연용 독립 도구 | Python, FastAPI, WebSocket |

- 성능 목표: 기침 발생 → 대시보드 표시 **≤ 3초** (NFR-03)
- 프라이버시: 실명 대신 alias, 임베딩만 저장·원본 음성 비보존 (NFR-06)
- 설계 문서: [`docs/`](docs/) — 개발 생명주기 7단계, UML 4종(Use Case·Activity·Class·Sequence), 스토리보드·와이어프레임

## 2. 저장소 구조

```
cough-id/
├── edge/                  # 라즈베리파이 5 상주 프로세스
│   ├── audio_capture.py   #   마이크 스트림 → 링버퍼
│   ├── cough_detector.py  #   기침 검출 + 구간 절단
│   ├── event_sender.py    #   서버 전송 + 재시도 큐
│   └── main.py
├── server/
│   └── app/
│       ├── api/           #   라우터 (events, persons, alerts, auth)
│       ├── ml/            #   특징 추출·화자 식별
│       ├── core/          #   알림 엔진·WebSocket
│       ├── models.py      #   Person / CoughEvent / Alert
│       ├── db.py
│       └── main.py
├── dashboard/             # React + Vite 웹 대시보드
├── monitor/               # 실시간 소리 모니터 (기침/대화 판별 튜닝·데모용 독립 도구)
│   ├── audio_source.py    #   sounddevice / arecord I2S / 시뮬레이션 입력
│   ├── classifier.py      #   규칙 기반 기침·대화 판별
│   ├── server.py          #   FastAPI + WebSocket
│   └── static/index.html  #   단일 파일 대시보드 UI
├── tools/                 # 개발·수집 보조 스크립트 (라즈베리파이에서 실행)
│   ├── i2s_mic_check.py   #   I2S 마이크 배선 진단
│   └── collect_cough.py   #   기침 샘플 수집 + 자동 라벨링
└── docs/                  # 설계 문서 (UML, 스토리보드 등)
```

### 하드웨어 (2026-08-05 검증 완료)

I2S MEMS 마이크(MS3625) 1개를 라즈베리파이 5에 연결해 실제 오디오 캡처를 검증했다.
배선표·오버레이 설정·캡처 파라미터는 [`docs/hardware_i2s_mic.md`](docs/hardware_i2s_mic.md) 참조.

핵심 값만 요약하면:

| 항목 | 값 |
|---|---|
| ALSA 장치 | `hw:2` (card 0 아님) |
| 포맷 | `S32_LE` / 2ch / 48000 Hz |
| 유효 채널 | LEFT만 (RIGHT는 항상 0) |
| 비트 정렬 | 24bit가 32bit 슬롯에 left-justified → **`>> 8` 시프트 필수** |

### 데이터 수집

기침 샘플 수집 프로토콜(거리·횟수·라벨링 규칙·품질 기준)은 [`docs/data_collection_protocol.md`](docs/data_collection_protocol.md) 참조.

> **등록용과 검증용은 반드시 다른 날에 녹음할 것.** 한 세션에서 몰아 찍고 그 안에서 학습/검증을 나누면 모델이 화자의 목소리가 아니라 그날의 녹음 환경을 외운다. 측정 정확도가 부풀려지고 데모에서 무너진다.

수집된 음성(`cough_data/`, `*.wav`)은 개인 식별 정보이므로 `.gitignore`로 커밋을 차단해 두었다.

## 3. 개발 과정 (P1~P8)

서버부터 만들고, 엣지는 PC 마이크로 먼저 개발한 뒤 라즈베리파이에 포팅한다. Class 다이어그램의 클래스 1개 = 파일 1개로 설계-코드 추적성을 유지한다.

| 단계 | 내용 | 검증 기준 | 상태 |
|------|------|-----------|:----:|
| **P1** | 프로젝트 골격 + Git/GitHub 구성 | 3개 모듈 폴더 구조, 리포 푸시 | ✅ |
| **P2** | DB 스키마 + 서버 API 뼈대 (식별은 스텁) | Swagger에서 전 엔드포인트 동작 | 🔄 |
| **P3** | 화자 식별 ML 파이프라인 (임베딩+코사인 매칭) | 팀원 4~5인 오프라인 식별 정확도 달성 | ⬜ |
| **P4** | 엣지 기침 검출·전송 (PC → RPi 포팅) | 실기침 → DB 도착, 네트워크 단절 복구 | ⬜ |
| **P5** | 알림 엔진 + WebSocket 실시간 피드 | 1시간 N회 규칙 → 웹훅 발송 | ⬜ |
| **P6** | React 대시보드 7화면 (S0~S4·M1) | 와이어프레임 1:1 대조 | ⬜ |
| **P7** | 통합 테스트 (E2E·지연·장애 시나리오) | TC-01~05, 지연 ≤ 3초 | ⬜ |
| **P8** | 데모 시나리오 리허설 + 발표 준비 | Sequence 다이어그램 대본 시연 | ⬜ |

### 주요 API (P2)

| 엔드포인트 | 기능 |
|-----------|------|
| `POST /events` | 엣지에서 기침 wav + 메타데이터 수신 |
| `GET /events` | 이력 조회 (기간·화자·미등록 필터) |
| `GET/POST/DELETE /persons` | 화자 관리·등록 |
| `GET/PUT /alerts, /alert-rules` | 알림 이력·규칙 |
| `POST /auth/login` | JWT 인증 (admin / guardian 역할) |
| `WS /ws/feed` | 실시간 이벤트 푸시 |

## 4. 실행 방법

### 서버

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

### 대시보드

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:5173
```

### 엣지 (개발: PC 마이크 / 운영: 라즈베리파이 5)

```bash
cd edge
pip install -r requirements.txt
python main.py --server http://<서버주소>:8000
```

RPi 배포 시 systemd 서비스로 등록해 부팅 시 자동 시작.

## 5. 협업 규칙

- 브랜치: `main` 보호(PR 필수) · `feat/edge-*` · `feat/server-*` · `feat/dash-*`
- 커밋 메시지: `P<단계>: 내용` (예: `P2: events 라우터 구현`)
- 진행 상황·산출물은 노션 "제 24회 임베디드 소프트웨어 경진대회" 페이지에 공유

## 6. 팀

| 역할 | 담당 |
|------|------|
| 서버·API | - |
| ML·엣지 | - |
| 프론트엔드 | - |

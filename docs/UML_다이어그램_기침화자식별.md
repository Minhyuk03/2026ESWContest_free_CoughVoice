# UML 다이어그램 4종 — 기침 소리 기반 화자 식별 시스템

> 작성일: 2026-07-02 · 연계: 개발 생명주기 7단계(3.6절), FR-01~10
> 노션 "제 24회 임베디드 소프트웨어 경진대회" 페이지에 동일 내용 공유됨.

## 1. Use Case 다이어그램

```mermaid
flowchart LR
  R["거주자<br>(등록/미등록 화자)"]
  M["관리자<br>(사감·요양보호사)"]
  G["보호자"]
  subgraph S["기침 화자 식별 시스템"]
    UC1(["기침 이벤트 감지"])
    UC2(["기침 화자 식별"])
    UC3(["미등록 화자 거절"])
    UC4(["이상 징후 알림"])
    UC5(["대시보드 모니터링"])
    UC6(["기침 이력 조회"])
    UC7(["신규 화자 등록"])
  end
  R --- UC1
  UC1 -.->|include| UC2
  UC3 -.->|extend| UC2
  UC4 -.->|extend| UC2
  UC4 --- G
  M --- UC5
  M --- UC6
  M --- UC7
  G --- UC6
```

- 거주자: 기침으로 시스템을 트리거하는 수동적 액터(무조작·비착용).
- 관리자: 대시보드(FR-08)·이력 조회(FR-06)·화자 등록(FR-09). 보호자: 알림 수신(FR-07)·이력 조회.
- 감지 «include» 식별(항상 실행), 미등록 거절·이상 징후 알림은 «extend»(조건부).

## 2. Activity 다이어그램

```mermaid
flowchart TD
  ST((시작)) --> A1
  subgraph EDGE["엣지 — 라즈베리파이 5"]
    A1[오디오 상시 캡처] --> A2["기침 후보 검출<br>에너지 + CNN 2차 필터"]
    A2 --> D1{기침인가?}
    D1 -->|아니오| A1
    D1 -->|예| A3[구간 절단 2~3초]
    A3 --> D2{네트워크 정상?}
    D2 -->|아니오| A4[재시도 큐 보관]
    A4 -.복구 시 재전송.-> D2
  end
  subgraph SRV["서버 — FastAPI"]
    A5[이벤트 수신·저장] --> A6[특징 추출 log-mel]
    A6 --> A7[임베딩·유사도 매칭]
    A7 --> D3{유사도 ≥ 임계치?}
    D3 -->|예| A8[화자 확정]
    D3 -->|아니오| A9[미등록 처리]
    A8 --> A10[이벤트 DB 기록]
    A9 --> A10
    A10 --> D4{알림 규칙 충족?}
    D4 -->|아니오| A11[대시보드 갱신]
  end
  subgraph USER["관리자·보호자"]
    B1[이상 징후 알림 수신]
    B2[실시간 현황 확인]
  end
  D2 -->|예| A5
  D4 -->|예| B1
  A11 -.-> B2
  A11 --> E((종료))
```

분기 4개가 핵심: 기침 판정(TC-03), 네트워크 재시도(TC-05), 임계치 기반 식별/거절(FR-04·05, 양쪽 모두 DB 기록), 알림 규칙(FR-07). 엣지는 검출·전송까지만, ML은 서버 — 엣지-서버 분담 구조.

## 3. Class 다이어그램

```mermaid
classDiagram
  class AudioCapture {
    -ring_buffer
    +start()
    +read()
  }
  class CoughDetector {
    -threshold
    -cnn_filter
    +detect()
    +cut_segment()
  }
  class EventSender {
    -retry_queue
    -server_url
    +send()
    +flush()
  }
  class IngestAPI {
    +post_event()
  }
  class FeatureExtractor {
    +to_logmel()
  }
  class SpeakerIdentifier {
    -threshold
    -model
    +embed()
    +match()
  }
  class AlertEngine {
    -rules
    +evaluate()
  }
  class Dashboard {
    +render_stats()
  }
  class Person {
    +id
    +alias
    +embedding_ref
  }
  class CoughEvent {
    +id
    +person_id
    +ts
    +confidence
    +doa
    +is_unknown
  }
  class Alert {
    +id
    +rule
    +count
    +window
    +sent_at
  }
  AudioCapture --> CoughDetector : PCM 프레임
  CoughDetector --> EventSender : 기침 wav
  EventSender ..> IngestAPI : HTTP POST /events
  IngestAPI ..> FeatureExtractor : use
  IngestAPI ..> SpeakerIdentifier : use
  IngestAPI ..> AlertEngine : use
  SpeakerIdentifier ..> CoughEvent : create
  AlertEngine ..> Alert : create
  Dashboard ..> CoughEvent : read
  Person "1" -- "0..*" CoughEvent
  Person "1" -- "0..*" Alert
```

생명주기 3.3·3.4 기반. 엣지는 단방향 파이프라인(단위 테스트 용이), 서버는 IngestAPI 진입점 + «use» 호출(모델 교체 시 무수정), Person은 alias·embedding_ref만 저장(NFR-06 익명화), is_unknown으로 미등록 기침도 기록.

## 4. Sequence 다이어그램

```mermaid
sequenceDiagram
  actor R as 거주자
  participant E as 엣지 (RPi 5)
  participant API as 서버 API
  participant ML as 식별 모듈
  participant DB as DB
  actor U as 보호자·관리자
  R->>E: 기침 소리
  E->>E: 기침 검출·구간 절단 (2~3초)
  E->>API: POST /events (wav, 타임스탬프)
  API->>ML: 특징 추출·임베딩 요청
  ML->>ML: 코사인 유사도 매칭
  alt 유사도 ≥ 임계치 (등록 화자)
    ML-->>API: 화자 A (conf 0.92)
    API->>DB: 이벤트 저장 (person=A)
    API->>API: 알림 규칙 평가
    opt 임계치 초과 (예: 1시간 10회)
      API->>U: 이상 징후 알림 (웹훅)
    end
  else 유사도 < 임계치 (미등록)
    ML-->>API: unknown
    API->>DB: 이벤트 저장 (unknown)
  end
  API->>U: 대시보드 실시간 갱신
  Note over R,U: NFR-03 — 기침 발생 → 결과 표시 ≤ 3초
```

TC-01·02 통합 시나리오. alt = 등록/미등록 상호 배타 분기, opt = 알림 규칙 충족 시에만(TC-04). 전 과정 지연 목표 NFR-03 ≤ 3초. 발표 데모 대본으로 사용 가능.

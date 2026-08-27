"""SQLAlchemy 모델 — Class 다이어그램의 Person·CoughEvent·Alert 대응."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary,
                        String, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alias: Mapped[str] = mapped_column(String(50), unique=True)  # 실명 대신 alias (NFR-06)
    room: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # 호실 표기 (예: "301호")
    embedding_ref: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)  # 임베딩 평균
    sample_count: Mapped[int] = mapped_column(Integer, default=0)  # 등록 시 녹음한 샘플 수
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    events: Mapped[List["CoughEvent"]] = relationship(back_populates="person")


class CoughEvent(Base):
    __tablename__ = "cough_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 엣지가 부여하는 고유 ID. 재전송 큐가 같은 클립을 다시 보내도 중복 저장되지 않는다.
    # 구버전 엣지는 보내지 않으므로 nullable.
    event_id: Mapped[Optional[str]] = mapped_column(String(40), nullable=True, index=True)
    device_id: Mapped[str] = mapped_column(String(50))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    person_id: Mapped[Optional[int]] = mapped_column(ForeignKey("persons.id"), nullable=True)  # None = unknown
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # 화자를 누가 정했는가. 사람이 손으로 지정하면 similarity는 **그 이전 모델 점수**라
    # 지금 라벨과 아무 관계가 없다. 구분해 두지 않으면 화면이 "s02 · 93%"처럼
    # 있지도 않은 확신을 말하게 된다(2026-08-28 실제로 그렇게 보였다).
    person_source: Mapped[str] = mapped_column(String(10), default="model")  # model | manual
    peak_rms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_path: Mapped[str] = mapped_column(String(255))  # 저장된 wav 경로
    # 등록(enroll-from-events)에 쓰인 이벤트. 보존 정책(NFR-06, 기본 7일)이 원음을
    # 지울 때 이 이벤트의 wav는 예외로 남긴다 — 재등록·등록 구성 확인에 필요하기 때문.
    enrolled: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- 음향 지표 (P6) ---
    # 원음은 사생활 문제로 오래 두지 않으므로, 나중에 다시 계산할 수 없는 값만 남긴다.
    # 원음 자체는 core/retention.py가 기본 7일 후 자동 삭제한다(NFR-06). 아래 특징량은
    # 삭제 후에도 남아 지표·이력이 유지된다.
    # 시간 기반 지표(시간당 횟수·발작 수 등)는 captured_at에서 언제든 다시 뽑을 수 있어
    # 컬럼으로 두지 않는다 — core/cough_metrics.py 참조.
    cough_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # PANNs(CNN14)는 한 번의 forward에서 AudioSet 527클래스를 모두 계산한다.
    # 지금까지 Cough(47)·Throat clearing(48)만 읽고 나머지를 버렸는데,
    # Wheeze(42)·Gasp(44)는 참고자료가 기록을 권한 지표라 함께 저장한다.
    # **미검증 지표다.** 우리 데이터로 정확도를 재본 적이 없으므로 판정에 쓰지 않고
    # 기록·전시 용도로만 둔다. 특히 AudioSet "Whoop"(10)은 환호성이지 백일해의
    # 흡기성 whoop이 아니므로 쓰지 않는다.
    wheeze_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gasp_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    person: Mapped[Optional[Person]] = relationship(back_populates="events")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[Optional[int]] = mapped_column(ForeignKey("persons.id"), nullable=True)
    rule: Mapped[str] = mapped_column(String(100))       # 예: "1h>=10"
    message: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # 화면 강조·정렬용. 질환 중증도가 아니라 "얼마나 급히 확인할 일인가"다.
    severity: Mapped[str] = mapped_column(String(10), default="info")
    # 경계값의 임상 출처. 근거 있는 값과 탐색용 값을 화면에서 구분하기 위해 남긴다.
    source: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    # --- 후속 조치 상태 ---
    # 알림을 띄우기만 하면 "누가 봤고 무엇을 했는가"가 어디에도 남지 않는다. 교대 근무가
    # 있는 현장에서는 같은 알림을 두 사람이 각자 확인하거나, 아무도 확인하지 않은 채
    # 목록에서 밀려난다. 상태·담당자·확인 시각·메모를 알림 자체에 붙여 둔다.
    STATUS_OPEN = "open"    # 미확인
    STATUS_ACK = "ack"      # 확인함 (사람이 보았다)
    STATUS_DONE = "done"    # 조치 완료
    STATUSES = (STATUS_OPEN, STATUS_ACK, STATUS_DONE)

    status: Mapped[str] = mapped_column(String(10), default=STATUS_OPEN)
    # 확인·조치한 관리자 계정. 상태가 미확인으로 되돌아가면 함께 비운다.
    assignee: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    acked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)


class AlertRule(Base):
    """S4 알림 규칙 카드.

    condition_text는 화면 표시용 문구이고, 실제 평가는 kind/threshold_count/
    window_minutes로 한다. 문구를 파싱해서 판단하면 사용자가 문구를 고치는 순간
    동작이 바뀌어버리므로 분리해 둔다.
    """

    __tablename__ = "alert_rules"

    # 절대 횟수 규칙. 질환 판정 근거로는 쓸 수 없고(횟수 분포가 질환 간 크게 겹친다,
    # core/guidance.py 참조) 사용자가 직접 정한 관찰 기준으로만 의미가 있다.
    KIND_COUNT = "count_window"    # 지정 시간 안에 N회 이상
    KIND_NIGHT = "night_window"    # 야간 시간대에 한해 N회 이상
    KIND_UNKNOWN = "unknown"       # 미등록 화자의 기침이 발생하면 즉시

    # 참고자료가 권고한 경고 구조 (P6).
    KIND_BASELINE = "baseline_delta"   # 개인 기준선 대비 배수 증가가 지속될 때
    KIND_DURATION = "duration_days"    # 기침이 N일 이상 이어질 때
    KIND_URGENT = "urgent_symptom"     # 긴급 증상 입력 시 횟수와 무관하게 즉시

    # 근거가 임상 지침에 있는 종류. 화면에서 '탐색용'과 구분해 표시한다.
    CLINICAL_KINDS = (KIND_DURATION, KIND_URGENT)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50))            # 예: "이상 징후"
    condition_text: Mapped[str] = mapped_column(String(100))  # 예: "기침 ≥ 10회 / 1시간"
    target_text: Mapped[str] = mapped_column(String(50), default="전체 화자")
    # 실제로 알림이 도달하는 경로만 적는다. 웹훅 전송은 아직 구현하지 않았으므로
    # (alert_engine에 자리만 있음) "웹훅 발송"이라고 표시하면 화면이 거짓을 말하게 된다.
    channels_text: Mapped[str] = mapped_column(String(100), default="대시보드 표시")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # --- 평가 파라미터 ---
    kind: Mapped[str] = mapped_column(String(20), default=KIND_COUNT)
    threshold_count: Mapped[int] = mapped_column(Integer, default=10)
    window_minutes: Mapped[int] = mapped_column(Integer, default=60)
    night_start_hour: Mapped[int] = mapped_column(Integer, default=22)   # 현지 시각 기준
    night_end_hour: Mapped[int] = mapped_column(Integer, default=6)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=30)   # 재알림 억제

    # --- 변화 경고(KIND_BASELINE) ---
    baseline_days: Mapped[int] = mapped_column(Integer, default=7)        # 기준선 학습 창
    ratio_threshold: Mapped[float] = mapped_column(Float, default=2.0)    # 기준선 대비 배수
    sustain_hours: Mapped[int] = mapped_column(Integer, default=24)       # 증가가 이어져야 하는 시간

    # --- 기간 경고(KIND_DURATION) ---
    duration_days: Mapped[int] = mapped_column(Integer, default=14)
    # 기침 없는 날이 이 일수를 넘으면 '한 번 멎었다'로 보고 지속일수를 리셋한다.
    allowed_gap_days: Mapped[int] = mapped_column(Integer, default=2)


class AlertRuleChange(Base):
    """알림 규칙 변경 이력.

    규칙은 알림이 언제 울릴지를 정하는 값이라, 조용해진 이유가 "기침이 줄어서"인지
    "누가 규칙을 껐거나 기준을 올려서"인지 구분할 수 있어야 한다. 변경 시점의 요약과
    누가 바꿨는지를 남긴다. 규칙 행이 지워져도 이력은 남도록 FK를 걸지 않는다.
    """

    __tablename__ = "alert_rule_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    rule_name: Mapped[str] = mapped_column(String(50), default="")
    action: Mapped[str] = mapped_column(String(20), default="update")  # create|update|duplicate
    summary: Mapped[str] = mapped_column(String(300), default="")      # "기준 10회 → 12회" 식
    actor: Mapped[str] = mapped_column(String(50), default="—")        # 변경한 관리자 계정
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SymptomReport(Base):
    """사용자·보호자가 입력하는 동반 증상 (P6).

    참고자료: 객혈·호흡곤란·청색증·지속적 흉통·의식저하·낮은 SpO₂가 있으면
    기침 횟수와 무관하게 즉시 진료해야 한다. 이 값들은 소리로 알 수 없으므로
    사람이 입력하는 경로가 반드시 필요하다.

    증상 코드는 core/guidance.py의 SYMPTOM_LABELS 키를 쉼표로 이어 붙인다.
    """

    __tablename__ = "symptom_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[Optional[int]] = mapped_column(ForeignKey("persons.id"), nullable=True)
    reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    symptoms: Mapped[str] = mapped_column(String(255), default="")   # 쉼표 구분 코드
    spo2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    respiratory_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    def codes(self) -> list[str]:
        return [c for c in (self.symptoms or "").split(",") if c]


class DeviceUptime(Base):
    """엣지 장치의 시간대별 생존 기록 (P6).

    왜 필요한가 — **엣지는 기침이 있을 때만 서버에 말을 건다.** 그래서 CoughEvent만으로는
    "조용한 시간"과 "장치가 꺼진 시간"이 DB에서 똑같이 보인다. 이 때문에 두 가지가 망가진다:
      1. `device_online` 판정이 사실상 "최근에 진짜 기침이 있었나"가 되어 생존 신호가 못 된다
      2. 개인 기준선이 가동 중단 구간을 '기침 0회'로 세어 실제보다 낮게 잡히고,
         복구 후 '평소의 N배' 오경보가 난다

    비트 하나마다 행을 남기면 하루 1,440행이 쌓이므로 **(장치, UTC 시각) 단위로 묶어**
    개수만 센다. 하루 24행이면 충분하고, 시간대별 기준선과 해상도도 맞는다.
    """

    __tablename__ = "device_uptime"
    __table_args__ = (UniqueConstraint("device_id", "hour_utc", name="uq_device_hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str] = mapped_column(String(50), index=True)
    hour_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # 정시로 절삭
    beat_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    """대시보드 로그인 계정 — 비밀번호는 PBKDF2 해시로만 저장."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    salt: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20), default="admin")  # admin | guardian
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

"""발작 단위 화자 판정 — 기침 하나가 아니라 연속된 여러 개를 묶어서 판단한다.

**왜.** 기침 한 개로 화자를 맞히는 것은 2026-08-28 날짜 분리 측정에서 등록 2인 중
택1 기준 89.7%(T-norm)였다. 같은 데이터를 발작 단위로 묶으면:

    기침 1개  89.7%   ·  3개 100%  ·  5개 100%  ·  10개 100%
    (등록=8/27 s01 8건, 시험=8/28 s01 19건 + 8/26 hwang 20건, 코호트 누수 제거)

한 번의 발작은 같은 사람이 낸 것이 거의 확실하므로 묶어서 평균 내면 클립 하나의
운(마이크와의 각도, 그때의 자세)이 상쇄된다. 판정 단위를 올리는 것이 이 프로젝트에서
모델을 바꾸는 것보다 훨씬 크게 먹혔다.

**한계 — 미등록자에게는 오히려 해롭다.** 묶으면 분산이 줄어 **틀린 답도 더 확신하게**
된다. 등록되지 않은 choi 40건은 묶음 3개 이상에서 마진 0.8까지 보류율 0.0%로,
전부 자신 있게 등록자 중 하나로 찍혔다. **이 모듈은 "둘 중 누구인가"를 답할 뿐
"등록된 사람이 맞는가"는 답하지 않는다.** 미등록 거부를 주장하면 안 된다.

**마진 보류**도 같은 이유로 미등록자 필터가 아니다. 등록자 둘이 비슷하게 나올 때
잘못된 이름을 붙이느니 비워두는 장치다(잘못된 이름은 이력을 오염시키지만 unknown은
그렇지 않다). 실측상 묶음 3개에서 마진 0.2는 커버리지 97.0%에 정확도 100%로
사실상 공짜다 — 이 표본에서 정확도를 올려주지는 않지만 잃는 것도 없다.
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import timedelta
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import CoughEvent, Person
from ..ml.identifier import bytes_to_embedding
from ..ml.tnorm import tnorm

# 한 발작으로 묶는 시간 창. 엣지 쿨다운이 2초라 60초면 연속 기침 최대 30개가 들어간다.
WINDOW_S = float(os.environ.get("COUGHID_BOUT_WINDOW_S", "60"))
# 이보다 적으면 판정하지 않는다. 1~2개짜리는 위 표에서 89.7%라 이름을 붙일 근거가 약하다.
MIN_CLIPS = int(os.environ.get("COUGHID_BOUT_MIN", "3"))
# 1등과 2등의 차이가 이보다 작으면 보류. T-norm이 켜져 있으면 z 단위다.
MARGIN = float(os.environ.get("COUGHID_BOUT_MARGIN", "0.2"))

ENABLED = os.environ.get("COUGHID_BOUT", "1") not in ("0", "false", "False")


class BoutResult:
    def __init__(self, person_id: Optional[int], size: int, reason: str,
                 margin: Optional[float] = None):
        self.person_id = person_id
        self.size = size
        self.reason = reason        # decided | too_few | low_margin | no_candidates
        self.margin = margin

    def as_dict(self) -> dict:
        return {"person_id": self.person_id, "size": self.size,
                "reason": self.reason, "margin": self.margin}


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


def members(db: Session, event: CoughEvent) -> list[CoughEvent]:
    """같은 장치에서 WINDOW_S 안에 발생한 이벤트들 (event 포함, 시간순)."""
    lo = event.captured_at - timedelta(seconds=WINDOW_S)
    rows = db.scalars(
        select(CoughEvent)
        .where(CoughEvent.device_id == event.device_id,
               CoughEvent.captured_at >= lo,
               CoughEvent.captured_at <= event.captured_at)
        .order_by(CoughEvent.captured_at)).all()
    return list(rows)


def _candidate_scores(db: Session, rows: list[CoughEvent]) -> dict[int, list[float]]:
    """이벤트별로 저장된 상위 2인 점수를 후보별로 모은다.

    저장 규약(events.py와 짝):
        person_id 가 있으면      person_id/similarity = 1등, runner_up_* = 2등
        person_id 가 None 이면   runner_up_* = **임계치를 못 넘은 1등**
    즉 어느 경우든 runner_up 슬롯은 비지 않는다. 1등 이름을 잃지 않으려는 규약이다.

    점수는 저장된 원본 코사인이고, 여기서 등록본별 T-norm을 다시 걸어 z로 바꾼다.
    코사인 그대로 평균 내면 후한 등록본(중심에 가까운 것)이 이긴다.
    """
    templates = {}
    for p in db.scalars(select(Person)).all():
        if p.embedding_ref:
            templates[p.id] = _l2(bytes_to_embedding(p.embedding_ref))

    scores: dict[int, list[float]] = defaultdict(list)
    for e in rows:
        pairs = []
        if e.person_id is not None and e.similarity is not None:
            pairs.append((e.person_id, e.similarity))
        if e.runner_up_id is not None and e.runner_up_sim is not None:
            pairs.append((e.runner_up_id, e.runner_up_sim))
        for pid, sim in pairs:
            ref = templates.get(pid)
            if ref is None:
                continue                      # 등록본이 지워진 화자
            z = tnorm.z(sim, ref)
            scores[pid].append(sim if z is None else z)
    return scores


def judge(db: Session, event: CoughEvent) -> BoutResult:
    """발작을 판정하고 구성원 이벤트의 화자를 갱신한다.

    **사람이 손으로 지정한 이벤트(person_source='manual')는 건드리지 않는다.**
    보정을 모델 판정으로 되돌리면 사용자가 고친 의미가 사라진다.
    """
    rows = members(db, event)
    size = len(rows)
    if size < MIN_CLIPS:
        return _apply(db, rows, None, size, "too_few", None)

    scores = _candidate_scores(db, rows)
    if not scores:
        return _apply(db, rows, None, size, "no_candidates", None)

    ranked = sorted(((float(np.mean(v)), pid) for pid, v in scores.items()), reverse=True)
    best_key, best_id = ranked[0]
    margin = round(best_key - ranked[1][0], 4) if len(ranked) > 1 else None
    if margin is not None and margin < MARGIN:
        return _apply(db, rows, None, size, "low_margin", margin)
    return _apply(db, rows, best_id, size, "decided", margin)


def _apply(db: Session, rows: list[CoughEvent], person_id: Optional[int],
           size: int, reason: str, margin: Optional[float]) -> BoutResult:
    changed = False
    for e in rows:
        if e.person_source == "manual":
            continue
        if e.person_id != person_id or e.person_source != "bout":
            e.person_id = person_id
            e.person_source = "bout"
            changed = True
    if changed:
        db.commit()
    return BoutResult(person_id, size, reason, margin)

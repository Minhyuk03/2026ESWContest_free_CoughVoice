"""SpeakerIdentifier — 사전학습 임베딩 + 코사인 매칭 (P3).

샘플 수(화자당 10~20개)가 밑바닥부터 학습하기엔 턱없이 부족하므로, 사전학습 모델을
**고정 특징 추출기로만** 쓰고 판정은 코사인 유사도로 한다. 파인튜닝은 그다음 문제다.

임베더는 `backbone.py`에서 고른다 (기본 WavLM, `COUGHID_BACKEND=ecapa`로 되돌릴 수 있다).
2026-08-27 교체: 실사용 엣지 클립 EER이 ECAPA 49.9% → WavLM 38.0%로, 신뢰구간이 겹치지
않는 개선이었다. **차원이 192 → 512로 바뀌므로 기존 등록본은 무효다.**

등록은 샘플 N개 임베딩의 평균을 Person.embedding_ref에 저장하는 방식이라
원본 음성을 보존하지 않는다(NFR-06).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .backbone import get_backbone
from .centering import get_centering
from .features import preprocess
from .projection import projection

# 임계치 0.40 — **이 값이 근거하던 수치는 2026-08-26에 정정되었다.**
#
#   이전 주석은 "동일인 30 / 타인 30에서 EER 16.7%, 0.40에서 FAR 7% · 정밀도 91%"라고
#   적었다. 그 수치는 현재 도구·데이터로 **재현되지 않는다.** 재현 시도 결과:
#     tools/eval_speakers.py     (통제 녹음, 전 세션, 동일인 220 / 타인 340)
#       원본 코사인 40.9% · Coswara 중심화 36.4% · Coswara 투영층 38.7%
#     투영층 운용 곡선 실측
#       0.35 → 재현율 72.3% / FAR 45.6% / 정밀도 50.6%
#       0.40 → 재현율 58.6% / FAR 36.8% / 정밀도 50.8%   ← 현재 이 값
#       0.50 → 재현율 34.1% / FAR 20.6% / 정밀도 51.7%
#   s01의 세션을 2개로 줄여 옛 조건을 최대한 되살려도 투영층 EER은 27.5%였다.
#   세션을 늘릴수록 나빠진다 = 날짜가 바뀌면 무너진다.
#
#   **실사용 조건에서는 더 나쁘다.** 엣지 상시 동작 중 화자 2명이 번갈아 2블록씩
#   기침한 실사용 클립 80건 기준 EER 46.3% — 동전 던지기와 구분되지 않는다
#   (tools/eval_label_session.py, 2026-08-26). 화자 내 유사도가 블록 내부에서는
#   +0.23~0.25지만 블록 간에는 타인 간(+0.18) 수준으로 떨어진다. 모델이 잡는 것은
#   목소리가 아니라 그 몇 분간의 마이크 위치·자세다.
#
#   따라서 **이 임계치는 "정밀도 91%를 내는 운용점"이 아니다.** 0.40은 옛 근거로
#   정해진 값을 호환을 위해 유지하는 것뿐이며, 식별 결과를 성능 주장의 근거로
#   쓰면 안 된다. 정밀도를 우선한다는 설계 의도(잘못된 이름은 이력을 오염시키지만
#   unknown은 그렇지 않다) 자체는 유효하나, 현재 모델은 그 의도를 달성하지 못한다.
#   **백본마다 유사도 분포가 다르므로 임계치도 따로 잡아야 한다.** WavLM 원본 코사인은
#   타인 평균이 +0.55로 ECAPA(+0.35)보다 훨씬 높다. ECAPA용 0.40을 그대로 쓰면 전부
#   수락된다. 아래 값은 2026-08-27 실측 EER 지점이며 `_THRESHOLDS`에 백본별로 둔다.
_THRESHOLDS = {
    "ecapa": 0.40,
    # WavLM 0.75 — 2026-08-27 실사용 엣지 클립 실측. **단일-단일 EER 지점(0.597)을
    # 그대로 쓰면 안 된다.** 운용은 등록 템플릿(여러 개 평균) 대 단일 클립이라 유사도가
    # 전반적으로 올라가고, 0.60에서는 FAR 76%가 나온다. 등록 템플릿 기준 실측 곡선:
    #   0.60 → 재현율 96.2% / FAR 76.2% / 정밀도 38.7%
    #   0.70 → 재현율 82.5% / FAR 40.0% / 정밀도 50.8%
    #   0.75 → 재현율 67.5% / FAR 21.9% / 정밀도 60.7%   ← 곡선의 무릎. 현재 값
    #   0.82 → 재현율 33.8% / FAR  6.2% / 정밀도 73.0%
    #   0.85 → 재현율 13.8% / FAR  1.2% / 정밀도 84.6%
    # 정밀도 우선 설계(잘못된 이름은 이력을 오염시키지만 unknown은 그렇지 않다)에 따라
    # FAR이 가장 크게 꺾이는 지점을 골랐다. **타인 대조군이 화자 1명(160 트라이얼)뿐이라
    # FAR 추정의 신뢰구간이 넓다** — 이 값은 잠정이며 화자를 늘려 재확정해야 한다.
    "wavlm": 0.75,
}


def threshold_for(backend: str) -> float:
    """백본별 기본 임계치. COUGHID_THRESHOLD가 있으면 그것이 우선한다."""
    env = os.environ.get("COUGHID_THRESHOLD")
    if env:
        return float(env)
    return _THRESHOLDS.get(backend, 0.40)


DEFAULT_THRESHOLD = threshold_for(os.environ.get("COUGHID_BACKEND", "wavlm"))


class IdentifyResult:
    def __init__(self, person_id: Optional[int], similarity: Optional[float]):
        self.person_id = person_id
        self.similarity = similarity


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    return np.asarray(emb, dtype=np.float32).tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


class SpeakerIdentifier:
    def __init__(self, threshold: Optional[float] = None, backend: Optional[str] = None):
        self.backbone = get_backbone(backend)
        # ECAPA는 투영층(중심화를 내부에 포함)을 쓴다.
        # WavLM의 중심화는 **기본 끔**이다 — 2026-08-27 실측에서 도메인에 따라 부호가
        # 갈렸다. 통제 녹음에서는 EER 37.2% → 30.5%로 도움이 되지만, 실제 배포 조건인
        # 엣지 클립(16kHz·게인·거리 제각각)에서는 38.0% → 41.8%로 악화된다. 중심 벡터를
        # Coswara(스마트폰 녹음)에서 뽑았기 때문으로 보인다. 배포 조건을 기준으로 삼아
        # 끄고, 통제 조건 실험에서만 COUGHID_CENTERING=1 로 켠다.
        self.centering = None
        if not self.backbone.uses_projection and os.environ.get("COUGHID_CENTERING"):
            self.centering = get_centering(self.backbone.name)
        self.threshold = threshold if threshold is not None else threshold_for(self.backbone.name)

    @property
    def embed_dim(self) -> int:
        return self.backbone.dim

    def embed(self, wav_path: str, project: bool = True, **prep) -> np.ndarray:
        """WAV 1개 → L2 정규화된 임베딩. prep은 preprocess로 그대로 전달된다.

        ECAPA 백본에서는 기본이 투영층까지 적용한 128차원이다. 투영층을 **학습하거나
        평가하는** 코드는 project=False로 원본 192차원을 받아야 한다 — 그러지 않으면
        투영이 두 번 걸린다. WavLM 백본은 투영층 대상이 아니므로 이 인자가 무의미하다.
        """
        wav = preprocess(wav_path, **prep)
        emb = self.backbone.encode(wav)
        # 투영층은 **L2 정규화된** 임베딩으로 학습됐다(train_cough_projection.py의
        # embed_crops가 정규화 후 저장하고, mu도 그 분포에서 뽑았다). 정규화 전
        # 원본을 넣으면 스케일이 어긋나 투영이 무의미해진다.
        emb = _l2_normalize(emb)
        if not project:
            return emb                      # 보정 학습·평가용 원본
        if self.backbone.uses_projection:
            return _l2_normalize(projection.apply(emb))
        if self.centering is not None:
            return _l2_normalize(self.centering.apply(emb))
        return emb

    def enroll(self, wav_paths: Iterable[str], **prep) -> tuple[bytes, int]:
        """등록 샘플들의 평균 임베딩을 반환한다 → Person.embedding_ref, sample_count.

        등록과 검증은 **같은 전처리**를 써야 한다. prep을 넘길 때 양쪽을 일치시킬 것.
        """
        embs = [self.embed(p, **prep) for p in wav_paths]
        if not embs:
            raise ValueError("등록 샘플이 없습니다")
        mean = _l2_normalize(np.mean(np.stack(embs), axis=0))
        return embedding_to_bytes(mean), len(embs)

    def match(self, emb: np.ndarray,
              registry: Sequence[tuple[int, bytes]]) -> IdentifyResult:
        """등록 화자 중 가장 가까운 1명. 임계치 미달이면 unknown(FR-05)."""
        best_id, best_sim = None, -1.0
        stale = 0
        for person_id, blob in registry:
            ref = bytes_to_embedding(blob)
            if ref.size != emb.size:
                stale += 1        # 차원이 다른 낡은 등록본은 건너뛴다
                continue
            sim = float(np.dot(emb, ref))   # 양쪽 다 L2 정규화 → 내적 = 코사인
            if sim > best_sim:
                best_id, best_sim = person_id, sim
        if stale:
            # 백본을 바꾸면 기존 등록본의 차원이 맞지 않는다. 조용히 넘기면 전부
            # "미등록"으로 보여 원인을 못 찾으므로 한 번은 알린다.
            self._warn_stale(stale, len(registry))
        if best_id is None:
            return IdentifyResult(None, None)
        if best_sim < self.threshold:
            return IdentifyResult(None, round(best_sim, 4))   # unknown이어도 점수는 남긴다
        return IdentifyResult(best_id, round(best_sim, 4))

    def _warn_stale(self, stale: int, total: int) -> None:
        if getattr(self, "_stale_warned", False):
            return
        self._stale_warned = True
        print(f"[identifier] 등록본 {stale}/{total}건이 현재 백본"
              f"({self.backbone.name}, {self.backbone.dim}차원)과 차원이 달라 무시된다. "
              f"화자 재등록이 필요하다.", flush=True)

    def identify(self, wav_path: str,
                 registry: Sequence[tuple[int, bytes]] = ()) -> IdentifyResult:
        if not registry:
            return IdentifyResult(None, None)   # 등록 화자가 없으면 전부 unknown
        return self.match(self.embed(wav_path), registry)


identifier = SpeakerIdentifier()  # 싱글턴 — 모델 로딩 비용 1회

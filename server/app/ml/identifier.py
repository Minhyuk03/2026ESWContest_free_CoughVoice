"""SpeakerIdentifier — 사전학습 ECAPA-TDNN 임베딩 + 코사인 매칭 (P3).

샘플 수(화자당 10~20개)가 밑바닥부터 학습하기엔 턱없이 부족하므로,
VoxCeleb로 사전학습된 ECAPA를 **고정 특징 추출기로만** 쓰고 판정은 코사인 유사도로 한다.
파인튜닝은 그다음 문제다.

등록은 샘플 N개 임베딩의 평균을 Person.embedding_ref에 저장하는 방식이라
원본 음성을 보존하지 않는다(NFR-06).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .features import preprocess
from .projection import projection

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = os.environ.get(
    "COUGHID_MODEL_DIR", os.path.expanduser("~/.cache/coughid/ecapa"))
EMBED_DIM = 192

# 임계치 0.40 — 2026-08-24 측정 근거 (Coswara 투영층 적용 기준):
#   동일인 30건(s01 ses02 20 + x01 10) vs 타인 20건(s02). EER 15.8%.
#     0.30 → 재현율 83% / FAR 15% / 정밀도 89%
#     0.35 → 재현율 80% / FAR  5% / 정밀도 96%
#     0.40 → 재현율 70% / FAR  0% / 정밀도 100%   ← 채택
#   "확신할 때만 이름을 붙이고 나머지는 unknown"이 이 시스템에 맞는 동작이라
#   정밀도를 우선했다. 잘못된 이름은 이력을 오염시키지만 unknown은 그렇지 않다.
#
#   주의: 타인 표본이 s02 한 명(20건)뿐이다. FAR 0%의 95% 신뢰 상한은 약 15%다.
#   화자를 늘려 재확인할 것. 투영층 없이 원본 임베딩만 쓰면 EER 34.2%로 떨어진다.
DEFAULT_THRESHOLD = float(os.environ.get("COUGHID_THRESHOLD", "0.40"))


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
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._model = None

    def _ensure_model(self):
        """모델 로딩은 첫 호출까지 미룬다 — 서버 기동 시간과 테스트 비용을 줄이기 위함."""
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source=MODEL_SOURCE, savedir=MODEL_DIR)
        return self._model

    def embed(self, wav_path: str, project: bool = True, **prep) -> np.ndarray:
        """WAV 1개 → L2 정규화된 임베딩. prep은 preprocess로 그대로 전달된다.

        기본은 투영층까지 적용한 128차원이다. 투영층을 **학습하거나 평가하는**
        코드는 project=False로 원본 192차원을 받아야 한다 — 그러지 않으면 투영이
        두 번 걸린다.
        """
        model = self._ensure_model()
        wav = preprocess(wav_path, **prep)
        with torch.no_grad():
            emb = model.encode_batch(wav).squeeze().cpu().numpy()
        return _l2_normalize(projection.apply(emb) if project else emb)

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
        for person_id, blob in registry:
            ref = bytes_to_embedding(blob)
            if ref.size != emb.size:
                continue          # 차원이 다른 낡은 등록본은 건너뛴다
            sim = float(np.dot(emb, ref))   # 양쪽 다 L2 정규화 → 내적 = 코사인
            if sim > best_sim:
                best_id, best_sim = person_id, sim
        if best_id is None:
            return IdentifyResult(None, None)
        if best_sim < self.threshold:
            return IdentifyResult(None, round(best_sim, 4))   # unknown이어도 점수는 남긴다
        return IdentifyResult(best_id, round(best_sim, 4))

    def identify(self, wav_path: str,
                 registry: Sequence[tuple[int, bytes]] = ()) -> IdentifyResult:
        if not registry:
            return IdentifyResult(None, None)   # 등록 화자가 없으면 전부 unknown
        return self.match(self.embed(wav_path), registry)


identifier = SpeakerIdentifier()  # 싱글턴 — 모델 로딩 비용 1회

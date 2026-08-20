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

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = os.environ.get(
    "COUGHID_MODEL_DIR", os.path.expanduser("~/.cache/coughid/ecapa"))
EMBED_DIM = 192

# 잠정 임계치 0.45 — 2026-08-20 측정 근거:
#   s01 등록(ses01 10개) → 검증(ses02 20개) 동일인 유사도 평균 0.600, 최저 0.422.
#   0.45에서 FRR 10%, 0.60에서는 FRR 50%로 실사용 불가였다.
# 다만 **등록 화자가 1명뿐이라 FAR(타인 수락률)을 측정하지 못했다.** s02 수집 후
# tools/eval_identify.py의 임계치 곡선으로 반드시 재확정할 것 — 감으로 정하지 말 것.
DEFAULT_THRESHOLD = float(os.environ.get("COUGHID_THRESHOLD", "0.45"))


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

    def embed(self, wav_path: str, **prep) -> np.ndarray:
        """WAV 1개 → L2 정규화된 192차원 임베딩. prep은 preprocess로 그대로 전달된다."""
        model = self._ensure_model()
        wav = preprocess(wav_path, **prep)
        with torch.no_grad():
            emb = model.encode_batch(wav).squeeze().cpu().numpy()
        return _l2_normalize(emb)

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

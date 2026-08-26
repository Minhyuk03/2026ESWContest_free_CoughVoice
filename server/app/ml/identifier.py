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

            # speechbrain 지연 모듈이 나중에 무관한 코드(linecache 등)에서 깨어나
            # 미설치 의존성을 끌어오며 죽는 것을 막는다. 특히 예외 출력 경로에서
            # 발생하면 원래 에러가 가려져 원인 추적이 불가능해진다.
            from ._speechbrain_compat import neutralize_lazy_modules
            neutralize_lazy_modules()

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
        # 투영층은 **L2 정규화된** 임베딩으로 학습됐다(train_cough_projection.py의
        # embed_crops가 정규화 후 저장하고, mu도 그 분포에서 뽑았다). 정규화 전
        # 원본을 넣으면 스케일이 어긋나 투영이 무의미해진다.
        emb = _l2_normalize(emb)
        return _l2_normalize(projection.apply(emb)) if project else emb

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

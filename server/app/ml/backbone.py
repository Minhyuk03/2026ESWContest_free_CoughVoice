"""화자 임베더 — 교체 가능한 백본. 서버와 평가 도구가 같은 구현을 공유한다.

왜 분리했나:
    2026-08-27 측정에서 VoxCeleb 사전학습 ECAPA가 실사용 엣지 클립에서 EER 49.9%
    (동전 던지기)인 반면 WavLM-base-plus-sv는 38.0%로, 신뢰구간이 겹치지 않는
    개선을 보였다. 백본을 바꿔 끼울 자리가 필요해졌고, 그 자리가 여기다.

    더 중요한 이유는 **평가와 운영이 같은 임베더를 쓰도록 강제**하는 것이다. 이 프로젝트는
    등록·검증에 다른 전처리를 적용해 무의미한 수치를 낸 적이 있다(2026-08-20). 백본이
    두 벌로 갈리면 같은 사고가 재발한다.

선택:
    COUGHID_BACKEND=wavlm|ecapa (기본 wavlm)

주의:
    **임베딩 차원이 백본마다 다르다** (ECAPA 192 / WavLM 512). 백본을 바꾸면 기존
    등록본은 무효다. identifier.match()가 차원 불일치를 건너뛰므로 조용히 "미등록"이
    될 뿐 에러는 안 난다 — 교체 후에는 반드시 재등록할 것.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
import torch

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_DIR = os.environ.get(
    "COUGHID_MODEL_DIR", os.path.expanduser("~/.cache/coughid/ecapa"))
WAVLM_MODEL = os.environ.get(
    "COUGHID_WAVLM_MODEL", "microsoft/wavlm-base-plus-sv")

DEFAULT_BACKEND = os.environ.get("COUGHID_BACKEND", "wavlm")


class EcapaBackbone:
    """VoxCeleb 사전학습 ECAPA-TDNN. Coswara 투영층이 이 임베딩 위에서 학습됐다."""

    name = "ecapa"
    dim = 192
    uses_projection = True

    def __init__(self) -> None:
        self._model = None

    def _ensure(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            # speechbrain 지연 모듈이 나중에 무관한 코드(linecache 등)에서 깨어나
            # 미설치 의존성을 끌어오며 죽는 것을 막는다. 특히 예외 출력 경로에서
            # 발생하면 원래 에러가 가려져 원인 추적이 불가능해진다.
            from ._speechbrain_compat import neutralize_lazy_modules
            neutralize_lazy_modules()

            self._model = EncoderClassifier.from_hparams(
                source=ECAPA_SOURCE, savedir=ECAPA_DIR)
        return self._model

    def encode(self, wav: torch.Tensor) -> np.ndarray:
        """[1, N] 16kHz 텐서 → [dim] 임베딩 (정규화 전)."""
        model = self._ensure()
        with torch.no_grad():
            return model.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)


class WavLMBackbone:
    """WavLM + x-vector 헤드 (microsoft/wavlm-base-plus-sv).

    Coswara 투영층은 ECAPA 192차원 위에서 학습됐으므로 여기엔 적용하지 않는다
    (uses_projection=False). 애초에 그 투영층은 Coswara에 "같은 화자 다른 세션" 쌍이
    없어 세션 불변성을 배우지 못했고, 실사용에서 중심화보다 나빴다.
    """

    name = "wavlm"
    dim = 512
    uses_projection = False

    def __init__(self, model_id: str = WAVLM_MODEL) -> None:
        self.model_id = model_id
        self._fe = None
        self._model = None

    def _ensure(self):
        if self._model is None:
            # AutoModel 로 받으면 WavLM / UniSpeech-SAT 계열을 같은 코드로 쓴다.
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
            self._fe = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioXVector.from_pretrained(self.model_id)
            self._model.eval()
        return self._fe, self._model

    def encode(self, wav: torch.Tensor) -> np.ndarray:
        fe, model = self._ensure()
        x = wav.squeeze(0).cpu().numpy().astype(np.float32)
        inp = fe([x], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            return model(**inp).embeddings[0].cpu().numpy().astype(np.float32)


BACKENDS = {"ecapa": EcapaBackbone, "wavlm": WavLMBackbone}

_cache: dict = {}
_lock = threading.Lock()


def get_backbone(name: Optional[str] = None):
    """이름별로 한 번만 만들어 재사용한다. 모델 적재가 비싸기 때문이다."""
    key = (name or DEFAULT_BACKEND).lower()
    if key not in BACKENDS:
        raise ValueError(f"알 수 없는 백본: {key} (가능: {', '.join(BACKENDS)})")
    with _lock:
        if key not in _cache:
            _cache[key] = BACKENDS[key]()
        return _cache[key]

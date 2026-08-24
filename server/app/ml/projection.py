"""기침 화자 판별용 투영층 — ECAPA 임베딩을 기침에 맞게 재배치한다.

VoxCeleb 사전학습 ECAPA를 그대로 쓰면 기침에서 화자 변별력이 약하다
(2026-08-24 실측 EER 34.2%). Coswara 화자 770명으로 학습한 투영층을 얹으면
같은 데이터에서 EER 15.8%로 떨어진다. 백본은 건드리지 않으므로 CPU에서도 가볍다.

가중치는 tools/train_cough_projection.py가 만들어 ~/.cache/coughid/projection.npz에
저장한다. 파일이 없으면 투영 없이 원본 임베딩을 그대로 쓴다(성능은 떨어지지만 동작은 함).

출처: Coswara (LEAP Lab, IISc Bangalore) — CC BY 4.0
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

WEIGHTS = os.environ.get(
    "COUGHID_PROJECTION", os.path.expanduser("~/.cache/coughid/projection.npz"))


class Projection(nn.Module):
    """192차원 ECAPA 임베딩 → 128차원 기침 화자 공간."""

    def __init__(self, dim_in: int = 192, hidden: int = 512, dim_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, dim_out), nn.BatchNorm1d(dim_out),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)


class CoughProjection:
    """학습된 투영층을 지연 로딩해 적용한다. 없으면 통과시킨다."""

    def __init__(self, path: str = WEIGHTS):
        self.path = path
        self._model = None
        self._mu = None
        self._loaded = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.path):
            print(f"[projection] 가중치 없음 ({self.path}) — 원본 임베딩을 사용한다",
                  flush=True)
            return
        z = np.load(self.path, allow_pickle=True)
        model = Projection()
        model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files
                               if k.startswith("w_")})
        model.eval()
        self._model, self._mu = model, z["mu"]

    def apply(self, emb: np.ndarray) -> np.ndarray:
        """[192] 또는 [N,192] 임베딩에 중심화 + 투영을 적용한다."""
        self._ensure()
        if self._model is None:
            return emb
        single = emb.ndim == 1
        a = emb[None, :] if single else emb
        with torch.no_grad():
            out = self._model(torch.from_numpy((a - self._mu).astype(np.float32))).numpy()
        return out[0] if single else out


projection = CoughProjection()

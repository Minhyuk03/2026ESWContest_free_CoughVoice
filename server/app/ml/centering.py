"""중심화 — 임베딩에서 "기침 전반의 평균 방향"을 빼 화자 성분만 남긴다.

이 프로젝트에서 가장 값싸고 효과가 큰 보정이다. 2026-08-27 실측(WavLM, 통제 녹음):
원본 코사인 EER 37.2% → 중심화 25.9%. 백본 교체(40.9% → 37.2%)보다 기여가 크다.

왜 되는가:
    기침 임베딩은 서로 매우 닮아 있다(타인 간 코사인이 +0.55~0.79). 그 공통 성분이
    유사도의 대부분을 차지해 화자 차이를 덮는다. 평균을 빼면 공통 성분이 사라지고
    남은 차이만 비교하게 된다.

**평균은 반드시 외부 데이터에서 뽑아야 한다.** 평가셋 자신의 평균을 쓰면
(1) 시험 데이터 통계를 미리 본 셈이라 낙관적이고 (2) 서버는 기침 1건을 받는 시점에
평가셋이 없으므로 운영에서 재현이 불가능하다. 그래서 Coswara 화자 980명에서 뽑은
고정 벡터를 쓴다 — `tools/compute_center.py`가 만들어
`~/.cache/coughid/center_<백본>.npz`에 저장한다.

파일이 없으면 중심화 없이 통과시킨다(성능은 떨어지지만 동작은 한다).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def default_path(backend: str) -> str:
    env = os.environ.get("COUGHID_CENTER")
    if env:
        return env
    return os.path.expanduser(f"~/.cache/coughid/center_{backend}.npz")


class Centering:
    """백본별 중심 벡터를 지연 로딩해 적용한다. 없으면 통과시킨다."""

    def __init__(self, backend: str, path: Optional[str] = None):
        self.backend = backend
        self.path = path or default_path(backend)
        self._mu = None
        self._loaded = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._mu is not None

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.path):
            print(f"[centering] 중심 벡터 없음 ({self.path}) — 중심화 없이 진행한다. "
                  f"tools/compute_center.py --backbone {self.backend} 로 만들 수 있다",
                  flush=True)
            return
        z = np.load(self.path, allow_pickle=True)
        self._mu = np.asarray(z["mu"], dtype=np.float32)

    def apply(self, emb: np.ndarray) -> np.ndarray:
        """[dim] 또는 [N, dim] 임베딩에서 중심 벡터를 뺀다."""
        self._ensure()
        if self._mu is None:
            return emb
        if emb.shape[-1] != self._mu.shape[-1]:
            # 백본을 바꿨는데 옛 중심 벡터가 남아 있는 경우. 조용히 틀린 값을 내는 것보다
            # 중심화를 건너뛰는 편이 낫다.
            print(f"[centering] 차원 불일치 (임베딩 {emb.shape[-1]} vs 중심 "
                  f"{self._mu.shape[-1]}) — 중심화를 건너뛴다", flush=True)
            return emb
        return emb - self._mu


_cache: dict = {}


def get_centering(backend: str) -> Centering:
    if backend not in _cache:
        _cache[backend] = Centering(backend)
    return _cache[backend]

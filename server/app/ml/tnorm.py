"""T-norm — 등록본마다 다른 '후함'을 걷어내는 점수 표준화.

**왜 필요한가.** 2026-08-28 측정에서 s01 등록본이 전역 평균 방향과 코사인 0.946으로
거의 중심에 놓여 있었다(hwang 등록본은 0.851). 중심에 가까운 템플릿은 **모두에게 후하다** —
전체 79건에 대해 s01 쪽 점수가 평균 +0.075 높았다. 그 결과:

    미등록자 choi 40건 중 35건(87.5%)이 s01로 찍혔다.
    choi가 s01에서 얻는 우위(+0.107)가 hwang 본인이 hwang 템플릿에서 얻는 우위(+0.095)보다 컸다.

즉 원본 코사인으로는 "s01을 알아본 것"과 "애매하면 s01로 보낸 것"이 구분되지 않는다.
등록 샘플을 늘릴수록 평균이 중심으로 끌려가 이 현상이 심해진다(hwang 등록 8개 → 본인
정답 80%, 20개 → 65%). 화자마다 등록 개수가 다르면 개수 차이만으로 승부가 갈린다.

**무엇을 하는가.** 등록본 T의 점수를, 등록자와 무관한 코호트 클립들이 T에서 받는
점수 분포로 표준화한다:

    z = (cos(x, T) - μ_T) / σ_T          μ_T, σ_T = 코호트가 T에서 받는 점수의 평균·표준편차

후한 템플릿은 μ_T가 높아 그만큼 깎인다. 2026-08-28 실측(코호트 누수 제거, 등록 2인 중 택1):

    기침 1개   원본 82.1% → **T-norm 89.7%**
    발작 3개   원본 97.0% → **T-norm 100%**
    발작 5개   원본 100%  → T-norm 100%

**하지 못하는 것 — 미등록자 거부가 아니다.** T-norm을 걸어도 choi의 발작 묶음은 여전히
자신 있게 s01로 간다(묶음 3개 이상에서 마진 0.8까지 보류율 0.0%). 묶음은 분산을 줄이므로
**틀린 답도 더 확신하게 만든다.** 미등록자를 거부하려면 이것 말고 다른 장치가 필요하다.

**코호트는 등록자가 포함되지 않아야 한다.** 코호트에 등록자 본인 클립이 섞이면 μ_T가
본인 점수에 끌려 올라가 본인이 손해를 본다. 시험셋과 겹쳐도 안 된다(누수).
`tools/build_tnorm_cohort.py`가 8/26 라벨 세션 아카이브(등록자가 아닌 hwang·choi 80건,
전부 엣지 클립)로 만든다. 통제 녹음(collect_cough.py)은 채널이 달라 쓰지 않는다.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

COHORT_PATH = os.environ.get(
    "COUGHID_TNORM_COHORT",
    os.path.join(os.path.expanduser("~"), ".cache", "coughid", "tnorm_cohort.npy"))

# 기본 켬. 코호트 파일이 없으면 조용히 꺼진다(운영을 멈추게 하지 않는다).
ENABLED = os.environ.get("COUGHID_TNORM", "1") not in ("0", "false", "False")

# 코호트가 이보다 적으면 μ·σ 추정이 불안정하다. 2026-08-28 기준 아카이브는 80건.
MIN_COHORT = 20


class TNorm:
    def __init__(self, path: str = COHORT_PATH):
        self.path = path
        self._cohort: Optional[np.ndarray] = None
        self._loaded = False
        self._warned = False

    @property
    def cohort(self) -> Optional[np.ndarray]:
        """(N, D) L2 정규화된 코호트 임베딩. 없으면 None."""
        if not self._loaded:
            self._loaded = True
            if ENABLED and os.path.isfile(self.path):
                try:
                    a = np.load(self.path).astype(np.float32)
                    if a.ndim == 2 and len(a) >= MIN_COHORT:
                        self._cohort = a
                    else:
                        self._warn(f"코호트 형식이 맞지 않는다 (shape={a.shape}, "
                                   f"최소 {MIN_COHORT}건)")
                except Exception as exc:                      # 손상된 파일
                    self._warn(f"코호트를 읽지 못했다: {exc}")
            elif ENABLED:
                self._warn(f"코호트 파일이 없다 ({self.path}) — "
                           f"tools/build_tnorm_cohort.py 로 만들 것")
        return self._cohort

    def _warn(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"[tnorm] {msg}. 원본 코사인으로 판정한다 "
                  f"(등록 개수가 다른 화자끼리 불리해진다)", flush=True)

    def available(self, dim: int) -> bool:
        c = self.cohort
        return c is not None and c.shape[1] == dim

    def stats(self, template: np.ndarray) -> Optional[tuple[float, float]]:
        """등록본 하나에 대한 (μ, σ). 코호트가 없거나 차원이 다르면 None."""
        c = self.cohort
        if c is None or c.shape[1] != template.size:
            return None
        s = c @ template
        sd = float(s.std())
        if sd < 1e-6:            # 코호트가 사실상 한 점 — 나누면 폭발한다
            return None
        return float(s.mean()), sd

    def z(self, sim: float, template: np.ndarray) -> Optional[float]:
        st = self.stats(template)
        return None if st is None else (sim - st[0]) / st[1]


tnorm = TNorm()   # 싱글턴 — 코호트 적재 1회

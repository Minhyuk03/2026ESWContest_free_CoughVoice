"""CoughGate — "이게 기침인가"를 판정하는 2차 게이트 (AudioSet 사전학습 PANNs).

엣지의 CoughDetector는 에너지·지속시간만 보므로 박수·문 닫기·말소리가 그대로 통과한다
(2026-08-20 실측: 임계치 전 구간에서 비기침 오탐률 100%). 화자 식별은 "누구와 닮았나"만
계산하므로, 걸러지지 않은 비기침은 그대로 오식별이 된다. 그 사이를 막는 것이 이 게이트다.

우리 데이터 50개로 직접 분류기를 학습하는 방안은 leave-one-out 82%에 그쳤다. 한 방·한 사람
데이터라 그 수치조차 낙관적이다. 그래서 AudioSet 527클래스로 사전학습된 PANNs(CNN14)를
고정 판정기로 쓴다 — 수천 개의 실제 기침으로 학습돼 있어 우리 녹음 환경에 과적합되지 않는다.

2026-08-20 1차 (기침 30 / 비기침 20, 조용한 방): 임계치 0.005에서 오탐률 0%.
2026-08-24 재측정 — **그 0.005는 과소평가였다.** TV를 켠 거실 20분 연속 녹음에서
    시간당 오탐이 201회 나왔다. 조용한 방 네거티브 20개(최고 0.0026)로 잡은 값이라
    실제 생활소음이 그 선을 넘나든 것이다. 3초 통제 샘플로 정한 임계치를 연속 운용에
    그대로 쓰면 안 된다는 사례다.

    연속 녹음 기준 운용 곡선 (기침 검출률은 우리 기침 60개 기준):
        0.005 → 201회/시간, 98%      0.010 → 24회/시간, 98%
        0.020 →  12회/시간, 97%      0.050 →  9회/시간, 97%
        0.200 →   6회/시간, 92%      0.300 →  6회/시간, 90%
    0.05를 택했다. 0.005 대비 오탐이 22배 줄고 검출률은 1%p만 손해다.
    참고: Hyfe(상용, 손목 착용) 시간당 1.03회. 우리는 방에 둔 마이크라 조건이 다르다.
"""
from __future__ import annotations

import os
import urllib.request

# panns_inference는 matplotlib을 import한다. speechbrain(식별 모듈)이 먼저 적재된 뒤에
# 그 import가 일어나면, matplotlib이 스택을 추적하는 과정에서 speechbrain의 지연 로딩
# 모듈을 건드려 미설치 의존성(k2)을 끌어오다 ImportError로 죽는다. 여기서 먼저 올려
# 적재 순서를 고정한다. 이 두 줄을 지우면 /events 첫 호출에서 서버가 터진다.
import matplotlib
matplotlib.use("Agg")           # 서버에는 디스플레이가 없다

import numpy as np
import torch
import torchaudio

from .features import read_wav

PANNS_RATE = 32000        # PANNs 입력 규격
MIN_SAMPLES = PANNS_RATE  # 1초. CNN14의 풀링 단계를 통과하려면 최소 길이가 필요하다
COUGH_CLASS = 47          # AudioSet "Cough"
THROAT_CLASS = 48         # AudioSet "Throat clearing" — 헛기침도 수집 대상이므로 함께 센다

# 임계치 근거는 위 모듈 docstring 참조. 감으로 정한 값이 아니다.
DEFAULT_COUGH_THRESHOLD = float(os.environ.get("COUGHID_COUGH_THRESHOLD", "0.05"))

PANNS_DIR = os.path.join(os.path.expanduser("~"), "panns_data")
LABELS_CSV = os.path.join(PANNS_DIR, "class_labels_indices.csv")
CHECKPOINT = os.path.join(PANNS_DIR, "Cnn14_mAP=0.431.pth")
LABELS_URL = ("http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
              "class_labels_indices.csv")
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def _ensure_assets() -> None:
    """panns_inference는 wget으로 자산을 받는데 macOS에는 wget이 없다. 직접 받아 둔다."""
    os.makedirs(PANNS_DIR, exist_ok=True)
    if not os.path.isfile(LABELS_CSV):
        print("[cough_gate] AudioSet 라벨 다운로드 중...", flush=True)
        urllib.request.urlretrieve(LABELS_URL, LABELS_CSV)
    if not os.path.exists(CHECKPOINT) or os.path.getsize(CHECKPOINT) < 3e8:
        print("[cough_gate] PANNs 체크포인트 다운로드 중 (약 300MB, 최초 1회)...", flush=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)


class CoughGate:
    def __init__(self, threshold: float = DEFAULT_COUGH_THRESHOLD):
        self.threshold = threshold
        self._tagger = None

    def _ensure_tagger(self):
        """모델 로딩은 첫 호출까지 미룬다 — 서버 기동과 테스트 비용을 줄이기 위함."""
        if self._tagger is None:
            _ensure_assets()
            from panns_inference import AudioTagging
            self._tagger = AudioTagging(checkpoint_path=CHECKPOINT, device="cpu")
        return self._tagger

    def score(self, wav_path: str) -> float:
        """Cough + Throat clearing 확률의 합. 크롭하지 않은 원본 전체를 넣는다."""
        tagger = self._ensure_tagger()
        x, rate = read_wav(wav_path)
        if len(x) == 0:
            return 0.0
        t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        if rate != PANNS_RATE:
            t = torchaudio.functional.resample(t, rate, PANNS_RATE)
        # 엣지 검출기는 링버퍼 상황에 따라 2.5초보다 짧은 클립을 보낼 수 있다.
        # 그대로 넣으면 CNN14가 "output size is too small"로 죽어 /events가 500을 낸다
        # (2026-08-24 장시간 녹음 평가에서 실제 발생).
        if t.shape[-1] < MIN_SAMPLES:
            t = torch.nn.functional.pad(t, (0, MIN_SAMPLES - t.shape[-1]))
        with torch.no_grad():
            clipwise, _ = tagger.inference(t.numpy())
        p = clipwise[0]
        return float(p[COUGH_CLASS]) + float(p[THROAT_CLASS])

    def check(self, wav_path: str) -> tuple[bool, float]:
        s = self.score(wav_path)
        return s >= self.threshold, round(s, 5)


gate = CoughGate()   # 싱글턴 — 체크포인트 로딩 비용 1회

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

2026-08-28 **0.05 → 0.01로 내렸다. 위 곡선이 낡았기 때문이다.**
    위 곡선은 엣지 하이패스(80Hz) *이전*에 그린 것이다. 그때는 엣지가 시간당 774회를
    올려보내서 게이트가 강하게 막아야 했다. 하이패스 도입 후 엣지는 23회/시간만 올린다.
    게이트가 보는 후보 자체가 34배 줄었는데 임계치는 그대로 두고 있었다.

    실환경 재측정 (2026-08-28, TV 켠 거실 69.2분 연속. 사람이 귀로 라벨:
    기침 23개 / 비기침 27개. 엣지는 기침 23개를 모두 트리거했고, 갈린 것은 게이트다):
        임계치   검출률   오탐/시간
        0.005    100.0%    3.5회
        0.010     82.6%    0.0회   ← 채택
        0.020     65.2%    0.0회
        0.050     39.1%    0.0회   (종전 값)
    **0.05는 검출률 43%p를 공짜로 버리고 있었다.** 0.01로 내려도 오탐은 0회
    (69.2분 무오탐 → 95% 상한 2.6회/시간)라 잃는 것이 없다.

    주의 — 라벨된 기침 23개의 게이트 점수 중앙값은 **0.038**이다. 통제 녹음 기침이
    0.83~0.96인 것과 비교하면 20분의 1이다. 실환경 기침은 PANNs가 잘 못 알아본다.
    표본은 환경 하나·하루·기침 23개뿐이므로 0.01은 **잠정값**이다. 환경이 바뀌면
    같은 방식(연속 녹음 → 사람 라벨 → 곡선)으로 다시 그릴 것.
    재현: tools/eval_ambient.py <녹음.wav> --threshold 0.05
"""
from __future__ import annotations

import os
import urllib.request
from typing import NamedTuple, Optional

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

# CNN14는 한 번의 forward로 527클래스를 전부 계산한다. 지금까지 위 두 개만 읽고
# 나머지를 버렸는데, 아래 두 개는 참고자료가 기록을 권한 지표라 함께 꺼낸다.
# 추가 연산 비용은 0이다 — 이미 계산된 벡터에서 인덱스를 더 읽을 뿐이다.
#
# **판정에는 쓰지 않는다.** 우리 데이터로 정확도를 재본 적이 없는 미검증 지표다.
# 특히 AudioSet "Whoop"(10)은 환호성을 뜻하지 백일해의 흡기성 whoop이 아니라서
# 쓰지 않는다. Gasp이 흡기음에 가깝지만 그 역시 검증된 대응은 아니다.
WHEEZE_CLASS = 42         # AudioSet "Wheeze"
GASP_CLASS = 44           # AudioSet "Gasp"

# 임계치 근거는 위 모듈 docstring 참조. 감으로 정한 값이 아니다.
DEFAULT_COUGH_THRESHOLD = float(os.environ.get("COUGHID_COUGH_THRESHOLD", "0.01"))

PANNS_DIR = os.path.join(os.path.expanduser("~"), "panns_data")
LABELS_CSV = os.path.join(PANNS_DIR, "class_labels_indices.csv")
CHECKPOINT = os.path.join(PANNS_DIR, "Cnn14_mAP=0.431.pth")
# 라벨은 https로 받는다(같은 호스트가 https를 지원한다). http면 중간자가 내용을
# 바꿔치기해도 알 수 없다.
LABELS_URL = ("https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
              "class_labels_indices.csv")
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
# 체크포인트 무결성 검증용 SHA256. 신뢰할 수 있는 값을 환경변수로 주면 그것과
# 일치할 때만 사용한다(공급망 변조 방지). 없으면 검증을 건너뛰되 경고한다 —
# 잘못된 해시를 코드에 박아 모두의 로딩을 깨뜨리는 것보다 낫다.
CHECKPOINT_SHA256 = os.environ.get("COUGHID_PANNS_SHA256", "").strip().lower()
CHECKPOINT_MIN_BYTES = 3 * 10**8   # 약 300MB — 부분 다운로드 걸러내기


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download_atomic(url: str, dest: str) -> None:
    """임시 파일로 받은 뒤 rename한다. 다운로드가 중간에 끊겨도 목적지에는
    잘린 파일이 '완성된 것처럼' 남지 않는다(예전 방식은 부분 파일을 그대로 뒀다)."""
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _ensure_assets() -> None:
    """panns_inference는 wget으로 자산을 받는데 macOS에는 wget이 없다. 직접 받아 둔다."""
    os.makedirs(PANNS_DIR, exist_ok=True)
    if not os.path.isfile(LABELS_CSV):
        print("[cough_gate] AudioSet 라벨 다운로드 중...", flush=True)
        _download_atomic(LABELS_URL, LABELS_CSV)
    if not os.path.exists(CHECKPOINT) or os.path.getsize(CHECKPOINT) < CHECKPOINT_MIN_BYTES:
        print("[cough_gate] PANNs 체크포인트 다운로드 중 (약 300MB, 최초 1회)...", flush=True)
        _download_atomic(CHECKPOINT_URL, CHECKPOINT)

    if CHECKPOINT_SHA256:
        actual = _sha256(CHECKPOINT)
        if actual != CHECKPOINT_SHA256:
            os.unlink(CHECKPOINT)   # 변조 가능성 — 다음 기동에서 다시 받게 지운다
            raise RuntimeError(
                f"PANNs 체크포인트 SHA256 불일치 (기대 {CHECKPOINT_SHA256[:12]}…, "
                f"실제 {actual[:12]}…) — 변조 가능성이 있어 파일을 삭제했다")
    else:
        print("[cough_gate] ⚠ COUGHID_PANNS_SHA256 미설정 — 체크포인트 무결성 검증을 "
              "건너뛴다", flush=True)


class GateResult(NamedTuple):
    is_cough: bool
    cough_score: float
    wheeze: Optional[float]     # 미검증 부가 지표 — 판정에 쓰지 않는다
    gasp: Optional[float]


class CoughGate:
    def __init__(self, threshold: float = DEFAULT_COUGH_THRESHOLD):
        self.threshold = threshold
        self._tagger = None

    def _ensure_tagger(self):
        """모델 로딩은 첫 호출까지 미룬다 — 서버 기동과 테스트 비용을 줄이기 위함."""
        if self._tagger is None:
            _ensure_assets()
            # speechbrain이 이미 적재돼 있다면(식별 모듈이 먼저 쓰였을 때) 지연 모듈을
            # 무력화해 둔다. 예외가 나면 traceback 출력 중에 그것들이 깨어나
            # 원래 에러를 가려버린다.
            from ._speechbrain_compat import neutralize_lazy_modules
            neutralize_lazy_modules()

            from panns_inference import AudioTagging
            self._tagger = AudioTagging(checkpoint_path=CHECKPOINT, device="cpu")
        return self._tagger

    def analyze(self, wav_path: str) -> "GateResult":
        """한 번의 forward로 판정 점수와 부가 지표를 함께 돌려준다.

        게이트 판정에 쓰는 것은 cough_score 하나뿐이다. wheeze·gasp는 기록용이다.
        """
        tagger = self._ensure_tagger()
        x, rate = read_wav(wav_path)
        if len(x) == 0:
            return GateResult(False, 0.0, None, None)
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
        score = float(p[COUGH_CLASS]) + float(p[THROAT_CLASS])
        return GateResult(
            is_cough=score >= self.threshold,
            cough_score=round(score, 5),
            wheeze=round(float(p[WHEEZE_CLASS]), 5),
            gasp=round(float(p[GASP_CLASS]), 5),
        )

    def score(self, wav_path: str) -> float:
        """Cough + Throat clearing 확률의 합. 크롭하지 않은 원본 전체를 넣는다."""
        return self.analyze(wav_path).cough_score

    def check(self, wav_path: str) -> tuple[bool, float]:
        r = self.analyze(wav_path)
        return r.is_cough, r.cough_score


gate = CoughGate()   # 싱글턴 — 체크포인트 로딩 비용 1회

"""FeatureExtractor — 원본 WAV를 ECAPA-TDNN 입력 규격으로 변환.

수집 스크립트(tools/collect_cough.py)가 남기는 원본은 mono / 32bit int(24bit 유효) /
48kHz / 3초 고정이다. 이를 그대로 모델에 넣지 않고 세 단계를 거친다.

1. **활성 구간 크롭** — 3초 중 기침은 0.3~0.5초뿐이고 나머지는 무음이다.
   전체를 넣으면 임베딩이 그 방의 무음·잔향에 지배된다. 2026-08-20 측정:
   크롭 없이는 목소리가 아닌 생활잡음(박수·문 닫기)조차 등록 화자와 0.626까지
   유사해져(최고 0.835) 진짜 기침 0.771과 거의 겹쳤다. 크롭하면 0.427로 떨어진다.
   **크롭 없이 관측되는 높은 유사도는 성능이 아니라 채널 착시다.**
2. **48k → 16k 리샘플** — ECAPA(VoxCeleb) 입력 규격. 정확히 3:1이라 깔끔하다.
3. **RMS 정규화** — 세션 간 레벨이 2배 이상 차이나(ses01 peak 평균 39% vs ses02 18%)
   필요할 것으로 봤으나, 측정 결과 **유사도에 영향이 전혀 없었다.** ECAPA가 입력
   특징을 발화 단위로 정규화하기 때문이다. 무음 클립 증폭을 막는 안전장치 겸
   ECAPA 외 백엔드로 교체할 경우를 대비해 남겨둔다.
"""
from __future__ import annotations

import wave

import numpy as np
import torch
import torchaudio

TARGET_RATE = 16000       # ECAPA 사전학습 규격
CROP_S = 1.2              # 크롭 길이 — 기침 본체 + 직후 숨소리까지 포함
PRE_ROLL_S = 0.15         # 피크보다 이만큼 앞에서 시작 (기침 어택 손실 방지)
TARGET_RMS = 0.05         # 정규화 목표 RMS
ENERGY_WIN_S = 0.02       # 활성 구간 탐색용 단시간 에너지 창

# collect_cough.py가 24bit 정렬(>>8)로 저장하므로 32bit 파일의 유효 스케일은 2^23이다.
FULL_SCALE = {2: 2 ** 15, 4: 2 ** 23}


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """WAV를 [-1, 1] 범위 float32 mono로 읽는다."""
    with wave.open(path, "rb") as w:
        n_ch, width, rate, n_frames = (
            w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
        raw = w.readframes(n_frames)

    if width not in FULL_SCALE:
        raise ValueError(f"지원하지 않는 샘플 폭: {width}바이트 ({path})")
    dtype = np.int16 if width == 2 else np.int32
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_ch > 1:
        x = x.reshape(-1, n_ch)[:, 0]   # 왼쪽 채널만 (L/R→GND 구성)
    return x / FULL_SCALE[width], rate


def crop_active(x: np.ndarray, rate: int,
                crop_s: float = CROP_S, pre_roll_s: float = PRE_ROLL_S) -> np.ndarray:
    """단시간 에너지가 가장 큰 지점을 찾아 그 앞뒤로 crop_s 만큼 잘라낸다."""
    n_crop = int(rate * crop_s)
    if len(x) <= n_crop:
        return x

    win = max(1, int(rate * ENERGY_WIN_S))
    # 제곱합의 이동평균 — 창 단위 에너지
    energy = np.convolve(x.astype(np.float64) ** 2, np.ones(win), mode="same")
    peak_i = int(energy.argmax())

    start = max(0, peak_i - int(rate * pre_roll_s))
    start = min(start, len(x) - n_crop)   # 끝을 넘지 않도록
    return x[start:start + n_crop]


def normalize_rms(x: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """RMS를 목표값으로 맞춘다. 클리핑이 나면 피크 기준으로 되돌린다."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return x                      # 완전 무음 — 증폭하면 잡음만 키운다
    y = x * (target_rms / rms)
    peak = float(np.abs(y).max())
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def preprocess(path: str, crop: bool = True, normalize: bool = True) -> torch.Tensor:
    """WAV 경로 → ECAPA에 바로 넣을 수 있는 [1, N] 16kHz 텐서.

    crop/normalize 플래그는 tools/eval_identify.py의 ablation 측정용이다.
    운영 경로에서는 둘 다 켜 둔 기본값을 쓴다.
    """
    x, rate = read_wav(path)
    if crop:
        x = crop_active(x, rate)
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    if rate != TARGET_RATE:
        t = torchaudio.functional.resample(t, rate, TARGET_RATE)
    x = t.squeeze(0).numpy()
    if normalize:
        x = normalize_rms(x)
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)

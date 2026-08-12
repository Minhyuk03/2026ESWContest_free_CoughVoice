# -*- coding: utf-8 -*-
"""규칙 기반 기침 / 대화 판별기 (Python 3.9 호환, numpy만 사용).

설계
----
1) 프레임 단위 특징 추출 (32ms 창 / 16ms 홉)
   - RMS(dBFS), 스펙트럴 센트로이드, 스펙트럴 플랫니스, 롤오프85, ZCR
2) 적응형 노이즈 플로어 + 히스테리시스 상태머신으로 "소리 이벤트" 분리
3) 이벤트 레벨 특징으로 가중 점수 계산 -> cough / speech / other

기침 vs 대화의 물리적 차이
   기침 : 폭발적 onset(<30ms), 짧은 지속(0.2~0.7s), 광대역/무성음(플랫니스↑,
          센트로이드↑), 에너지 피크가 앞쪽, 단일 봉우리
   대화 : 완만한 onset, 긴 지속(>0.8s), 하모닉 구조(플랫니스↓, 센트로이드↓),
          3~8Hz 음절 변조로 봉우리 여러 개
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------
class Config(object):
    def __init__(self):
        self.sr = 16000
        self.frame = 512          # 32 ms
        self.hop = 256            # 16 ms
        self.trigger_delta = 12.0  # 노이즈 플로어 대비 +dB 이면 이벤트 시작
        self.release_delta = 6.0   # 이 아래로 내려가면 종료 카운트 시작
        self.abs_floor_db = -58.0  # 절대 최소 트리거 레벨
        self.hang_frames = 10      # 종료 유예 (~160ms)
        self.min_dur = 0.08        # 이보다 짧으면 무시
        self.max_dur = 4.0         # 강제 종료
        self.cough_threshold = 0.55

    def update(self, d: Dict[str, Any]):
        for k, v in d.items():
            if hasattr(self, k) and isinstance(getattr(self, k), (int, float)):
                setattr(self, k, float(v))


# --------------------------------------------------------------------------
# 보조 함수
# --------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _ramp(x: float, lo: float, hi: float) -> float:
    """lo에서 0, hi에서 1로 선형 증가."""
    if hi == lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))


def _band(x: float, lo: float, peak_lo: float, peak_hi: float, hi: float) -> float:
    """사다리꼴 소속함수."""
    if x <= lo or x >= hi:
        return 0.0
    if x < peak_lo:
        return (x - lo) / max(peak_lo - lo, EPS)
    if x <= peak_hi:
        return 1.0
    return (hi - x) / max(hi - peak_hi, EPS)


def _count_peaks(env: np.ndarray, rel: float = 0.45) -> int:
    """에너지 포락선의 유의미한 봉우리 개수 (음절 변조 추정)."""
    if len(env) < 5:
        return 1
    e = env / (np.max(env) + EPS)
    # 3프레임 이동평균으로 평활
    k = np.ones(3) / 3.0
    e = np.convolve(e, k, mode="same")
    peaks = 0
    i = 1
    while i < len(e) - 1:
        if e[i] >= e[i - 1] and e[i] > e[i + 1] and e[i] > rel:
            peaks += 1
            i += 3  # 최소 간격
        else:
            i += 1
    return max(peaks, 1)


# --------------------------------------------------------------------------
# 프레임 특징
# --------------------------------------------------------------------------
class FrameFeatures(object):
    __slots__ = ("db", "rms", "centroid", "flatness", "rolloff", "zcr", "spec")

    def __init__(self, db, rms, centroid, flatness, rolloff, zcr, spec):
        self.db = db
        self.rms = rms
        self.centroid = centroid
        self.flatness = flatness
        self.rolloff = rolloff
        self.zcr = zcr
        self.spec = spec  # power spectrum (linear)


# --------------------------------------------------------------------------
# 스트림 분석기
# --------------------------------------------------------------------------
class StreamAnalyzer(object):
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.window = np.hanning(self.cfg.frame).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.cfg.frame, 1.0 / self.cfg.sr)
        self.buf = np.zeros(0, dtype=np.float32)
        self.noise_db = -70.0
        self.frame_index = 0
        self.warmup_frames = 40      # 약 0.65초 동안 배경 소음 측정
        self._warm = []  # type: List[float]

        # 로그 스펙트로그램 밴드 (UI 표시용)
        self.n_bands = 48
        edges = np.geomspace(60.0, 7800.0, self.n_bands + 1)
        self.band_idx = [
            np.where((self.freqs >= edges[i]) & (self.freqs < edges[i + 1]))[0]
            for i in range(self.n_bands)
        ]
        for i in range(self.n_bands):
            if len(self.band_idx[i]) == 0:
                near = int(np.argmin(np.abs(self.freqs - (edges[i] + edges[i + 1]) / 2)))
                self.band_idx[i] = np.array([near])

        # 이벤트 상태
        self.in_event = False
        self.hang = 0
        self.ev_frames = []  # type: List[FrameFeatures]
        self.ev_start_frame = 0
        self.pre_db = -70.0

    # ---------------- 프레임 계산 ----------------
    def _analyze_frame(self, x: np.ndarray) -> FrameFeatures:
        xw = x * self.window
        rms = float(np.sqrt(np.mean(xw * xw) + EPS))
        db = 20.0 * math.log10(rms + EPS)

        spec = np.abs(np.fft.rfft(xw)) ** 2
        total = float(np.sum(spec)) + EPS
        centroid = float(np.sum(self.freqs * spec) / total)

        pos = spec + 1e-10
        flatness = float(math.exp(float(np.mean(np.log(pos)))) / float(np.mean(pos)))

        csum = np.cumsum(spec)
        ri = int(np.searchsorted(csum, 0.85 * csum[-1]))
        rolloff = float(self.freqs[min(ri, len(self.freqs) - 1)])

        sign = np.signbit(x)
        zcr = float(np.mean(sign[1:] != sign[:-1])) if len(x) > 1 else 0.0

        return FrameFeatures(db, rms, centroid, flatness, rolloff, zcr, spec)

    def band_levels(self, spec: np.ndarray) -> List[int]:
        out = []
        for idx in self.band_idx:
            p = float(np.mean(spec[idx])) + EPS
            d = 10.0 * math.log10(p)
            v = int(_clamp01((d + 95.0) / 85.0) * 100.0)
            out.append(v)
        return out

    # ---------------- 메인 push ----------------
    def push(self, block: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """오디오 블록을 넣고 (프레임 결과 리스트, 완료된 이벤트 리스트) 반환."""
        cfg = self.cfg
        self.buf = np.concatenate([self.buf, block.astype(np.float32)])
        frames_out = []  # type: List[Dict[str, Any]]
        events_out = []  # type: List[Dict[str, Any]]

        while len(self.buf) >= cfg.frame:
            x = self.buf[: cfg.frame]
            self.buf = self.buf[cfg.hop:]
            ff = self._analyze_frame(x)
            self.frame_index += 1

            # --- 워밍업: 배경 소음 레벨 측정 (이 구간에서는 이벤트 검출 안 함) ---
            if self.frame_index <= self.warmup_frames:
                self._warm.append(ff.db)
                if self.frame_index == self.warmup_frames:
                    self.noise_db = float(np.percentile(np.array(self._warm), 25))
                    self._warm = []
                else:
                    self.noise_db = float(np.min(self._warm))
                frames_out.append(self._frame_msg(ff, self.noise_db + cfg.trigger_delta))
                continue

            # 적응형 노이즈 플로어: 아래로는 빠르게, 위로는 천천히 추종
            if not self.in_event:
                a = 0.25 if ff.db < self.noise_db else 0.004
                self.noise_db = (1 - a) * self.noise_db + a * ff.db
                self.noise_db = max(self.noise_db, -85.0)

            on_th = max(self.noise_db + cfg.trigger_delta, cfg.abs_floor_db)
            off_th = max(self.noise_db + cfg.release_delta, cfg.abs_floor_db - 6.0)

            if not self.in_event:
                if ff.db > on_th:
                    self.in_event = True
                    self.hang = 0
                    self.ev_frames = [ff]
                    self.ev_start_frame = self.frame_index
            else:
                self.ev_frames.append(ff)
                if ff.db < off_th:
                    self.hang += 1
                else:
                    self.hang = 0
                dur = len(self.ev_frames) * cfg.hop / float(cfg.sr)
                if self.hang >= int(cfg.hang_frames) or dur >= cfg.max_dur:
                    ev = self._finish_event()
                    if ev is not None:
                        events_out.append(ev)

            frames_out.append(self._frame_msg(ff, on_th))

        return frames_out, events_out

    def _frame_msg(self, ff: FrameFeatures, on_th: float) -> Dict[str, Any]:
        return {
            "db": round(ff.db, 2),
            "noise_db": round(self.noise_db, 2),
            "on_th": round(on_th, 2),
            "centroid": round(ff.centroid, 1),
            "flatness": round(ff.flatness, 4),
            "zcr": round(ff.zcr, 4),
            "active": self.in_event,
            "bands": self.band_levels(ff.spec),
        }

    # ---------------- 이벤트 종료 & 판정 ----------------
    def _finish_event(self) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        frames = self.ev_frames
        self.in_event = False
        self.hang = 0
        self.ev_frames = []

        # 배경 소음이 임계값 위로 올라와 계속 "이벤트"로 잡히는 경우 → 재보정
        if frames:
            all_db = np.array([f.db for f in frames])
            dur_all = len(frames) * cfg.hop / float(cfg.sr)
            if dur_all >= cfg.max_dur * 0.95 and (np.max(all_db) - np.median(all_db)) < 6.0:
                self.noise_db = float(np.percentile(all_db, 25))
                return None

        # 뒤쪽 무음(hangover) 제거
        while frames and frames[-1].db < max(self.noise_db + cfg.release_delta, cfg.abs_floor_db - 6.0):
            frames.pop()
        if len(frames) < 3:
            return None

        dur = len(frames) * cfg.hop / float(cfg.sr)
        if dur < cfg.min_dur:
            return None

        env = np.array([f.rms for f in frames], dtype=np.float64)
        dbs = np.array([f.db for f in frames], dtype=np.float64)
        cen = np.array([f.centroid for f in frames], dtype=np.float64)
        flat = np.array([f.flatness for f in frames], dtype=np.float64)
        zcr = np.array([f.zcr for f in frames], dtype=np.float64)
        roll = np.array([f.rolloff for f in frames], dtype=np.float64)

        w = env / (np.sum(env) + EPS)  # 에너지 가중 평균
        peak_i = int(np.argmax(env))
        peak_pos = peak_i / float(max(len(env) - 1, 1))
        attack_time = peak_i * cfg.hop / float(cfg.sr)

        crest = float(np.max(env) / (np.mean(env) + EPS))
        n_peaks = _count_peaks(env)
        centroid_m = float(np.sum(cen * w))
        flat_m = float(np.sum(flat * w))
        zcr_m = float(np.sum(zcr * w))
        roll_m = float(np.sum(roll * w))
        peak_db = float(np.max(dbs))

        # ---- 점수 구성요소 (1에 가까울수록 기침) ----
        s = {
            # 기침은 0.15~0.7s
            "duration": _band(dur, 0.06, 0.16, 0.70, 1.30),
            # 폭발적 onset: 피크까지 60ms 이내
            "attack": 1.0 - _ramp(attack_time, 0.02, 0.18),
            # 에너지 피크가 앞쪽 + 뒤로 감쇠
            "shape": (1.0 - _ramp(peak_pos, 0.12, 0.55)) * 0.6 + _ramp(crest, 1.4, 2.6) * 0.4,
            # 광대역 -> 센트로이드 높음
            "centroid": _ramp(centroid_m, 800.0, 2600.0),
            # 무성 잡음 -> 플랫니스 높음 (하모닉 음성은 낮음)
            "flatness": _ramp(flat_m, 0.03, 0.22),
            # 단일 봉우리 (음절 변조 없음)
            "unimodal": 1.0 - _ramp(float(n_peaks), 1.0, 4.0),
        }
        weights = {
            "duration": 0.18,
            "attack": 0.17,
            "shape": 0.13,
            "centroid": 0.20,
            "flatness": 0.18,
            "unimodal": 0.14,
        }
        score = sum(s[k] * weights[k] for k in weights)
        score = _clamp01(score)

        # ---- 라벨 결정 ----
        speechish = (dur >= 0.30 and n_peaks >= 2 and flat_m < 0.20 and centroid_m < 2600)
        if score >= cfg.cough_threshold and not (dur > 1.4 and n_peaks >= 4):
            label = "cough"
        elif speechish or dur >= 0.55:
            label = "speech"
        else:
            label = "other"

        return {
            "label": label,
            "score": round(score, 3),
            "duration": round(dur, 3),
            "peak_db": round(peak_db, 1),
            "attack_time": round(attack_time, 3),
            "peak_pos": round(peak_pos, 3),
            "crest": round(crest, 2),
            "n_peaks": int(n_peaks),
            "centroid": round(centroid_m, 1),
            "flatness": round(flat_m, 4),
            "zcr": round(zcr_m, 4),
            "rolloff": round(roll_m, 1),
            "parts": dict((k, round(v, 3)) for k, v in s.items()),
            "weights": weights,
        }


LABEL_KO = {"cough": "기침", "speech": "대화", "other": "기타 소음"}

#!/usr/bin/env python3
"""1단계 — Coswara로 기침 화자 투영층 학습 (ECAPA는 동결).

VoxCeleb 사전학습 ECAPA를 그대로 쓰면 기침에서 화자를 구분하지 못한다
(2026-08-24 자체 측정 EER 55% = 동전 던지기). 선행 연구는 같은 ECAPA를 기침
데이터로 적응시켜 EER 13.39%를 얻었다. 전체 파인튜닝은 CPU만 있는 환경에서
무리이므로, **임베딩은 한 번만 뽑아 동결하고 그 위 투영층만 학습**한다.

평가는 화자가 겹치지 않게 나눈다. 같은 화자가 학습과 평가에 함께 들어가면
성능이 부풀려진다 — 우리가 8/20에 겪은 것과 같은 종류의 함정이다.
등록=heavy(강한 기침) / 검증=shallow(얕은 기침)로 나누어, 같은 사람이라도
다른 발성을 맞히는지 본다.

출처: Coswara (LEAP Lab, IISc Bangalore) — CC BY 4.0

사용 예:
    python3 train_cough_projection.py                 # 임베딩 추출 + 학습 + 평가
    python3 train_cough_projection.py --skip-extract  # 캐시된 임베딩으로 재학습
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# speechbrain이 먼저 적재된 뒤 torch._dynamo가 처음 import되면(AdamW 생성 시점)
# 스택 추적 과정에서 speechbrain의 지연 모듈을 건드려 미설치 의존성(k2)을 끌어오다
# ImportError로 죽는다. server/app/ml/cough_gate.py의 matplotlib 문제와 같은 원인이므로
# 여기서 미리 올려 순서를 고정한다. 지우면 학습 시작 직후 터진다.
import torch._dynamo  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from app.ml.features import crop_peaks, normalize_rms, read_wav  # noqa: E402
from app.ml.identifier import SpeakerIdentifier, _l2_normalize  # noqa: E402
from app.ml.projection import Projection  # noqa: E402  — 서버와 같은 정의를 쓴다

DATA = os.path.expanduser("~/datasets/Coswara/coughs")
CACHE = os.path.expanduser("~/.cache/coughid/coswara_embeddings_mc.npz")
OUT = os.path.expanduser("~/.cache/coughid/projection.npz")
ENROLL_FILE, TEST_FILE = "cough-heavy.wav", "cough-shallow.wav"


# --------------------------------------------------------------- 임베딩 추출
def embed_crops(ident, path: str, n_crops: int) -> list[np.ndarray]:
    """한 녹음에서 피크 여러 개를 잘라 각각 임베딩한다."""
    import torch
    import torchaudio
    from app.ml.features import TARGET_RATE

    x, rate = read_wav(path)
    out = []
    model = ident._ensure_model()
    for seg in crop_peaks(x, rate, n_crops):
        t = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0)
        if rate != TARGET_RATE:
            t = torchaudio.functional.resample(t, rate, TARGET_RATE)
        wav = torch.from_numpy(
            np.ascontiguousarray(normalize_rms(t.squeeze(0).numpy()))).unsqueeze(0)
        with torch.no_grad():
            # 여기서는 투영을 적용하지 않는다. 투영층을 학습하는 중이므로 원본 임베딩이 필요하다.
            out.append(_l2_normalize(model.encode_batch(wav).squeeze().cpu().numpy()))
    return out


def extract(data_dir: str, n_crops: int = 3):
    """화자별로 heavy/shallow 각각에서 여러 크롭을 뽑는다.

    화자당 샘플이 2개(heavy 1 + shallow 1)뿐이면 같은 사람 안에서의 변이를 배울
    자료가 없어 거리 학습이 되지 않는다(2026-08-24 확인: 투영층이 중심화보다 나빴다).
    Coswara 녹음에는 연속 기침이 흔해 크롭을 여러 개 뽑을 수 있다.
    """
    ident = SpeakerIdentifier()
    speakers = []
    for date in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, date)
        if os.path.isdir(d):
            speakers += [(f"{date}/{p}", os.path.join(d, p)) for p in sorted(os.listdir(d))]

    print(f"참가자 {len(speakers)}명 임베딩 추출 (파일당 최대 {n_crops}크롭)...", flush=True)
    X, spk, role, names = [], [], [], []
    for i, (sid, pdir) in enumerate(speakers):
        per_file = {}
        for r, fn in enumerate((ENROLL_FILE, TEST_FILE)):
            path = os.path.join(pdir, fn)
            try:
                per_file[r] = embed_crops(ident, path, n_crops)
            except Exception as exc:
                print(f"  건너뜀 {sid}/{fn}: {exc}", flush=True)
                break
        if len(per_file) == 2 and all(len(v) for v in per_file.values()):
            sidx = len(names)
            for r, embs in per_file.items():
                for e in embs:
                    X.append(e); spk.append(sidx); role.append(r)
            names.append(sid)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(speakers)}  (유효 화자 {len(names)}명, 임베딩 {len(X)}개)",
                  flush=True)

    X = np.stack(X).astype(np.float32)
    spk = np.array(spk, dtype=np.int64)
    role = np.array(role, dtype=np.int64)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, spk=spk, role=role, names=np.array(names))
    print(f"캐시 저장: {CACHE}  ({len(X)}개 임베딩 / 화자 {len(names)}명 "
          f"/ 평균 {len(X)/max(len(names),1):.1f}개")
    return X, spk, role, names


# --------------------------------------------------------------- 평가
def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """동일인 거부율(FRR)과 타인 수락률(FAR)이 만나는 지점과 그때의 임계치."""
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    ts = np.linspace(lo, hi, 4000)
    frrs = np.array([(genuine < t).mean() for t in ts])
    fars = np.array([(impostor >= t).mean() for t in ts])
    i = int(np.argmin(np.abs(frrs - fars)))
    return float((frrs[i] + fars[i]) / 2), float(ts[i])


def build_trials(X, spk, role, keep, project=None):
    """등록=heavy 크롭 평균, 검증=shallow 크롭 각각. 화자별 최고 유사도로 점수를 낸다."""
    ids = sorted(keep)
    enroll, tests = [], []
    for si in ids:
        m = spk == si
        e = X[m & (role == 0)]
        t = X[m & (role == 1)]
        if len(e) == 0 or len(t) == 0:
            continue
        if project is not None:
            e, t = project(e), project(t)
        ref = e.mean(0)
        enroll.append(ref / (np.linalg.norm(ref) + 1e-9))
        tests.append(t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9))
    E = np.stack(enroll)                       # [화자, 차원]
    genuine, impostor = [], []
    rng = np.random.default_rng(0)
    for i, T in enumerate(tests):
        S = T @ E.T                            # [검증크롭, 등록화자]
        genuine.append(S[:, i])
        others = rng.choice(np.delete(np.arange(len(E)), i),
                            size=min(20, len(E) - 1), replace=False)
        impostor.append(S[:, others].ravel())
    return np.concatenate(genuine), np.concatenate(impostor)


def report(label, X, spk, role, keep, project=None):
    g, x = build_trials(X, spk, role, keep, project)
    e, t = eer(g, x)
    print(f"  {label:<28} EER {e*100:5.2f}%   동일인 {g.mean():+.3f} / 타인 {x.mean():+.3f}"
          f"   격차 {g.mean()-x.mean():+.3f}")
    return e


# --------------------------------------------------------------- 학습
class AAMSoftmax(nn.Module):
    """각도 마진 손실 — 화자 분류를 통해 임베딩 사이 각도를 벌린다."""

    def __init__(self, dim, n_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.margin, self.scale = margin, scale

    def forward(self, emb, y):
        W = nn.functional.normalize(self.W, dim=1)
        cos = emb @ W.T
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.margin)
        onehot = torch.zeros_like(cos).scatter_(1, y.view(-1, 1), 1.0)
        return nn.functional.cross_entropy(self.scale * (onehot * target + (1 - onehot) * cos), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    if args.skip_extract and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        X, spk, role, names = z["X"], z["spk"], z["role"], list(z["names"])
        print(f"캐시 로드: 임베딩 {len(X)}개 / 화자 {len(names)}명 "
              f"(평균 {len(X)/len(names):.1f}개)")
    else:
        X, spk, role, names = extract(args.data, args.crops)

    n_spk = len(names)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_spk)
    n_test = max(30, int(n_spk * args.test_frac))
    test_spk = set(perm[:n_test].tolist())
    train_spk = set(perm[n_test:].tolist())
    print(f"\n화자 분할: 학습 {len(train_spk)}명 / 평가 {len(test_spk)}명 (겹침 없음)\n")

    print("■ 기준선 — 사전학습 ECAPA 그대로 (평가 화자)")
    base = report("원본 코사인", X, spk, role, test_spk)

    tr_mask = np.isin(spk, sorted(train_spk))
    mu = X[tr_mask].mean(0, keepdims=True)
    print("\n■ 중심화만 적용")
    cent = report("centering", X, spk, role, test_spk, project=lambda A: A - mu)

    print(f"\n■ 투영층 학습 (AAM-softmax, {args.epochs} epoch)")
    Xtr = torch.from_numpy((X[tr_mask] - mu).astype(np.float32))
    remap = {s_: i for i, s_ in enumerate(sorted(train_spk))}
    ytr = torch.tensor([remap[int(v)] for v in spk[tr_mask]], dtype=torch.long)
    print(f"    학습 샘플 {len(Xtr)}개 / 화자 {len(train_spk)}명 "
          f"(화자당 평균 {len(Xtr)/len(train_spk):.1f}개)")

    torch.manual_seed(0)
    model = Projection()
    head = AAMSoftmax(128, len(train_spk))
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n, bs = len(Xtr), 256
    for ep in range(args.epochs):
        model.train(); head.train()
        order = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            b = order[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = head(model(Xtr[b]), ytr[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:>3}  loss {tot/n:.4f}", flush=True)

    model.eval()

    def project(A):
        with torch.no_grad():
            return model(torch.from_numpy((A - mu).astype(np.float32))).numpy()

    print()
    proj = report("투영층 적용", X, spk, role, test_spk, project=project)

    np.savez(OUT, mu=mu, **{f"w_{k}": v.numpy() for k, v in model.state_dict().items()})
    print(f"\n투영층 저장: {OUT}")
    best = min(base, cent, proj)
    print(f"\n기준선 {base*100:.2f}%  ·  중심화 {cent*100:.2f}%  ·  투영 {proj*100:.2f}%"
          f"   → 최고 {best*100:.2f}%")
    print("참고: 선행 연구(Coswara + ECAPA 백본 파인튜닝) EER 13.39%")


if __name__ == "__main__":
    main()

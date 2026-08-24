#!/usr/bin/env bash
# 서버 상시 운영 장비(맥) 설치 스크립트
#
# 저장소를 iCloud 폴더 안에 두면 파일이 evict되어 git·python이 "Operation timed out"으로
# 랜덤 실패한다(실제로 겪음). 그래서 ~/srv 아래에 클론한다.
# venv도 같은 이유로 iCloud 밖(~/.venvs)에 만든다.
#
# 사용법:
#   bash setup_server_mac.sh
# 이후 투영층 가중치를 개발 장비에서 복사해야 한다 (스크립트가 안내한다).
set -euo pipefail

REPO_URL="https://github.com/Minhyuk03/Cough_EmbeddedSystem.git"
SRV_DIR="$HOME/srv/Cough_EmbeddedSystem"
VENV="$HOME/.venvs/coughid"
PROJ="$HOME/.cache/coughid/projection.npz"

say() { printf "\n\033[1m▶ %s\033[0m\n" "$1"; }

say "1/5 사전 확인"
command -v git >/dev/null || { echo "git이 없습니다. Xcode Command Line Tools를 설치하세요: xcode-select --install"; exit 1; }
PY=$(command -v python3) || { echo "python3가 없습니다."; exit 1; }
echo "  git  : $(git --version)"
echo "  python: $($PY -V)"

say "2/5 저장소 준비 ($SRV_DIR)"
if [ -d "$SRV_DIR/.git" ]; then
  git -C "$SRV_DIR" pull --ff-only
else
  mkdir -p "$(dirname "$SRV_DIR")"
  git clone "$REPO_URL" "$SRV_DIR"
fi

say "3/5 가상환경 ($VENV)"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
# torch·speechbrain·panns는 용량이 커서 시간이 걸린다 (합계 약 1GB)
"$VENV/bin/pip" install -q -r "$SRV_DIR/server/requirements.txt"
echo "  설치 완료"

say "4/5 모델 자산 확인"
if [ -f "$PROJ" ]; then
  echo "  투영층 가중치 있음: $PROJ"
else
  cat <<MSG
  ⚠ 투영층 가중치가 없습니다.
    개발 장비에서 아래를 실행해 복사하세요 (657KB):

      scp ~/.cache/coughid/projection.npz $(whoami)@$(scutil --get LocalHostName).local:~/.cache/coughid/

    먼저 이 장비에서 폴더를 만들어 두세요:  mkdir -p ~/.cache/coughid
    없으면 서버는 동작하지만 화자 식별 정확도가 크게 떨어집니다
    (EER 16.7% → 34.2%).
MSG
fi
mkdir -p "$HOME/.cache/coughid"
echo "  ECAPA·PANNs 체크포인트는 첫 요청 처리 시 자동으로 내려받습니다 (약 400MB, 1회)."

say "5/5 동작 확인"
cd "$SRV_DIR/server"
"$VENV/bin/python" - <<'PY'
import sys; sys.path.insert(0, ".")
from app.main import app
from app.ml.identifier import identifier
from app.ml.cough_gate import gate
print(f"  앱 임포트 OK · 식별 임계치 {identifier.threshold} · 게이트 임계치 {gate.threshold}")
PY

cat <<MSG

설치 완료.

  서버 수동 실행:
    cd $SRV_DIR/server && $VENV/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

  상시 운영(자동 시작)으로 등록하려면:
    bash $SRV_DIR/deploy/install_launchd.sh

  이 장비의 접속 이름: $(scutil --get LocalHostName).local
MSG

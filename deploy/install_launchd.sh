#!/usr/bin/env bash
# 맥에서 서버를 상시 실행하도록 launchd에 등록한다.
#
# launchd는 프로세스가 죽으면 자동으로 다시 띄우고(KeepAlive), 로그인 없이도
# 부팅 시 시작된다(RunAtLoad). 맥이 잠들면 서버가 멈추므로 절전도 함께 끈다.
set -euo pipefail

SRV_DIR="$HOME/srv/Cough_EmbeddedSystem"
VENV="$HOME/.venvs/coughid"
LABEL="com.coughid.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/Library/Logs/coughid"

[ -d "$SRV_DIR/server" ] || { echo "저장소가 없습니다: $SRV_DIR — setup_server_mac.sh를 먼저 실행하세요"; exit 1; }
[ -x "$VENV/bin/uvicorn" ] || { echo "가상환경이 없습니다: $VENV"; exit 1; }
mkdir -p "$LOGDIR" "$(dirname "$PLIST")"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV/bin/uvicorn</string>
        <string>app.main:app</string>
        <string>--host</string><string>0.0.0.0</string>
        <string>--port</string><string>8000</string>
    </array>
    <key>WorkingDirectory</key><string>$SRV_DIR/server</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$LOGDIR/server.log</string>
    <key>StandardErrorPath</key><string>$LOGDIR/server.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key><string>$HOME</string>
${COUGHID_DEVICE_TOKEN:+        <key>COUGHID_DEVICE_TOKEN</key><string>$COUGHID_DEVICE_TOKEN</string>}
    </dict>
</dict>
</plist>
PLIST_EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "등록 완료: $LABEL"
echo "  상태 확인 : launchctl list | grep coughid"
echo "  로그      : tail -f $LOGDIR/server.log"
echo "  중지      : launchctl unload $PLIST"

cat <<'MSG'

⚠ 절전 설정은 관리자 권한이 필요해 이 스크립트가 하지 않습니다.
  맥이 잠들면 서버도 멈추므로, 상시 운영하려면 직접 실행하세요:

    sudo pmset -a sleep 0 disablesleep 1

  되돌리려면:  sudo pmset -a sleep 10 disablesleep 0
MSG

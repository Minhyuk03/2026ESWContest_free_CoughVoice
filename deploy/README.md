# 상시 운영 배포

핫스팟 의존을 없애고 24시간 동작시키기 위한 설정. 2026-08-24 기준.

## 구성

```
집 WiFi
├── 맥미니 (mhui-Macmini.local)  서버 24시간 — launchd
├── 라즈베리파이 (mh)             엣지 상주 — systemd
└── 맥북                          개발 · 대시보드 접속
```

**IP가 아니라 mDNS 이름으로 연결한다.** 공유기가 IP를 바꿔도 끊기지 않는다.
핫스팟을 쓰던 때는 IP가 매번 달라져 그때마다 설정을 고쳐야 했다.

## 서버 (맥미니)

```bash
bash deploy/setup_server_mac.sh      # 저장소·venv·의존성
bash deploy/install_launchd.sh       # 자동 시작 등록
sudo pmset -a sleep 0 disablesleep 1 # 절전 해제 (관리자 권한 필요)
```

`projection.npz`(657KB)는 자동으로 받아지지 않는다. Coswara로 학습한 결과물이므로
개발 장비에서 복사한다.

```bash
# 개발 장비에서
scp ~/.cache/coughid/projection.npz <사용자>@mhui-Macmini.local:~/.cache/coughid/
```

없어도 서버는 동작하지만 **화자 식별 EER이 16.7% → 34.2%로 떨어진다.**

ECAPA와 PANNs 체크포인트(약 400MB)는 첫 요청 처리 시 자동으로 내려받는다.

### 대시보드

서버가 `dashboard/dist`를 직접 서빙한다. 별도 프로세스가 필요 없다.

    http://mhui-Macmini.local:8000/

같은 출처에서 열리므로 **로그인 화면의 서버 주소를 비워둬도 된다**. IP가 바뀌어도
화면이 열린 주소를 그대로 따라간다.

코드를 고친 뒤에는 다시 빌드해야 반영된다.

```bash
cd ~/srv/Cough_EmbeddedSystem/dashboard && npm run build
```

**주의**: 화면 경로 중 알림 센터는 `/alert-center`다. `/alerts`는 API 엔드포인트라
같은 서버에서 서빙하면 API가 먼저 잡혀 화면이 열리지 않는다.

## 엣지 (라즈베리파이)

```bash
# .local 해석이 간헐적으로 실패하므로 먼저 /etc/hosts에 박는다 (아래 설명 참조)
echo "192.168.219.134 mhui-Macmini.local mhui-macmini.local" | sudo tee -a /etc/hosts
sudo cp deploy/coughid-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now coughid-edge
```

**`/etc/hosts` 항목을 빠뜨리면 안 된다.** 이 환경에서 파이의 mDNS 해석이 간헐적으로
실패하는데(`Failed to resolve 'mhui-macmini.local'`), 그때마다 이벤트가 디스크 큐로
밀린다. 2026-08-25에 그 상태로 방치돼 큐가 4,314파일 177MB까지 자랐고 SD 여유가
977MB(93% 사용)까지 떨어졌다. `requests`가 호스트명을 소문자로 바꾸므로 대소문자
두 형태를 모두 넣는다.

큐 상태 점검:

```bash
ls ~/Cough_EmbeddedSystem/edge/queue/ | wc -l    # 정상이면 0에 가깝다
df -h /
```

큐는 500건에서 오래된 것부터 버리도록 상한이 걸려 있다(`event_sender.py`).
`queue/bad/`에 파일이 쌓이면 서버가 반복 거절한 항목이므로 내용을 확인해야 한다.

서버 주소는 유닛 파일의 `--server` 값이다. 서버 장비를 바꾸면 그 줄만 고친다.

```bash
systemctl status coughid-edge
journalctl -u coughid-edge -f
```

## 네트워크 메모

**파이를 집 WiFi에 붙이기** (핫스팟 프로필은 지우지 말 것 — 밖에서 쓸 때 백업이 된다)

```bash
sudo nmcli device wifi connect "<SSID>" password "<비밀번호>"
sudo nmcli connection modify "<SSID>" connection.autoconnect-priority 10
nmcli connection show
```

**`.local` 해석 주의.** 이 집 공유기(LG U+)의 ISP DNS가 실패한 조회에 항상 응답해서,
맥에서 `mh.local`을 물으면 mDNS 응답을 기다리지 않고 ISP가 준 엉뚱한 공인 IP를 쓴다.

- 파이 → 맥미니 (`mhui-Macmini.local`) : **정상**. avahi가 제대로 해석한다
- 맥 → 맥미니 (`mhui-Macmini.local`) : **정상**
- 맥 → 파이 (`mh.local`) : **가로채기 발생**. SSH 편의 문제이며 시스템 동작에는 영향 없다

맥에서 파이로 SSH가 필요하면 공유기에서 파이 MAC(`2c:cf:67:cd:c4:d7`)에 DHCP 예약을 걸고
`/etc/hosts`에 등록한다.

**AP 격리(AP isolation)** 가 켜져 있으면 같은 WiFi 안에서도 기기끼리 통신이 안 된다.
연결이 안 되면 공유기 설정에서 확인할 것.

## 확인

```bash
curl http://mhui-Macmini.local:8000/health        # {"status":"ok"}
journalctl -u coughid-edge -n 20                  # 엣지 로그
```

대시보드 로그인 화면의 서버 주소에 `http://mhui-Macmini.local:8000` 을 입력한다.

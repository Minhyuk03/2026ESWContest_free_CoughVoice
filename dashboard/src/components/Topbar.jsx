import { fmtElapsed, fmtDateTime } from '../lib/format'

/** 화면 상단 바.
 *
 *  디바이스 칩은 온·오프라인만이 아니라 **마지막 통신 시각과 상태 설명**을 함께 보인다.
 *  "오프라인"만 뜨면 언제부터인지, 무엇을 확인해야 하는지 알 수 없다.
 *  actions에는 각 화면이 갱신 제어 같은 버튼을 넣는다.
 */
export default function Topbar({ title, device, actions }) {
  const state = device
    ? (device.online ? 'on' : 'off')
    : null

  return (
    <header className="topbar">
      <h2>{title}</h2>
      <div className="topbar-right">
        {actions}
        {device && (
          <span className={`device-chip ${state}`} title={device.reason || ''}>
            <i className={`dot ${state}`} />
            <span className="device-chip-main">
              엣지 디바이스 {device.online ? '온라인' : '오프라인'}
            </span>
            <span className="device-chip-sub">
              {device.lastSeen
                ? `마지막 통신 ${fmtElapsed(device.secondsSinceSeen)}`
                : '통신 기록 없음'}
              {device.lastSeen && ` (${fmtDateTime(device.lastSeen)})`}
            </span>
          </span>
        )}
      </div>
    </header>
  )
}

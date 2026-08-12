export default function Topbar({ title, deviceOnline }) {
  return (
    <header className="topbar">
      <h2>{title}</h2>
      {deviceOnline !== undefined && (
        <span className="device-chip">
          엣지 디바이스 <i className={`dot ${deviceOnline ? 'on' : 'off'}`} /> {deviceOnline ? '온라인' : '오프라인'}
        </span>
      )}
    </header>
  )
}

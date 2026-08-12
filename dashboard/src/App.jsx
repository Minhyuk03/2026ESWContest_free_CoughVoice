import { useEffect, useRef, useState } from 'react'
import './App.css'

const STORAGE_KEY = 'cough-id-server-url'
const POLL_MS = 2000

function loadServerUrl() {
  return localStorage.getItem(STORAGE_KEY) || 'http://mh.local:8000'
}

function formatTime(iso) {
  return new Date(iso).toLocaleTimeString('ko-KR', { hour12: false })
}

function App() {
  const [serverUrl, setServerUrl] = useState(loadServerUrl)
  const [health, setHealth] = useState('checking') // checking | ok | down
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const pollRef = useRef(null)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, serverUrl)
  }, [serverUrl])

  useEffect(() => {
    async function poll() {
      const base = serverUrl.replace(/\/$/, '')
      try {
        const h = await fetch(`${base}/health`, { signal: AbortSignal.timeout(3000) })
        setHealth(h.ok ? 'ok' : 'down')
      } catch {
        setHealth('down')
      }
      try {
        const r = await fetch(`${base}/events?limit=30`, { signal: AbortSignal.timeout(3000) })
        if (r.ok) {
          setEvents(await r.json())
          setError(null)
        }
      } catch (e) {
        setError(e.message)
      }
    }

    poll()
    pollRef.current = setInterval(poll, POLL_MS)
    return () => clearInterval(pollRef.current)
  }, [serverUrl])

  return (
    <div id="app">
      <header>
        <h1>기침 이벤트 실시간 피드</h1>
        <div className="server-row">
          <span className={`dot ${health}`} title={health} />
          <input
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            placeholder="http://mh.local:8000"
            spellCheck={false}
          />
        </div>
        {error && <p className="error">연결 실패: {error}</p>}
      </header>

      <main>
        {events.length === 0 ? (
          <p className="empty">아직 감지된 이벤트가 없습니다.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>시간</th>
                <th>장치</th>
                <th>Peak RMS</th>
                <th>화자</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.id}>
                  <td>{formatTime(ev.captured_at)}</td>
                  <td>{ev.device_id}</td>
                  <td>{ev.peak_rms?.toFixed(3) ?? '-'}</td>
                  <td>{ev.person_id ?? 'unknown'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </main>
    </div>
  )
}

export default App

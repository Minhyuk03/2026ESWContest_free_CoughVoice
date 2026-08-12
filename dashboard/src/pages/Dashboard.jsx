import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'
import EventModal from '../components/EventModal'

const POLL_MS = 3000 // ≤ 3초 갱신 (NFR-03)

function speakerLabel(ev) {
  if (!ev.person_alias) return '미등록'
  return ev.person_room ? `${ev.person_alias} (${ev.person_room})` : ev.person_alias
}

function timeOf(iso) {
  const d = new Date(iso)
  const pad = (n) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

export default function Dashboard() {
  const [overview, setOverview] = useState(null)
  const [hourly, setHourly] = useState([])
  const [byPerson, setByPerson] = useState([])
  const [events, setEvents] = useState([])
  const [alerts, setAlerts] = useState([])
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    let alive = true
    async function poll() {
      try {
        const [ov, hr, bp, ev, al] = await Promise.all([
          api('/stats/overview'),
          api('/stats/hourly'),
          api('/stats/by-person'),
          api('/events?limit=8'),
          api('/alerts?limit=1'),
        ])
        if (!alive) return
        setOverview(ov)
        setHourly(hr.counts)
        setByPerson(bp)
        setEvents(ev)
        setAlerts(al)
      } catch { /* 다음 폴링에서 재시도 */ }
    }
    poll()
    const t = setInterval(poll, POLL_MS)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const maxHour = Math.max(1, ...hourly)
  const maxPerson = Math.max(1, ...byPerson.map((r) => r.count))
  const banner = alerts[0]

  return (
    <>
      <Topbar title="실시간 대시보드" deviceOnline={overview?.device_online} />
      <main className="page">
        {banner && (
          <div className="alert-banner">
            <span>
              ⚠ {banner.rule}: {banner.person_room ? `${banner.person_room} ` : ''}
              {banner.person_alias ? `${banner.person_alias}님 — ` : ''}{banner.message}
            </span>
            <Link to="/history">[이력 보기]</Link>
          </div>
        )}

        <div className="stat-grid">
          <div className="card stat">
            <p className="muted">오늘 기침 횟수</p>
            <p className="stat-num">{overview ? `${overview.today_cough_count}회` : '–'}</p>
          </div>
          <div className="card stat">
            <p className="muted">활성 알림</p>
            <p className="stat-num">{overview ? `${overview.active_alerts}건` : '–'}</p>
          </div>
          <div className="card stat">
            <p className="muted">등록 화자</p>
            <p className="stat-num">{overview ? `${overview.person_count}명` : '–'}</p>
          </div>
          <div className="card stat">
            <p className="muted">디바이스 상태</p>
            <p className="stat-num">{overview ? (overview.device_online ? '온라인' : '오프라인') : '–'}</p>
          </div>
        </div>

        <div className="dash-cols">
          <div className="dash-main">
            <div className="card">
              <h3>시간대별 기침 발생 추이 (24h)</h3>
              <div className="hour-chart">
                {hourly.map((n, i) => (
                  <div
                    key={i}
                    className="hour-bar"
                    style={{ height: `${(n / maxHour) * 100}%` }}
                    title={`${i}시 · ${n}회`}
                  />
                ))}
              </div>
              <div className="hour-axis">
                <span>00시</span><span>06시</span><span>12시</span><span>18시</span><span>24시</span>
              </div>
            </div>

            <div className="card">
              <h3>화자별 오늘 기침 현황</h3>
              <div className="person-rows">
                {byPerson.map((r) => (
                  <div key={r.person_id ?? 'unknown'} className="person-row">
                    <span className="person-name">
                      {r.room ? `${r.alias} (${r.room})` : r.alias}
                    </span>
                    <div className="person-bar-track">
                      <div className="person-bar" style={{ width: `${(r.count / maxPerson) * 100}%` }} />
                    </div>
                    <span className="person-count">{r.count}회</span>
                  </div>
                ))}
                {byPerson.length === 0 && <p className="muted">데이터가 없습니다.</p>}
              </div>
            </div>
          </div>

          <div className="card feed">
            <div className="feed-head">
              <h3>실시간 이벤트 피드</h3>
              <span className="muted small">≤ 3초 갱신</span>
            </div>
            {events.length === 0 && <p className="muted">아직 감지된 이벤트가 없습니다.</p>}
            {events.map((ev) => (
              <button type="button" key={ev.id} className="feed-item" onClick={() => setSelected(ev)}>
                <span className="feed-title">
                  {speakerLabel(ev)}
                  {!ev.person_alias && <em className="badge-unreg">미등록</em>}
                </span>
                <span className="muted small">
                  {timeOf(ev.captured_at)} · 신뢰도 {ev.similarity != null ? ev.similarity.toFixed(2) : '–'}
                </span>
              </button>
            ))}
          </div>
        </div>
      </main>
      {selected && <EventModal event={selected} onClose={() => setSelected(null)} />}
    </>
  )
}

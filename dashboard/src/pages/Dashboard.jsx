import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'
import EventModal from '../components/EventModal'
import StateBlock from '../components/StateBlock'
import Confidence from '../components/Confidence'
import {
  SEV_LABEL, alertHistoryLink, alertKindLabel, fmtDateTime, fmtTime, needsReview, speakerLabel,
} from '../lib/format'

const POLL_MS = 3000 // ≤ 3초 갱신 (NFR-03)

function hourLabel(i) {
  return `${String(i).padStart(2, '0')}시`
}

export default function Dashboard() {
  const [data, setData] = useState(null)      // 마지막으로 성공한 응답 묶음
  const [loading, setLoading] = useState(true)  // 첫 로딩인가
  const [error, setError] = useState(null)      // 마지막 갱신 실패 사유
  const [updatedAt, setUpdatedAt] = useState(null)
  const [paused, setPaused] = useState(false)
  const [selected, setSelected] = useState(null)
  const aliveRef = useRef(true)

  const poll = useCallback(async () => {
    try {
      const [ov, hr, bp, ev, al] = await Promise.all([
        api('/stats/overview'),
        api('/stats/hourly'),
        api('/stats/by-person'),
        api('/events?limit=8'),
        api('/alerts?limit=5'),
      ])
      if (!aliveRef.current) return
      setData({
        overview: ov,
        hourly: hr.counts,
        byPerson: bp,
        events: ev,
        // /alerts는 목록이 아니라 { items, disclaimer } 를 준다. 면책 문구를 화면이
        // 따로 들고 있지 않도록 서버가 함께 내려주는 구조다(P6).
        alerts: al.items || [],
        openCount: al.open_count ?? 0,
        disclaimer: al.disclaimer || '',
      })
      setError(null)
      setUpdatedAt(new Date())
    } catch (e) {
      // 이전 데이터는 지우지 않는다. 화면을 비우면 "기침이 없다"로 읽힌다.
      if (aliveRef.current) setError(e.message)
    } finally {
      if (aliveRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    aliveRef.current = true
    poll()
    return () => { aliveRef.current = false }
  }, [poll])

  useEffect(() => {
    if (paused) return undefined
    const t = setInterval(poll, POLL_MS)
    return () => clearInterval(t)
  }, [paused, poll])

  const overview = data?.overview
  const hourly = data?.hourly || []
  const byPerson = data?.byPerson || []
  const events = data?.events || []
  const alerts = data?.alerts || []
  const banner = alerts.find((a) => a.status === 'open') || alerts[0]

  const maxHour = Math.max(1, ...hourly)
  const maxPerson = Math.max(1, ...byPerson.map((r) => r.count))
  const hourTotal = hourly.reduce((a, b) => a + b, 0)
  const reviewCount = events.filter(needsReview).length

  const device = overview && {
    online: overview.device_online,
    reason: overview.device_status_reason,
    lastSeen: overview.device_last_seen,
    secondsSinceSeen: overview.device_seconds_since_seen,
  }

  const actions = (
    <span className="refresh-bar">
      <span className="muted small">
        {updatedAt ? `${fmtTime(updatedAt.toISOString())} 갱신` : '갱신 대기'}
        {paused ? ' · 일시정지' : ' · 3초마다'}
      </span>
      <button type="button" onClick={() => setPaused((v) => !v)}>
        {paused ? '자동 갱신 재개' : '일시정지'}
      </button>
      <button type="button" onClick={poll}>새로고침</button>
    </span>
  )

  return (
    <>
      <Topbar title="실시간 대시보드" device={device} actions={actions} />
      <main className="page">
        {error && (
          <div className="banner banner-error">
            <span>
              갱신 실패 — {error}.{' '}
              {updatedAt
                ? `아래 값은 ${fmtDateTime(updatedAt.toISOString())} 기준입니다.`
                : '아직 한 번도 데이터를 받지 못했습니다.'}
            </span>
            <button type="button" onClick={poll}>다시 시도</button>
          </div>
        )}

        {loading && !data && (
          <StateBlock kind="loading" title="데이터를 불러오는 중입니다"
                      detail="서버 응답을 기다리고 있습니다." />
        )}

        {banner && (
          <div className={`alert-banner sev-${banner.severity || 'info'}`}>
            <span className="alert-banner-main">
              <span className="kind-chip">{alertKindLabel(banner)}</span>
              {/* 대상은 서버 메시지 안에 이미 들어 있다. 여기서 또 붙이면
                  "301호 s01님 · 301호 s01 · …"처럼 같은 이름이 두 번 나온다. */}
              <b>[{SEV_LABEL[banner.severity] || '주의'}]</b> {banner.rule} — {banner.message}
            </span>
            <span className="alert-banner-side">
              <span className={`status-pill st-${banner.status}`}>{banner.status_label}</span>
              <Link to={alertHistoryLink(banner)}>해당 시간 이력 보기 →</Link>
              <Link to="/alert-center">알림 센터 →</Link>
            </span>
          </div>
        )}
        {data?.disclaimer && <p className="disclaimer">{data.disclaimer}</p>}

        <div className="stat-grid">
          <div className="card stat">
            <p className="stat-label">오늘 기침 횟수</p>
            <p className="stat-num">{overview ? overview.today_cough_count : '–'}<span className="stat-unit">회</span></p>
            <p className="muted small">
              {overview?.today_start_local
                ? `${fmtDateTime(overview.today_start_local)}부터 현재까지`
                : '오늘 00:00부터 현재까지'}
            </p>
          </div>
          <div className="card stat">
            <p className="stat-label">미확인 알림</p>
            <p className={`stat-num${data?.openCount > 0 ? ' stat-attn' : ''}`}>
              {data ? data.openCount : '–'}<span className="stat-unit">건</span>
            </p>
            {/* 두 숫자의 기준 구간이 다르다 — 미확인은 기간 제한 없이 남아 있는 전부,
                발생 건수는 최근 24시간이다. 나란히 두면 헷갈리므로 각각 기준을 적는다. */}
            <p className="muted small">
              아직 확인하지 않은 전체 건수 · 최근 {overview?.alerts_window_hours ?? 24}시간 발생{' '}
              {overview ? overview.active_alerts : '–'}건
            </p>
          </div>
          <div className="card stat">
            <p className="stat-label">등록 화자</p>
            <p className="stat-num">{overview ? overview.person_count : '–'}<span className="stat-unit">명</span></p>
            <p className="muted small">
              <Link to="/speakers">화자 관리에서 등록·수정</Link>
            </p>
          </div>
          <div className={`card stat ${overview && !overview.device_online ? 'stat-bad' : ''}`}>
            <p className="stat-label">디바이스 상태</p>
            <p className="stat-num stat-num-sm">
              {overview ? (overview.device_online ? '온라인' : '오프라인') : '–'}
            </p>
            <p className="muted small">{overview?.device_status_reason || '상태 확인 중'}</p>
          </div>
        </div>

        <div className="dash-cols">
          <div className="dash-main">
            <div className="card">
              <div className="card-head">
                <h3>시간대별 기침 발생 추이</h3>
                <span className="muted small">오늘 00시–24시 · 총 {hourTotal}회</span>
              </div>
              {hourTotal === 0 ? (
                <StateBlock
                  kind="empty"
                  title="오늘 기록된 기침이 없습니다"
                  detail="장치가 온라인인데도 계속 0회라면 마이크 위치와 감지 감도를 확인해 보세요."
                  action={<Link className="linklike" to="/history">지난 기록 보기 →</Link>}
                />
              ) : (
                <>
                  <div className="hour-chart">
                    {hourly.map((n, i) => {
                      const pct = (n / maxHour) * 100
                      return (
                        <div key={i} className="hour-col" title={`${hourLabel(i)} · ${n}회`}>
                          {/* 막대에 마우스를 올리면 그 시간대의 횟수를 바로 보여 준다.
                              막대 높이만으로는 몇 회인지 읽을 수 없다. */}
                          <span className={`hour-value${pct > 82 ? ' inside' : ''}`}
                                style={{ bottom: pct > 82 ? `calc(${pct}% - 24px)` : `calc(${pct}% + 6px)` }}>
                            {hourLabel(i)} <b>{n}회</b>
                          </span>
                          <div className={`hour-bar${n === 0 ? ' zero' : ''}`}
                               style={{ height: `${pct}%` }} />
                        </div>
                      )
                    })}
                  </div>
                  <div className="hour-axis">
                    <span>00시</span><span>06시</span><span>12시</span><span>18시</span><span>24시</span>
                  </div>
                </>
              )}
            </div>

            <div className="card">
              <div className="card-head">
                <h3>화자별 오늘 기침 현황</h3>
                <span className="muted small">오늘 00시부터 누적</span>
              </div>
              {byPerson.length === 0 ? (
                <StateBlock kind="empty" title="집계할 기록이 없습니다"
                            detail="화자를 등록하면 사람별로 나누어 볼 수 있습니다."
                            action={<Link className="linklike" to="/speakers/new">화자 등록하기 →</Link>} />
              ) : (
                <div className="person-rows">
                  {byPerson.map((r) => (
                    <div key={r.person_id ?? 'unknown'} className="person-row">
                      <span className="person-name">
                        {r.room ? `${r.alias} (${r.room})` : r.alias}
                      </span>
                      <div className="person-bar-track">
                        <div className={`person-bar${r.person_id ? '' : ' unknown'}`}
                             style={{ width: `${(r.count / maxPerson) * 100}%` }} />
                      </div>
                      <span className="person-count">{r.count}회</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="card feed">
            <div className="card-head">
              <h3>실시간 이벤트 피드</h3>
              <span className="muted small">최근 8건 · {paused ? '일시정지' : '3초 갱신'}</span>
            </div>
            {reviewCount > 0 && (
              <Link className="feed-review-link" to="/history?review=1">
                신뢰도가 낮아 검토가 필요한 이벤트 {reviewCount}건 →
              </Link>
            )}
            {events.length === 0 ? (
              <StateBlock kind="empty" title="아직 감지된 기침이 없습니다"
                          detail="장치 앞에서 기침하면 3초 안에 여기에 나타납니다." />
            ) : (
              <div className="feed-list">
                {events.map((ev) => (
                  <button type="button" key={ev.id} className="feed-item" onClick={() => setSelected(ev)}>
                    <span className="feed-title">
                      <span>{speakerLabel(ev)}</span>
                      {!ev.person_alias && <em className="badge-unreg">미등록</em>}
                      {needsReview(ev) && <em className="badge-review">검토 필요</em>}
                    </span>
                    <span className="feed-meta">
                      <span className="muted small">{fmtTime(ev.captured_at)}</span>
                      <Confidence value={ev.similarity} source={ev.person_source} />
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
      {selected && <EventModal event={selected} onClose={() => setSelected(null)} />}
    </>
  )
}

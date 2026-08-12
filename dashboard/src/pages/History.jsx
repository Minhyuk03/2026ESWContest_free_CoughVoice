import { useEffect, useMemo, useRef, useState } from 'react'
import { api, audioUrl } from '../api'
import Topbar from '../components/Topbar'
import EventModal from '../components/EventModal'

const PERIODS = [
  { key: '1', label: '오늘' },
  { key: '7', label: '최근 7일' },
  { key: '30', label: '최근 30일' },
]

function fmt(iso) {
  const d = new Date(iso)
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

export default function History() {
  const [events, setEvents] = useState([])
  const [persons, setPersons] = useState([])
  const [period, setPeriod] = useState('7')
  const [personFilter, setPersonFilter] = useState('all')
  const [includeUnknown, setIncludeUnknown] = useState(true)
  const [selected, setSelected] = useState(null)
  const playerRef = useRef(null)

  useEffect(() => {
    api('/events?limit=500').then(setEvents).catch(() => {})
    api('/persons').then(setPersons).catch(() => {})
  }, [])

  const filtered = useMemo(() => {
    const cutoff = Date.now() - Number(period) * 86400_000
    return events.filter((e) => {
      if (new Date(e.captured_at).getTime() < cutoff) return false
      if (personFilter === 'all') {
        if (!includeUnknown && !e.person_id) return false
        return true
      }
      return String(e.person_id) === personFilter
    })
  }, [events, period, personFilter, includeUnknown])

  // 일자별 집계 → SVG 라인 차트
  const trend = useMemo(() => {
    const days = Number(period)
    const buckets = new Array(days).fill(0)
    const start = new Date()
    start.setHours(0, 0, 0, 0)
    const startMs = start.getTime() - (days - 1) * 86400_000
    for (const e of filtered) {
      const idx = Math.floor((new Date(e.captured_at).getTime() - startMs) / 86400_000)
      if (idx >= 0 && idx < days) buckets[idx] += 1
    }
    return buckets
  }, [filtered, period])

  const maxTrend = Math.max(1, ...trend)
  const points = trend
    .map((n, i) => {
      const x = trend.length === 1 ? 50 : (i / (trend.length - 1)) * 100
      const y = 90 - (n / maxTrend) * 75
      return `${x},${y}`
    })
    .join(' ')

  function exportCsv() {
    const head = 'captured_at,speaker,room,similarity,status,device_id'
    const rows = filtered.map((e) =>
      [
        e.captured_at,
        e.person_alias || '',
        e.person_room || '',
        e.similarity ?? '',
        e.person_id ? '등록' : '미등록',
        e.device_id,
      ].join(','),
    )
    const blob = new Blob(['﻿' + [head, ...rows].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `cough_history_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  function play(ev) {
    if (playerRef.current) {
      playerRef.current.src = audioUrl(ev.id)
      playerRef.current.play().catch(() => {})
    }
  }

  return (
    <>
      <Topbar title="기침 이력 조회" />
      <main className="page">
        <div className="card filter-bar">
          <select value={period} onChange={(e) => setPeriod(e.target.value)}>
            {PERIODS.map((p) => <option key={p.key} value={p.key}>기간: {p.label}</option>)}
          </select>
          <select value={personFilter} onChange={(e) => setPersonFilter(e.target.value)}>
            <option value="all">화자: 전체</option>
            {persons.map((p) => (
              <option key={p.id} value={p.id}>{p.alias}{p.room ? ` (${p.room})` : ''}</option>
            ))}
          </select>
          <label className="check">
            <input
              type="checkbox"
              checked={includeUnknown}
              onChange={(e) => setIncludeUnknown(e.target.checked)}
              disabled={personFilter !== 'all'}
            />
            미등록 포함
          </label>
          <span className="spacer" />
          <button type="button" onClick={exportCsv}>CSV 내보내기</button>
        </div>

        <div className="card">
          <h3>화자별 일자별 기침 추이</h3>
          <svg className="trend-chart" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polyline points={points} fill="none" stroke="var(--sidebar-active)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          </svg>
          <p className="muted small">{PERIODS.find((p) => p.key === period)?.label} · 총 {filtered.length}건</p>
        </div>

        <div className="card table-card">
          <table className="data-table">
            <thead>
              <tr>
                <th>시각</th><th>화자</th><th>신뢰도</th><th>DOA(방향)</th><th>상태</th><th>오디오</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e) => (
                <tr key={e.id} onClick={() => setSelected(e)}>
                  <td>{fmt(e.captured_at)}</td>
                  <td>{e.person_alias ? `${e.person_alias}${e.person_room ? ` (${e.person_room})` : ''}` : '—'}</td>
                  <td>{e.similarity != null ? e.similarity.toFixed(2) : '–'}</td>
                  <td>–</td>
                  <td>
                    <span className={e.person_id ? 'tag-reg' : 'tag-unreg'}>
                      {e.person_id ? '등록' : '미등록'}
                    </span>
                  </td>
                  <td>
                    <button type="button" onClick={(ev) => { ev.stopPropagation(); play(e) }}>▶ 재생</button>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={6} className="muted">조건에 맞는 이벤트가 없습니다.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <audio ref={playerRef} preload="none" />
      </main>
      {selected && <EventModal event={selected} onClose={() => setSelected(null)} />}
    </>
  )
}

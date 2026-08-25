import { useEffect, useMemo, useRef, useState } from 'react'
import { api, audioUrl } from '../api'
import Topbar from '../components/Topbar'
import EventModal from '../components/EventModal'

const PERIODS = [
  { key: '1', label: '오늘' },
  { key: '7', label: '최근 7일' },
  { key: '30', label: '최근 30일' },
]

// 화자별 라인 색. 색각 이상(적록·청황)에도 구분되는 Okabe-Ito 계열에서
// 흰 배경 가독이 낮은 노랑을 빼고 골랐다. persons 순서대로 배정한다.
const SERIES_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#8064A2']
const UNKNOWN_COLOR = '#9aa0ad' // 미등록은 중립 회색 — 특정 화자와 헷갈리지 않게

const PAGE_SIZE = 20 // 이력 표 한 페이지에 보여줄 행 수

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
  const [page, setPage] = useState(1)
  const playerRef = useRef(null)

  useEffect(() => {
    api('/events?limit=500').then(setEvents).catch(() => {})
    api('/persons').then(setPersons).catch(() => {})
  }, [])

  // 화자 id → 색. persons 순서를 고정해 필터를 바꿔도 같은 화자는 같은 색을 쓴다.
  const colorOf = useMemo(() => {
    const m = {}
    persons.forEach((p, i) => { m[String(p.id)] = SERIES_COLORS[i % SERIES_COLORS.length] })
    return m
  }, [persons])

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

  // 페이지네이션 — 필터가 바뀌면 1페이지로 되돌린다
  useEffect(() => { setPage(1) }, [period, personFilter, includeUnknown])
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const curPage = Math.min(page, pageCount)
  const pageStart = (curPage - 1) * PAGE_SIZE
  const pageItems = filtered.slice(pageStart, pageStart + PAGE_SIZE)

  // 화자별·일자별 집계 → 화자마다 하나의 라인
  const { series, days } = useMemo(() => {
    const days = Number(period)
    const start = new Date()
    start.setHours(0, 0, 0, 0)
    const startMs = start.getTime() - (days - 1) * 86400_000

    // 그릴 화자 목록(필터 반영). person이 없으면 '미등록' 묶음.
    const keys = []
    if (personFilter === 'all') {
      persons.forEach((p) => keys.push({
        id: String(p.id),
        label: `${p.alias}${p.room ? ` (${p.room})` : ''}`,
        color: colorOf[String(p.id)],
      }))
      if (includeUnknown) keys.push({ id: 'unknown', label: '미등록', color: UNKNOWN_COLOR })
    } else {
      const p = persons.find((pp) => String(pp.id) === personFilter)
      keys.push({
        id: personFilter,
        label: p ? p.alias : personFilter,
        color: p ? colorOf[personFilter] : UNKNOWN_COLOR,
      })
    }

    const map = {}
    keys.forEach((k) => { map[k.id] = new Array(days).fill(0) })
    for (const e of filtered) {
      const idx = Math.floor((new Date(e.captured_at).getTime() - startMs) / 86400_000)
      if (idx < 0 || idx >= days) continue
      const key = e.person_id ? String(e.person_id) : 'unknown'
      if (map[key]) map[key][idx] += 1
    }

    // 이 기간에 기침이 없는 화자는 라인·범례에서 숨긴다.
    const series = keys
      .map((k) => ({ ...k, values: map[k.id], total: map[k.id].reduce((a, b) => a + b, 0) }))
      .filter((s) => s.total > 0)
    return { series, days }
  }, [filtered, period, personFilter, includeUnknown, persons, colorOf])

  // 차트 좌표계 (고정 픽셀 → viewBox 균일 스케일이라 점이 원으로 유지된다)
  const W = 600, H = 200, padL = 10, padR = 10, padT = 12, padB = 22
  const plotW = W - padL - padR
  const plotH = H - padT - padB
  const maxY = Math.max(1, ...series.flatMap((s) => s.values))
  const xAt = (i) => (days === 1 ? W / 2 : padL + (i / (days - 1)) * plotW)
  const yAt = (v) => padT + plotH - (v / maxY) * plotH
  const showDots = days <= 7 // 30일은 점이 많아 지저분하므로 라인만

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
          {series.length === 0 ? (
            <p className="muted small">이 기간에 표시할 기침 이벤트가 없습니다.</p>
          ) : (
            <>
              <svg className="trend-chart" viewBox={`0 0 ${W} ${H}`} role="img"
                   aria-label="화자별 일자별 기침 추이">
                {/* 기준선(0) */}
                <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH}
                      stroke="var(--line)" strokeWidth="1" />
                {series.map((s) => (
                  <g key={s.id}>
                    {days > 1 && (
                      <polyline
                        points={s.values.map((v, i) => `${xAt(i)},${yAt(v)}`).join(' ')}
                        fill="none" stroke={s.color} strokeWidth="2"
                        strokeLinejoin="round" strokeLinecap="round"
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                    {showDots && s.values.map((v, i) => (
                      (v > 0 || days === 1) ? (
                        <circle key={i} cx={xAt(i)} cy={yAt(v)} r="3.5" fill={s.color} />
                      ) : null
                    ))}
                  </g>
                ))}
              </svg>
              <div className="chart-legend">
                {series.map((s) => (
                  <span key={s.id} className="legend-item">
                    <span className="legend-swatch" style={{ background: s.color }} />
                    {s.label} <b>{s.total}</b>
                  </span>
                ))}
              </div>
            </>
          )}
          <p className="muted small">{PERIODS.find((p) => p.key === period)?.label} · 총 {filtered.length}건</p>
        </div>

        <div className="card table-card">
          <table className="data-table zebra">
            <thead>
              <tr>
                <th className="row-num">#</th>
                <th>시각</th><th>화자</th><th>신뢰도</th><th>DOA(방향)</th><th>상태</th><th>오디오</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((e, i) => (
                <tr key={e.id} onClick={() => setSelected(e)}>
                  <td className="row-num">{pageStart + i + 1}</td>
                  <td>{fmt(e.captured_at)}</td>
                  <td>
                    {e.person_alias ? (
                      <span className="speaker-cell">
                        <span className="legend-swatch"
                              style={{ background: colorOf[String(e.person_id)] || UNKNOWN_COLOR }} />
                        {e.person_alias}{e.person_room ? ` (${e.person_room})` : ''}
                      </span>
                    ) : '—'}
                  </td>
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
                <tr><td colSpan={7} className="muted">조건에 맞는 이벤트가 없습니다.</td></tr>
              )}
            </tbody>
          </table>
          {filtered.length > PAGE_SIZE && (
            <div className="pagination">
              <button type="button" disabled={curPage <= 1} onClick={() => setPage(curPage - 1)}>‹ 이전</button>
              <span>{curPage} / {pageCount} 페이지 · 총 {filtered.length}건</span>
              <button type="button" disabled={curPage >= pageCount} onClick={() => setPage(curPage + 1)}>다음 ›</button>
            </div>
          )}
        </div>
        <audio ref={playerRef} preload="none" />
      </main>
      {selected && <EventModal event={selected} onClose={() => setSelected(null)} />}
    </>
  )
}

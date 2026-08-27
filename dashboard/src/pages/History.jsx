import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, audioUrl } from '../api'
import Topbar from '../components/Topbar'
import EventModal from '../components/EventModal'
import StateBlock from '../components/StateBlock'
import Confidence from '../components/Confidence'
import { REVIEW_BELOW, confidence, fmtDateTime, needsReview } from '../lib/format'

const PERIODS = [
  { key: '1', label: '오늘' },
  { key: '7', label: '최근 7일' },
  { key: '30', label: '최근 30일' },
]

const PAGE_SIZES = [20, 50, 100]

const HOUR_MS = 3_600_000
const DAY_MS = 86_400_000

/** 구간 길이에 맞는 집계 칸 크기. 칸이 너무 잘면 선이 톱니가 되고,
 *  너무 굵으면 하루 안의 분포가 사라진다. */
function pickBucketMs(spanMs) {
  if (spanMs <= 6 * HOUR_MS) return 15 * 60_000
  if (spanMs <= 2 * DAY_MS) return HOUR_MS
  if (spanMs <= 10 * DAY_MS) return 3 * HOUR_MS
  return 12 * HOUR_MS
}

/** 눈금 간격. 구간이 짧으면 날짜 하나만 남아 축이 비어 보이므로 시각까지 찍는다. */
function pickTickMs(spanMs) {
  if (spanMs <= 3 * HOUR_MS) return 30 * 60_000
  if (spanMs <= 8 * HOUR_MS) return HOUR_MS
  if (spanMs <= 2 * DAY_MS) return 6 * HOUR_MS
  return DAY_MS
}

/** 칸 하나가 덮는 시각 구간을 "8/28 00:15 ~ 00:30"처럼 적는다. */
function bucketRangeText(startMs, bucketMs) {
  const a = new Date(startMs)
  const b = new Date(startMs + bucketMs)
  const hhmm = (d) => `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  const day = (d) => `${d.getMonth() + 1}/${d.getDate()}`
  const tail = day(a) === day(b) ? hhmm(b) : `${day(b)} ${hhmm(b)}`
  return `${day(a)} ${hhmm(a)} ~ ${tail}`
}

function bucketText(ms) {
  if (ms < HOUR_MS) return `${ms / 60_000}분`
  if (ms < DAY_MS) return `${ms / HOUR_MS}시간`
  return `${ms / DAY_MS}일`
}

// 화자별 라인 색. 색각 이상(적록·청황)에도 구분되는 Okabe-Ito 계열에서
// 흰 배경 가독이 낮은 노랑을 빼고 골랐다. persons 순서대로 배정한다.
const SERIES_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#8064A2']
const UNKNOWN_COLOR = '#8a8f9c' // 미등록은 중립 회색 — 특정 화자와 헷갈리지 않게

const SORTS = {
  captured_at: (e) => new Date(e.captured_at).getTime(),
  similarity: (e) => (e.similarity == null ? -1 : e.similarity),
  person: (e) => (e.person_alias || '￿'), // 미등록은 항상 끝으로
}

export default function History() {
  const [params, setParams] = useSearchParams()

  const [events, setEvents] = useState([])
  const [persons, setPersons] = useState([])
  const [policy, setPolicy] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [period, setPeriod] = useState('7')
  const [personFilter, setPersonFilter] = useState(params.get('person') || 'all')
  const [includeUnknown, setIncludeUnknown] = useState(true)
  const [reviewOnly, setReviewOnly] = useState(params.get('review') === '1')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState({ key: 'captured_at', dir: 'desc' })
  const [pageSize, setPageSize] = useState(20)
  const [page, setPage] = useState(1)
  const [picked, setPicked] = useState(() => new Set())
  const [linkTo, setLinkTo] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState(null)
  const [selected, setSelected] = useState(null)
  const [hover, setHover] = useState(null)   // 차트 위 커서 위치(칸 번호)
  const playerRef = useRef(null)

  // 알림에서 넘어온 시간 구간. 있으면 기간 선택 대신 이 구간으로 좁힌다.
  const from = params.get('from')
  const to = params.get('to')
  const fromMs = from ? new Date(from).getTime() : null
  const toMs = to ? new Date(to).getTime() : null

  const load = useCallback(() => {
    setLoading(true)
    Promise.all([api('/events?limit=500'), api('/persons')])
      .then(([ev, ps]) => { setEvents(ev); setPersons(ps); setError(null) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(load, [load])
  // 원음 보존 정책은 화면이 "무엇을 재생하는지" 정확히 설명하기 위해 서버에서 받는다.
  useEffect(() => { api('/audio-policy').then(setPolicy).catch(() => {}) }, [])

  // 화자 id → 색. persons 순서를 고정해 필터를 바꿔도 같은 화자는 같은 색을 쓴다.
  const colorOf = useMemo(() => {
    const m = {}
    persons.forEach((p, i) => { m[String(p.id)] = SERIES_COLORS[i % SERIES_COLORS.length] })
    return m
  }, [persons])

  const filtered = useMemo(() => {
    const cutoff = Date.now() - Number(period) * 86400_000
    const q = query.trim().toLowerCase()
    return events.filter((e) => {
      const t = new Date(e.captured_at).getTime()
      if (fromMs != null || toMs != null) {
        if (fromMs != null && t < fromMs) return false
        if (toMs != null && t > toMs) return false
      } else if (t < cutoff) {
        return false
      }
      if (personFilter === 'unknown') {
        if (e.person_id) return false
      } else if (personFilter !== 'all') {
        if (String(e.person_id) !== personFilter) return false
      } else if (!includeUnknown && !e.person_id) {
        return false
      }
      if (reviewOnly && !needsReview(e)) return false
      if (q) {
        const hay = [
          e.person_alias || '미등록', e.person_room || '', e.device_id || '',
          fmtDateTime(e.captured_at),
        ].join(' ').toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [events, period, personFilter, includeUnknown, reviewOnly, query, fromMs, toMs])

  const sorted = useMemo(() => {
    const get = SORTS[sort.key] || SORTS.captured_at
    const rows = [...filtered]
    rows.sort((a, b) => {
      const va = get(a), vb = get(b)
      if (va < vb) return sort.dir === 'asc' ? -1 : 1
      if (va > vb) return sort.dir === 'asc' ? 1 : -1
      return 0
    })
    return rows
  }, [filtered, sort])

  // 페이지네이션 — 필터가 바뀌면 1페이지로 되돌린다
  useEffect(() => { setPage(1) },
    [period, personFilter, includeUnknown, reviewOnly, query, pageSize, from, to])
  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize))
  const curPage = Math.min(page, pageCount)
  const pageStart = (curPage - 1) * pageSize
  const pageItems = sorted.slice(pageStart, pageStart + pageSize)

  const pickedRows = useMemo(() => sorted.filter((e) => picked.has(e.id)), [sorted, picked])
  const allPagePicked = pageItems.length > 0 && pageItems.every((e) => picked.has(e.id))

  /* 시각 축 집계 — 하루 한 칸이 아니라 **시간까지 반영한** 칸으로 센다.
     하루 한 점만 찍으면 '오늘'을 보는 순간 점 하나가 되어 아무것도 읽히지 않고,
     7일을 봐도 하루 안에서 언제 몰렸는지가 사라진다. 구간 길이에 따라 칸 크기를
     정하고, 눈금은 날짜(자정) 자리에 놓아 어느 날인지 바로 보이게 한다. */
  const chart = useMemo(() => {
    const now = Date.now()
    let start
    let end = now
    if (fromMs != null || toMs != null) {
      // 알림에서 넘어온 구간이면 차트도 그 구간을 그린다. 표는 구간만 보여 주는데
      // 차트만 '최근 7일'을 그리면 두 값이 어긋나 보인다.
      start = fromMs ?? now - Number(period) * DAY_MS
      end = Math.min(toMs ?? now, now)
    } else {
      const midnight = new Date()
      midnight.setHours(0, 0, 0, 0)
      start = midnight.getTime() - (Number(period) - 1) * DAY_MS
    }
    if (end <= start) end = start + HOUR_MS

    const bucketMs = pickBucketMs(end - start)
    // 칸 경계를 그날 자정에 맞춘다. 00:37 같은 데서 끊기면 눈금과 칸이 어긋난다.
    const dayZero = new Date(start)
    dayZero.setHours(0, 0, 0, 0)
    const alignedStart =
      dayZero.getTime() + Math.floor((start - dayZero.getTime()) / bucketMs) * bucketMs
    const count = Math.max(2, Math.ceil((end - alignedStart) / bucketMs))
    const span = count * bucketMs

    // 그릴 화자 목록(필터 반영). person이 없으면 '미등록' 묶음.
    const keys = []
    if (personFilter === 'all') {
      persons.forEach((p) => keys.push({
        id: String(p.id),
        label: `${p.alias}${p.room ? ` (${p.room})` : ''}`,
        color: colorOf[String(p.id)],
      }))
      if (includeUnknown) keys.push({ id: 'unknown', label: '미등록', color: UNKNOWN_COLOR })
    } else if (personFilter === 'unknown') {
      keys.push({ id: 'unknown', label: '미등록', color: UNKNOWN_COLOR })
    } else {
      const p = persons.find((pp) => String(pp.id) === personFilter)
      keys.push({
        id: personFilter,
        label: p ? p.alias : personFilter,
        color: p ? colorOf[personFilter] : UNKNOWN_COLOR,
      })
    }

    const map = {}
    keys.forEach((k) => { map[k.id] = new Array(count).fill(0) })
    for (const e of filtered) {
      const idx = Math.floor((new Date(e.captured_at).getTime() - alignedStart) / bucketMs)
      if (idx < 0 || idx >= count) continue
      const key = e.person_id ? String(e.person_id) : 'unknown'
      if (map[key]) map[key][idx] += 1
    }

    // 이 기간에 기침이 없는 화자는 라인·범례에서 숨긴다.
    const series = keys
      .map((k) => ({ ...k, values: map[k.id], total: map[k.id].reduce((a, b) => a + b, 0) }))
      .filter((s) => s.total > 0)

    // 눈금 — 자정에서 출발해 일정 간격으로 찍는다. 날짜 경계가 항상 눈금에 오도록
    // 자정 기준으로 세는 것이 핵심이다(눈금이 03:00, 09:00…으로 밀리면 날짜를 못 읽는다).
    const tickMs = pickTickMs(span)
    const ticks = []
    const cursor = new Date(alignedStart)
    cursor.setHours(0, 0, 0, 0)
    while (cursor.getTime() < alignedStart) {
      if (tickMs >= DAY_MS) cursor.setDate(cursor.getDate() + 1)
      else cursor.setTime(cursor.getTime() + tickMs)
    }
    while (cursor.getTime() <= alignedStart + span) {
      ticks.push(cursor.getTime())
      if (tickMs >= DAY_MS) cursor.setDate(cursor.getDate() + 1)
      else cursor.setTime(cursor.getTime() + tickMs)
    }
    // 눈금이 빽빽하면 글자가 겹친다 — 7개 안쪽으로 솎는다.
    const step = Math.ceil(ticks.length / 7)
    const thinned = step > 1 ? ticks.filter((_, i) => i % step === 0) : ticks

    return { series, bucketMs, alignedStart, span, count, ticks: thinned, tickMs: tickMs * step }
  }, [filtered, period, personFilter, includeUnknown, persons, colorOf, fromMs, toMs])

  const { series, bucketMs, alignedStart, span, count } = chart

  // 차트 좌표계 (고정 픽셀 → viewBox 균일 스케일이라 점이 원으로 유지된다)
  const W = 600, H = 200, padL = 26, padR = 12, padT = 12, padB = 26
  const plotW = W - padL - padR
  const plotH = H - padT - padB
  const maxY = Math.max(1, ...series.flatMap((s) => s.values))
  // 시각 → x. 칸의 가운데에 찍는다(칸이 덮는 구간의 대표값이므로).
  const xAtTime = (ms) => padL + ((ms - alignedStart) / span) * plotW
  const xAt = (i) => xAtTime(alignedStart + (i + 0.5) * bucketMs)
  const yAt = (v) => padT + plotH - (v / maxY) * plotH
  const showDots = count <= 40   // 칸이 많으면 점이 뭉개지므로 선만 그린다

  function tickLabel(ms) {
    const d = new Date(ms)
    // 자정 눈금은 날짜로 적는다 — 어느 날인지가 먼저 보여야 한다.
    if (chart.tickMs >= DAY_MS || (d.getHours() === 0 && d.getMinutes() === 0)) {
      return `${d.getMonth() + 1}/${d.getDate()}`
    }
    const hh = String(d.getHours()).padStart(2, '0')
    return chart.tickMs % HOUR_MS === 0
      ? `${hh}시`
      : `${hh}:${String(d.getMinutes()).padStart(2, '0')}`
  }

  /* 커서를 따라다니는 세로 점선과 값 표시.
     화면 좌표를 viewBox 좌표로 되돌려 칸 번호를 구한다 — SVG가 폭에 맞춰 늘어나므로
     픽셀을 그대로 쓰면 넓은 화면에서 어긋난다. */
  function onChartMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    if (!rect.width) return
    const scale = rect.width / W
    const x = (e.clientX - rect.left) / scale
    const y = (e.clientY - rect.top) / scale
    const raw = Math.round(((x - padL) / plotW) * count - 0.5)
    const idx = Math.min(count - 1, Math.max(0, raw))
    // 커서에 가장 가까운 선을 굵게 표시한다 — 선이 겹칠 때 어느 화자를 보는지 알려준다.
    let near = null
    let best = Infinity
    for (const sr of series) {
      const d = Math.abs(yAt(sr.values[idx]) - y)
      if (d < best) { best = d; near = sr.id }
    }
    setHover({ idx, scale, width: rect.width, near: best <= 14 ? near : null })
  }

  function toggleSort(key) {
    setSort((s) => (s.key === key
      ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' }
      : { key, dir: key === 'person' ? 'asc' : 'desc' }))
  }

  function togglePick(id) {
    setPicked((cur) => {
      const next = new Set(cur)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  function togglePagePick() {
    setPicked((cur) => {
      const next = new Set(cur)
      if (allPagePicked) pageItems.forEach((e) => next.delete(e.id))
      else pageItems.forEach((e) => next.add(e.id))
      return next
    })
  }

  function exportCsv() {
    const rows = pickedRows.length > 0 ? pickedRows : sorted
    const head = '시각,화자,호실,화자지정,신뢰도(%),신뢰도등급,상태,기기,소리보관'
    const lines = rows.map((e) => {
      const c = confidence(e.similarity)
      return [
        fmtDateTime(e.captured_at),
        e.person_alias || '미등록',
        e.person_room || '',
        e.person_source === 'manual' ? '사람이 지정' : '모델 판정',
        c.pct ?? '',
        c.pct == null ? '비교 없음' : c.label,
        e.person_id ? '등록 화자' : '미등록',
        e.device_id,
        e.audio_available ? '보관 중' : '삭제됨',
      ].join(',')
    })
    const blob = new Blob(['﻿' + [head, ...lines].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `cough_history_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
    setNotice(`${rows.length}건을 CSV로 내려받았습니다.`)
  }

  async function linkSelected() {
    if (pickedRows.length === 0 || linkTo === '') return
    const target = persons.find((p) => String(p.id) === linkTo)
    const ok = window.confirm(
      `선택한 ${pickedRows.length}건을 ${target ? target.alias : '미등록'}(으)로 바꿉니다.\n\n` +
      '이력과 통계의 표시만 바뀌고, 다음 기침을 같은 사람으로 알아보게 하려면 ' +
      '화자 관리에서 재등록으로 목소리 특징을 갱신해야 합니다. 진행할까요?')
    if (!ok) return
    setBusy(true)
    try {
      await api('/events/assign', {
        method: 'POST',
        body: {
          event_ids: pickedRows.map((e) => e.id),
          person_id: linkTo === 'none' ? null : Number(linkTo),
        },
      })
      setPicked(new Set())
      setNotice(`${pickedRows.length}건의 화자를 바꿨습니다. 재등록으로 특징을 갱신하면 다음부터 자동 인식됩니다.`)
      load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  function play(ev) {
    if (playerRef.current) {
      playerRef.current.src = audioUrl(ev.id)
      playerRef.current.play().catch(() => setNotice('소리를 재생할 수 없습니다.'))
    }
  }

  function clearRange() {
    const next = new URLSearchParams(params)
    next.delete('from'); next.delete('to'); next.delete('alert')
    setParams(next, { replace: true })
  }

  const rangeText = fromMs != null || toMs != null
    ? `${from ? fmtDateTime(from) : '처음'} ~ ${to ? fmtDateTime(to) : '지금'}`
    : PERIODS.find((p) => p.key === period)?.label

  const sortMark = (key) => (sort.key === key ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : '')

  return (
    <>
      <Topbar title="기침 이력 조회" />
      <main className="page">
        {(fromMs != null || toMs != null) && (
          <div className="banner banner-info">
            <span>
              알림에서 넘어온 구간만 보고 있습니다 — <b>{rangeText}</b>
              {params.get('person') === 'unknown' && ' · 미등록 화자'}
            </span>
            <button type="button" onClick={clearRange}>구간 해제</button>
          </div>
        )}
        {error && (
          <div className="banner banner-error">
            <span>이력을 불러오지 못했습니다 — {error}</span>
            <button type="button" onClick={load}>다시 시도</button>
          </div>
        )}
        {notice && (
          <div className="banner banner-info">
            <span>{notice}</span>
            <button type="button" onClick={() => setNotice(null)}>닫기</button>
          </div>
        )}

        <div className="card filter-bar">
          <label className="field">
            <span className="field-label">기간</span>
            <select value={period} onChange={(e) => setPeriod(e.target.value)}
                    disabled={fromMs != null || toMs != null}>
              {PERIODS.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
            </select>
          </label>
          <label className="field">
            <span className="field-label">화자</span>
            <select value={personFilter} onChange={(e) => setPersonFilter(e.target.value)}>
              <option value="all">전체</option>
              <option value="unknown">미등록만</option>
              {persons.map((p) => (
                <option key={p.id} value={p.id}>{p.alias}{p.room ? ` (${p.room})` : ''}</option>
              ))}
            </select>
          </label>
          <label className="field field-grow">
            <span className="field-label">검색</span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="화자·호실·기기·날짜로 검색"
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={includeUnknown}
              onChange={(e) => setIncludeUnknown(e.target.checked)}
              disabled={personFilter !== 'all'}
            />
            미등록 포함
          </label>
          <label className="check">
            <input type="checkbox" checked={reviewOnly}
                   onChange={(e) => setReviewOnly(e.target.checked)} />
            검토 필요만 (신뢰도 {Math.round(REVIEW_BELOW * 100)}% 미만)
          </label>
          <span className="spacer" />
          <button type="button" onClick={load}>새로고침</button>
          <button type="button" onClick={exportCsv} disabled={sorted.length === 0}>
            CSV 내보내기 ({pickedRows.length > 0 ? `선택 ${pickedRows.length}건` : `전체 ${sorted.length}건`})
          </button>
        </div>

        <div className="card">
          <div className="card-head">
            <h3>화자별 기침 발생 추이</h3>
            <span className="muted small">
              {rangeText} · 총 {filtered.length}건 · {bucketText(bucketMs)} 단위 집계
            </span>
          </div>
          {loading ? (
            <StateBlock kind="loading" title="이력을 불러오는 중입니다" />
          ) : series.length === 0 ? (
            <StateBlock
              kind="empty"
              title="이 조건에 해당하는 기침이 없습니다"
              detail="기간을 넓히거나 화자 필터를 전체로 바꿔 보세요. 장치가 오프라인이었다면 그 구간은 기록이 남지 않습니다."
              action={<button type="button" onClick={() => { setPeriod('30'); setPersonFilter('all'); setQuery('') }}>
                최근 30일 · 전체 화자로 보기
              </button>}
            />
          ) : (
            <>
              <div className="chart-wrap">
              <svg className="trend-chart" viewBox={`0 0 ${W} ${H}`} role="img"
                   aria-label="화자별 기침 발생 추이"
                   onPointerMove={onChartMove}
                   onPointerDown={onChartMove}
                   onPointerLeave={() => setHover(null)}>
                {/* 가로 눈금 — 최댓값과 절반. 선의 높이가 몇 회인지 알 수 있어야 한다 */}
                {[maxY, maxY / 2].map((v) => (
                  <g key={v}>
                    <line x1={padL} y1={yAt(v)} x2={W - padR} y2={yAt(v)}
                          stroke="var(--line)" strokeWidth="1" strokeDasharray="3 3" />
                    <text x={padL - 5} y={yAt(v) + 3.5} textAnchor="end"
                          fontSize="9" fill="var(--muted)">
                      {Math.round(v)}
                    </text>
                  </g>
                ))}

                {/* 세로 눈금 — 날짜(자정) 자리. 어느 날의 값인지 바로 보이게 한다 */}
                {chart.ticks.map((t) => (
                  <g key={t}>
                    <line x1={xAtTime(t)} y1={padT} x2={xAtTime(t)} y2={padT + plotH}
                          stroke="var(--line)" strokeWidth="1" />
                    <text x={xAtTime(t)} y={H - 8} textAnchor="middle"
                          fontSize="10" fill="var(--muted)">
                      {tickLabel(t)}
                    </text>
                  </g>
                ))}

                {/* 기준선(0) */}
                <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH}
                      stroke="var(--muted)" strokeWidth="1" />

                {series.map((s) => (
                  <g key={s.id}>
                    <polyline
                      points={s.values.map((v, i) => `${xAt(i)},${yAt(v)}`).join(' ')}
                      fill="none" stroke={s.color}
                      strokeWidth={hover?.near === s.id ? 3 : 2}
                      strokeLinejoin="round" strokeLinecap="round"
                      vectorEffect="non-scaling-stroke"
                    />
                    {showDots && s.values.map((v, i) => (
                      v > 0 ? <circle key={i} cx={xAt(i)} cy={yAt(v)} r="3" fill={s.color} /> : null
                    ))}
                  </g>
                ))}

                {/* 커서 위치 — 세로 점선과 그 칸의 각 화자 값 */}
                {hover && (
                  <g pointerEvents="none">
                    <line x1={xAt(hover.idx)} y1={padT} x2={xAt(hover.idx)} y2={padT + plotH}
                          stroke="var(--muted)" strokeWidth="1" strokeDasharray="4 3" />
                    {series.map((s) => (
                      <circle key={s.id} cx={xAt(hover.idx)} cy={yAt(s.values[hover.idx])}
                              r={hover.near === s.id ? 5 : 4}
                              fill="var(--card)" stroke={s.color} strokeWidth="2" />
                    ))}
                  </g>
                )}
              </svg>

              {hover && (() => {
                // 툴팁은 커서가 가리키는 칸 옆에 놓되, 카드 밖으로 새지 않게 붙잡는다.
                const half = 84
                const left = Math.min(Math.max(xAt(hover.idx) * hover.scale, half),
                                      Math.max(half, hover.width - half))
                const topPx = Math.min(...series.map((s) => yAt(s.values[hover.idx]))) * hover.scale
                // 위쪽에 자리가 없으면(봉우리가 천장에 붙었을 때) 점 아래로 뒤집는다.
                const needed = 34 + series.length * 17
                const below = topPx < needed
                const total = series.reduce((a, s) => a + s.values[hover.idx], 0)
                return (
                  <div className={`chart-tip${below ? ' below' : ''}`}
                       style={{ left, top: below ? topPx + 14 : topPx - 12 }}>
                    <p className="tip-time">
                      {bucketRangeText(alignedStart + hover.idx * bucketMs, bucketMs)}
                    </p>
                    {series.map((s) => (
                      <p key={s.id}
                         className={`tip-row${hover.near === s.id ? ' on' : ''}`}>
                        <span className="legend-swatch" style={{ background: s.color }} />
                        <span className="tip-name">{s.label}</span>
                        <b>{s.values[hover.idx]}회</b>
                      </p>
                    ))}
                    {series.length > 1 && <p className="tip-total">합계 {total}회</p>}
                  </div>
                )
              })()}
              </div>
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
        </div>

        {pickedRows.length > 0 && (
          <div className="card select-bar">
            <span><b>{pickedRows.length}건</b> 선택됨</span>
            <label className="field">
              <span className="field-label">화자 연결</span>
              <select value={linkTo} onChange={(e) => setLinkTo(e.target.value)}>
                <option value="">화자 선택…</option>
                {persons.map((p) => (
                  <option key={p.id} value={p.id}>{p.alias}{p.room ? ` (${p.room})` : ''}</option>
                ))}
                <option value="none">미등록으로 되돌리기</option>
              </select>
            </label>
            <button type="button" className="primary" disabled={linkTo === '' || busy}
                    onClick={linkSelected}>
              {busy ? '적용 중…' : '선택 이벤트에 적용'}
            </button>
            <button type="button" onClick={exportCsv}>선택 항목 CSV</button>
            <span className="spacer" />
            <button type="button" onClick={() => setPicked(new Set())}>선택 해제</button>
          </div>
        )}

        <div className="card table-card">
          <div className="table-scroll">
            <table className="data-table zebra">
              <thead>
                <tr>
                  <th className="col-check">
                    <input type="checkbox" checked={allPagePicked} onChange={togglePagePick}
                           aria-label="이 페이지 전체 선택" />
                  </th>
                  <th className="row-num">#</th>
                  <th>
                    <button type="button" className="th-sort" onClick={() => toggleSort('captured_at')}>
                      시각{sortMark('captured_at')}
                    </button>
                  </th>
                  <th>
                    <button type="button" className="th-sort" onClick={() => toggleSort('person')}>
                      화자{sortMark('person')}
                    </button>
                  </th>
                  <th>
                    <button type="button" className="th-sort" onClick={() => toggleSort('similarity')}>
                      신뢰도{sortMark('similarity')}
                    </button>
                  </th>
                  <th>방향(DOA)</th>
                  <th>상태</th>
                  <th>소리</th>
                </tr>
              </thead>
              <tbody>
                {pageItems.map((e, i) => (
                  <tr key={e.id} className={picked.has(e.id) ? 'picked' : ''}
                      onClick={() => setSelected(e)}>
                    <td className="col-check" onClick={(ev) => ev.stopPropagation()}>
                      <input type="checkbox" checked={picked.has(e.id)}
                             onChange={() => togglePick(e.id)}
                             aria-label={`${fmtDateTime(e.captured_at)} 선택`} />
                    </td>
                    <td className="row-num">{pageStart + i + 1}</td>
                    <td data-label="시각">{fmtDateTime(e.captured_at)}</td>
                    <td data-label="화자">
                      {e.person_alias ? (
                        <span className="speaker-cell">
                          <span className="legend-swatch"
                                style={{ background: colorOf[String(e.person_id)] || UNKNOWN_COLOR }} />
                          {e.person_alias}{e.person_room ? ` (${e.person_room})` : ''}
                        </span>
                      ) : <span className="muted">미등록</span>}
                    </td>
                    <td data-label="신뢰도"><Confidence value={e.similarity} source={e.person_source} /></td>
                    <td data-label="방향(DOA)" className="muted small">미지원</td>
                    <td data-label="상태">
                      <span className={e.person_id ? 'tag-reg' : 'tag-unreg'}>
                        {e.person_id ? '등록 화자' : '미등록'}
                      </span>
                      {needsReview(e) && <span className="tag-review">검토 필요</span>}
                    </td>
                    <td data-label="소리" onClick={(ev) => ev.stopPropagation()}>
                      {e.audio_available ? (
                        <button type="button" onClick={() => play(e)}>듣기</button>
                      ) : (
                        <span className="muted small" title="보존 기간이 지나 소리가 삭제되었습니다">
                          삭제됨
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
                {!loading && sorted.length === 0 && (
                  <tr>
                    <td colSpan={8}>
                      <StateBlock kind="empty" title="조건에 맞는 이벤트가 없습니다"
                                  detail="검색어를 지우거나 기간을 넓혀 보세요." />
                    </td>
                  </tr>
                )}
                {loading && (
                  <tr><td colSpan={8}><StateBlock kind="loading" title="불러오는 중…" /></td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <label className="field">
              <span className="field-label">페이지당</span>
              <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
                {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}행</option>)}
              </select>
            </label>
            <span className="spacer" />
            <button type="button" disabled={curPage <= 1} onClick={() => setPage(curPage - 1)}>‹ 이전</button>
            <span>{curPage} / {pageCount} 페이지 · 총 {sorted.length}건</span>
            <button type="button" disabled={curPage >= pageCount} onClick={() => setPage(curPage + 1)}>다음 ›</button>
          </div>
        </div>

        {policy && (
          <p className="policy-note">
            <b>소리 재생에 대해</b> — {policy.summary} {policy.detail}
            {' '}목록의 “삭제됨”은 보존 기간이 지나 소리가 지워진 이벤트입니다.
          </p>
        )}
        <audio ref={playerRef} preload="none" />
      </main>
      {selected && <EventModal event={selected} onClose={() => setSelected(null)} />}
    </>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import { api, audioUrl } from '../api'

const THRESHOLD = 0.75 // 식별 임계치 (P3 튜닝 전 잠정값)

function fmt(iso) {
  const d = new Date(iso)
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/* 오디오를 내려받아 실제 파형 막대를 그린다. 실패하면 막대를 숨긴다. */
function useWaveform(eventId) {
  const [bars, setBars] = useState(null)
  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const res = await fetch(audioUrl(eventId), { signal: AbortSignal.timeout(5000) })
        if (!res.ok) throw new Error()
        const buf = await res.arrayBuffer()
        const ctx = new (window.AudioContext || window.webkitAudioContext)()
        const audio = await ctx.decodeAudioData(buf)
        const data = audio.getChannelData(0)
        const N = 40
        const step = Math.floor(data.length / N)
        const out = []
        for (let i = 0; i < N; i++) {
          let peak = 0
          for (let j = i * step; j < (i + 1) * step; j += 16) {
            const v = Math.abs(data[j])
            if (v > peak) peak = v
          }
          out.push(peak)
        }
        const max = Math.max(0.001, ...out)
        if (alive) setBars(out.map((v) => v / max))
        ctx.close()
      } catch {
        if (alive) setBars(null)
      }
    }
    load()
    return () => { alive = false }
  }, [eventId])
  return bars
}

export default function EventModal({ event, onClose }) {
  const [persons, setPersons] = useState([])
  const [personId, setPersonId] = useState(event.person_id ?? '')
  const [note, setNote] = useState(null)
  const audioRef = useRef(null)
  const bars = useWaveform(event.id)

  useEffect(() => {
    api('/persons').then(setPersons).catch(() => {})
  }, [])

  const simPct = useMemo(
    () => (event.similarity != null ? Math.min(100, event.similarity * 100) : null),
    [event.similarity],
  )

  async function savePerson() {
    try {
      await api(`/events/${event.id}/person`, {
        method: 'PATCH',
        body: { person_id: personId === '' ? null : Number(personId) },
      })
      setNote('화자 수정 완료 — 목록은 다음 갱신 때 반영됩니다')
    } catch (e) {
      setNote(`실패: ${e.message}`)
    }
  }

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>이벤트 상세 — {fmt(event.captured_at)}</h3>
          <button type="button" className="linklike" onClick={onClose}>✕</button>
        </div>

        <div className="wave-box">
          {bars
            ? bars.map((v, i) => (
                <div key={i} className="wave-bar" style={{ height: `${Math.max(6, v * 100)}%` }} />
              ))
            : <p className="muted">파형을 불러올 수 없습니다 (오디오 없음)</p>}
        </div>

        <div className="modal-row">
          <button type="button" className="primary" onClick={() => audioRef.current?.play()}>▶ 재생</button>
          <audio ref={audioRef} src={audioUrl(event.id)} preload="none" />
          <span className="muted small">16kHz mono · peak RMS {event.peak_rms ?? '–'}</span>
        </div>

        <h4>식별 결과</h4>
        <p>화자: {event.person_alias ? `${event.person_alias}${event.person_room ? ` (${event.person_room})` : ''}` : '미등록'}</p>
        <p className="muted small">
          코사인 유사도 {event.similarity != null ? event.similarity.toFixed(2) : '–'} / 임계치 {THRESHOLD}
        </p>
        <div className="sim-track">
          {simPct != null && <div className="sim-fill" style={{ width: `${simPct}%` }} />}
          <div className="sim-threshold" style={{ left: `${THRESHOLD * 100}%` }} />
        </div>
        <p className="muted small">
          디바이스: {event.device_id} · 상태: {event.person_alias ? '등록 화자' : '미등록'}
        </p>

        <div className="modal-actions">
          <select value={personId} onChange={(e) => setPersonId(e.target.value)}>
            <option value="">미등록</option>
            {persons.map((p) => (
              <option key={p.id} value={p.id}>{p.alias}{p.room ? ` (${p.room})` : ''}</option>
            ))}
          </select>
          <button type="button" onClick={savePerson}>화자 수정 (오식별 보정)</button>
          <button type="button" onClick={() => setNote('재학습 큐에 등록되었습니다 (P3 모델 연동 예정)')}>
            이 샘플로 재학습 큐 등록
          </button>
          <button type="button" className="primary" onClick={onClose}>닫기</button>
        </div>
        {note && <p className="muted small">{note}</p>}
      </div>
    </div>
  )
}

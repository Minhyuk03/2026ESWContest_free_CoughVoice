import { useEffect, useMemo, useRef, useState } from 'react'
import { api, audioUrl } from '../api'
import Confidence from './Confidence'
import { THRESHOLD, confidence, fmtDateTime, speakerLabel } from '../lib/format'

/* 오디오를 내려받아 실제 파형 막대를 그린다. 실패하면 막대를 숨긴다. */
function useWaveform(eventId, enabled) {
  const [bars, setBars] = useState(null)
  useEffect(() => {
    if (!enabled) { setBars(null); return undefined }
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
  }, [eventId, enabled])
  return bars
}

export default function EventModal({ event, onClose }) {
  const [persons, setPersons] = useState([])
  const [personId, setPersonId] = useState(event.person_id ?? '')
  const [policy, setPolicy] = useState(null)
  const [note, setNote] = useState(null)
  const audioRef = useRef(null)
  // 원음은 보존 기간이 지나면 삭제된다. 없는 파일에 재생 버튼을 띄우지 않는다.
  const hasAudio = event.audio_available !== false
  const bars = useWaveform(event.id, hasAudio)

  useEffect(() => {
    api('/persons').then(setPersons).catch(() => {})
    api('/audio-policy').then(setPolicy).catch(() => {})
  }, [])

  const conf = useMemo(() => confidence(event.similarity), [event.similarity])

  async function savePerson() {
    try {
      await api(`/events/${event.id}/person`, {
        method: 'PATCH',
        body: { person_id: personId === '' ? null : Number(personId) },
      })
      setNote('화자를 바꿨습니다 — 목록은 다음 갱신 때 반영됩니다. '
              + '다음 기침부터 자동으로 인식하게 하려면 화자 관리에서 재등록하세요.')
    } catch (e) {
      setNote(`실패: ${e.message}`)
    }
  }

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>이벤트 상세 — {fmtDateTime(event.captured_at)}</h3>
          <button type="button" className="linklike" onClick={onClose}>✕</button>
        </div>

        <div className="wave-box">
          {bars
            ? bars.map((v, i) => (
                <div key={i} className="wave-bar" style={{ height: `${Math.max(6, v * 100)}%` }} />
              ))
            : (
              <p className="muted">
                {hasAudio ? '파형을 불러올 수 없습니다' : '보존 기간이 지나 소리가 삭제되었습니다'}
              </p>
            )}
        </div>

        <div className="modal-row">
          {hasAudio ? (
            <>
              <button type="button" className="primary" onClick={() => audioRef.current?.play()}>듣기</button>
              <audio ref={audioRef} src={audioUrl(event.id)} preload="none" />
            </>
          ) : (
            <span className="muted small">이 이벤트의 소리는 더 이상 재생할 수 없습니다.</span>
          )}
          <span className="muted small">16kHz mono · peak RMS {event.peak_rms ?? '–'}</span>
        </div>
        {policy?.enabled && (
          <p className="muted small">
            재생되는 소리는 이 장치가 녹음한 기침 구간이며, {policy.retention_days}일 뒤 자동으로 삭제됩니다.
            {event.enrolled && ' 이 이벤트는 화자 등록에 사용되어 재등록을 위해 계속 보관됩니다.'}
          </p>
        )}

        <h4>식별 결과</h4>
        <p>화자: {speakerLabel(event)}</p>
        <div className="modal-conf">
          <Confidence value={event.similarity} source={event.person_source} />
          <span className="muted small">
            {event.person_source === 'manual'
              ? `사람이 지정한 화자입니다. 모델이 냈던 점수는 ${conf.pct != null ? `${conf.pct}%` : '없음'}이며, 지정 이전 값이라 지금 라벨의 확신도가 아닙니다.`
              : `식별 기준 ${Math.round(THRESHOLD * 100)}% 이상일 때만 등록 화자로 판정합니다`}
            {event.person_source !== 'manual' && conf.level === 'mid'
              && ' — 기준을 겨우 넘긴 값이라 확인을 권합니다'}
          </span>
        </div>
        <div className="sim-track">
          {conf.pct != null && (
            <div className={`sim-fill conf-${conf.level}`} style={{ width: `${conf.pct}%` }} />
          )}
          <div className="sim-threshold" style={{ left: `${THRESHOLD * 100}%` }} />
        </div>
        <p className="muted small">
          기기: {event.device_id} · 판정: {event.person_alias ? '등록 화자' : '미등록'}
          {event.cough_score != null && ` · 기침 확신도 ${Math.round(event.cough_score * 100)}%`}
        </p>

        <div className="modal-actions">
          <label className="field">
            <span className="field-label">화자 지정</span>
            <select value={personId} onChange={(e) => setPersonId(e.target.value)}>
              <option value="">미등록</option>
              {persons.map((p) => (
                <option key={p.id} value={p.id}>{p.alias}{p.room ? ` (${p.room})` : ''}</option>
              ))}
            </select>
          </label>
          <button type="button" onClick={savePerson}>화자 수정 (오식별 보정)</button>
          <button type="button" className="primary" onClick={onClose}>닫기</button>
        </div>
        {note && <p className="muted small">{note}</p>}
      </div>
    </div>
  )
}

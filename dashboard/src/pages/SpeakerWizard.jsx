import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, getServerUrl } from '../api'
import Topbar from '../components/Topbar'

const MIN_SAMPLES = 5
const STEPS = ['1. 별칭 입력', '2. 기침 샘플 선택', '3. 확인 및 완료']

// 등록 샘플은 실제 장치가 보낸 기침 이벤트에서 고른다. 브라우저 마이크로 녹음하면
// 등록과 식별의 마이크가 달라져 정확도가 떨어진다(실측: 같은 경로로 등록하면
// 유사도 0.608 → 0.684). 오디오는 이미 서버에 있으므로 옮길 필요도 없다.

function fmtTime(iso) {
  const d = new Date(iso)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

export default function SpeakerWizard() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const reenrollId = params.get('reenroll')

  const [step, setStep] = useState(0)
  const [alias, setAlias] = useState('')
  const [room, setRoom] = useState('')
  const [events, setEvents] = useState([])
  const [picked, setPicked] = useState([])       // 선택한 이벤트 id
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!reenrollId) return
    api('/persons').then((list) => {
      const p = list.find((x) => String(x.id) === reenrollId)
      if (p) { setAlias(p.alias); setRoom(p.room || '') }
    }).catch(() => {})
  }, [reenrollId])

  const loadEvents = useCallback(() => {
    setLoading(true)
    api('/events?limit=40')
      .then((list) => setEvents(list))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { if (step === 1) loadEvents() }, [step, loadEvents])

  function toggle(id) {
    setPicked((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]))
  }

  function play(id) {
    // GET /events/{id}/audio 는 인증 없이 열리는 재생용 엔드포인트다.
    new Audio(`${getServerUrl()}/events/${id}/audio`).play().catch(() => {
      setError('오디오를 재생할 수 없습니다.')
    })
  }

  async function finish() {
    setBusy(true)
    setError(null)
    try {
      let id = reenrollId
      if (id) {
        await api(`/persons/${id}`, { method: 'PATCH', body: { alias, room: room || null } })
      } else {
        const created = await api('/persons', { method: 'POST', body: { alias, room: room || null } })
        id = created.id
      }
      await api(`/persons/${id}/enroll-from-events`, {
        method: 'POST',
        body: { event_ids: picked },
      })
      navigate('/speakers')
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  const canNext =
    (step === 0 && alias.trim().length > 0) ||
    (step === 1 && picked.length >= MIN_SAMPLES) ||
    step === 2

  return (
    <>
      <Topbar title={`화자 관리 › ${reenrollId ? '재등록' : '신규 등록'}`} />
      <main className="page">
        <div className="card wizard">
          <div className="wizard-steps">
            {STEPS.map((s, i) => (
              <div key={s} className={`wizard-step ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}>
                {s}
              </div>
            ))}
          </div>

          {step === 0 && (
            <div className="wizard-body">
              <h3>별칭과 호실을 입력하세요</h3>
              <p className="muted">실명 대신 별칭만 저장합니다 (NFR-06).</p>
              <input placeholder="별칭 (예: A)" value={alias} onChange={(e) => setAlias(e.target.value)} />
              <input placeholder="호실 (예: 301호, 선택)" value={room} onChange={(e) => setRoom(e.target.value)} />
            </div>
          )}

          {step === 1 && (
            <div className="wizard-body">
              <h3>이 사람의 기침을 {MIN_SAMPLES}개 이상 고르세요</h3>
              <p className="muted">
                장치 앞에서 기침하면 아래 목록에 나타납니다. 재생해서 확인한 뒤 선택하세요.
                샘플이 많을수록 정확도가 올라갑니다.
              </p>
              <div className="page-head">
                <p>선택: <b>{picked.length}</b> / 최소 {MIN_SAMPLES}</p>
                <button type="button" onClick={loadEvents} disabled={loading}>
                  {loading ? '불러오는 중…' : '↻ 목록 새로고침'}
                </button>
              </div>
              <div className="alert-list">
                {events.map((e) => (
                  <label key={e.id} className="alert-item" style={{ cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={picked.includes(e.id)}
                      onChange={() => toggle(e.id)}
                    />
                    <div style={{ flex: 1 }}>
                      <p className="alert-title">{fmtTime(e.captured_at)} · {e.device_id}</p>
                      <p className="muted small">
                        현재 판정: {e.person_alias || '미등록'}
                        {e.similarity != null && ` (유사도 ${e.similarity})`}
                      </p>
                    </div>
                    <button type="button" onClick={(ev) => { ev.preventDefault(); play(e.id) }}>
                      ▶ 듣기
                    </button>
                  </label>
                ))}
                {events.length === 0 && !loading && (
                  <p className="muted">아직 수신된 기침이 없습니다. 장치 앞에서 기침해 보세요.</p>
                )}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="wizard-body">
              <h3>확인 및 완료</h3>
              <p>별칭: <b>{alias}</b>{room ? ` · 호실: ${room}` : ''}</p>
              <p>등록 샘플: <b>{picked.length}개</b></p>
              <p className="muted">
                완료를 누르면 선택한 기침으로 화자 지문을 만들어 {reenrollId ? '갱신' : '등록'}합니다.
                원본 음성은 이미 서버에 있는 이벤트 오디오를 그대로 사용합니다.
              </p>
              {error && <p className="form-error">{error}</p>}
            </div>
          )}

          <div className="wizard-nav">
            <button type="button" onClick={() => (step === 0 ? navigate('/speakers') : setStep(step - 1))}>
              ← {step === 0 ? '취소' : '이전'}
            </button>
            {step < 2 ? (
              <button type="button" className="primary" disabled={!canNext} onClick={() => setStep(step + 1)}>
                다음 →
              </button>
            ) : (
              <button type="button" className="primary" disabled={busy} onClick={finish}>
                {busy ? '등록 중…' : '완료'}
              </button>
            )}
          </div>
        </div>
      </main>
    </>
  )
}

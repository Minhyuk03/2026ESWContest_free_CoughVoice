import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'

const MIN_SAMPLES = 8
const STEPS = ['1. 별칭 입력', '2. 기침 샘플 녹음', '3. 확인 및 완료']

export default function SpeakerWizard() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const reenrollId = params.get('reenroll')

  const [step, setStep] = useState(0)
  const [alias, setAlias] = useState('')
  const [room, setRoom] = useState('')
  const [samples, setSamples] = useState(0)
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!reenrollId) return
    api('/persons').then((list) => {
      const p = list.find((x) => String(x.id) === reenrollId)
      if (p) {
        setAlias(p.alias)
        setRoom(p.room || '')
      }
    }).catch(() => {})
  }, [reenrollId])

  function record() {
    // 실제 녹음 파이프라인(P3 임베딩 추출)은 collect_cough.py로 수집 후 연동 예정.
    // 여기서는 등록 절차 UI 흐름을 검증하는 데모 카운트로 동작한다.
    setRecording(true)
    setTimeout(() => {
      setSamples((n) => n + 1)
      setRecording(false)
    }, 700)
  }

  async function finish() {
    setBusy(true)
    setError(null)
    try {
      if (reenrollId) {
        await api(`/persons/${reenrollId}`, {
          method: 'PATCH',
          body: { alias, room: room || null, sample_count: samples },
        })
      } else {
        await api('/persons', {
          method: 'POST',
          body: { alias, room: room || null, sample_count: samples },
        })
      }
      navigate('/speakers')
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  const canNext =
    (step === 0 && alias.trim().length > 0) ||
    (step === 1 && samples >= MIN_SAMPLES) ||
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
              <h3>기침 샘플 녹음 (최소 {MIN_SAMPLES}회)</h3>
              <p className="muted">
                마이크 앞에서 자연스럽게 기침해 주세요. 샘플이 많을수록 식별 정확도가 올라갑니다.
              </p>
              <button
                type="button"
                className={`rec-btn ${recording ? 'recording' : ''}`}
                onClick={record}
                disabled={recording}
              >
                ● REC
              </button>
              <p>진행: {samples} / {MIN_SAMPLES}</p>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.min(100, (samples / MIN_SAMPLES) * 100)}%` }} />
              </div>
              <div className="sample-chips">
                {Array.from({ length: samples }, (_, i) => (
                  <span key={i} className="sample-chip">샘플{i + 1} ✓</span>
                ))}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="wizard-body">
              <h3>확인 및 완료</h3>
              <p>별칭: <b>{alias}</b>{room ? ` · 호실: ${room}` : ''}</p>
              <p>녹음 샘플: <b>{samples}개</b></p>
              <p className="muted">완료를 누르면 화자가 {reenrollId ? '갱신' : '등록'}됩니다.</p>
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
                {busy ? '저장 중…' : '완료'}
              </button>
            )}
          </div>
        </div>
      </main>
    </>
  )
}

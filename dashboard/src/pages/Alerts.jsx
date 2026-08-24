import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'

function fmt(iso) {
  const d = new Date(iso)
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  const hh = String(d.getHours()).padStart(2, '0')
  const mi = String(d.getMinutes()).padStart(2, '0')
  return `${mm}-${dd} ${hh}:${mi}`
}

export default function Alerts() {
  const [alerts, setAlerts] = useState([])
  const [rules, setRules] = useState([])
  const [error, setError] = useState(null)

  function reload() {
    api('/alerts').then(setAlerts).catch((e) => setError(e.message))
    api('/alert-rules').then(setRules).catch((e) => setError(e.message))
  }
  useEffect(reload, [])

  async function toggle(rule) {
    try {
      await api(`/alert-rules/${rule.id}`, { method: 'PATCH', body: { enabled: !rule.enabled } })
      reload()
    } catch (e) {
      setError(e.message)
    }
  }

  // 조건을 자유 문구로 받지 않고 숫자로 받아 표시 문구를 생성한다.
  // 문구와 실제 평가 파라미터가 어긋나면 화면에는 "3회/10분"이라 적혀 있는데
  // 동작은 기본값대로 도는 사고가 난다.
  async function addRule() {
    const name = window.prompt('규칙 이름을 입력하세요', '새 규칙')
    if (!name) return
    const count = Number(window.prompt('기침 몇 회 이상일 때 알릴까요?', '10'))
    if (!Number.isFinite(count) || count < 1) return
    const minutes = Number(window.prompt('몇 분 안에 발생한 것을 셀까요?', '60'))
    if (!Number.isFinite(minutes) || minutes < 1) return
    const span = minutes % 60 === 0 ? `${minutes / 60}시간` : `${minutes}분`
    try {
      await api('/alert-rules', {
        method: 'POST',
        body: {
          name,
          condition_text: `기침 ≥ ${count}회 / ${span}`,
          kind: 'count_window',
          threshold_count: count,
          window_minutes: minutes,
        },
      })
      reload()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <>
      <Topbar title="알림 센터 & 규칙 설정" />
      <main className="page">
        {error && <p className="form-error">{error}</p>}
        <div className="alerts-cols">
          <div className="card">
            <h3>알림 이력</h3>
            <div className="alert-list">
              {alerts.map((a) => (
                <div key={a.id} className="alert-item">
                  <div>
                    <p className="alert-title">{a.rule}</p>
                    <p className="muted small">
                      대상: {a.person_alias ? `${a.person_alias}${a.person_room ? ` (${a.person_room})` : ''}` : '—'} · {fmt(a.created_at)}
                    </p>
                  </div>
                  <div className="alert-side">
                    <span className="muted small">웹훅 발송 ✓</span>
                    <Link to="/history" className="small">이력 보기 →</Link>
                  </div>
                </div>
              ))}
              {alerts.length === 0 && <p className="muted">알림 이력이 없습니다.</p>}
            </div>
          </div>

          <div className="card">
            <div className="page-head">
              <h3>알림 규칙</h3>
              <button type="button" onClick={addRule}>+ 규칙 추가</button>
            </div>
            <div className="rule-list">
              {rules.map((r) => (
                <div key={r.id} className="rule-card">
                  <div className="rule-head">
                    <p className="alert-title">{r.name}</p>
                    <button
                      type="button"
                      className={`toggle ${r.enabled ? 'on' : ''}`}
                      onClick={() => toggle(r)}
                    >
                      {r.enabled ? 'ON' : 'OFF'}
                    </button>
                  </div>
                  <p className="muted small">조건: {r.condition_text || '—'}</p>
                  <p className="muted small">
                    실제 평가:{' '}
                    {r.kind === 'unknown'
                      ? '미등록 화자 기침 발생 시 즉시'
                      : `${r.threshold_count}회 / ${r.window_minutes}분` +
                        (r.kind === 'night_window' ? ' (야간 시간대만)' : '')}
                    {r.cooldown_minutes > 0 && ` · 재알림 ${r.cooldown_minutes}분 억제`}
                  </p>
                  <p className="muted small">대상: {r.target_text}</p>
                  <p className="muted small">수신 채널: {r.channels_text}</p>
                </div>
              ))}
            </div>
            <p className="muted small note">
              ※ 알림 클릭 시 해당 시간대 필터가 적용된 기침 이력(S2)으로 이동
            </p>
          </div>
        </div>
      </main>
    </>
  )
}

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

  async function addRule() {
    const name = window.prompt('규칙 이름을 입력하세요', '새 규칙')
    if (!name) return
    const cond = window.prompt('조건을 입력하세요 (예: 기침 ≥ 10회 / 1시간)', '')
    try {
      await api('/alert-rules', { method: 'POST', body: { name, condition_text: cond || '' } })
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

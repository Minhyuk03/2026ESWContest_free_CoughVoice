import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'

function lastCoughText(iso) {
  if (!iso) return '기침 기록 없음'
  const d = new Date(iso)
  const today = new Date()
  if (d.toDateString() === today.toDateString()) {
    return `최근 기침 ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  return `최근 기침 ${d.getMonth() + 1}/${d.getDate()}`
}

export default function Speakers() {
  const navigate = useNavigate()
  const [persons, setPersons] = useState([])
  const [error, setError] = useState(null)

  function reload() {
    api('/persons').then(setPersons).catch((e) => setError(e.message))
  }
  useEffect(reload, [])

  async function remove(p) {
    if (!window.confirm(`${p.alias} 화자를 삭제할까요? 이벤트 이력은 미등록으로 남습니다.`)) return
    try {
      await api(`/persons/${p.id}`, { method: 'DELETE' })
      reload()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <>
      <Topbar title="화자 관리" />
      <main className="page">
        <div className="page-head">
          <p className="muted">
            등록 화자 {persons.length}명 · 원본 음성 비보존, 임베딩만 저장 (NFR-06)
          </p>
          <button type="button" className="primary" onClick={() => navigate('/speakers/new')}>
            + 신규 등록
          </button>
        </div>
        {error && <p className="form-error">{error}</p>}

        <div className="speaker-grid">
          {persons.map((p) => (
            <div key={p.id} className="card speaker-card">
              <div className="speaker-top">
                <div className="avatar">{p.alias.slice(0, 1)}</div>
                <div>
                  <h3>{p.alias} <span className="muted small">(별칭)</span></h3>
                  <p className="muted small">
                    {p.room ? `${p.room} · ` : ''}등록 {p.created_at ? p.created_at.slice(0, 10) : '–'}
                  </p>
                </div>
              </div>
              <p className="muted small">
                샘플 {p.sample_count}개 · {lastCoughText(p.last_cough_at)}
              </p>
              <div className="speaker-actions">
                <button type="button" onClick={() => navigate(`/speakers/new?reenroll=${p.id}`)}>재등록</button>
                <button type="button" className="danger" onClick={() => remove(p)}>삭제</button>
              </div>
            </div>
          ))}
          {persons.length === 0 && !error && (
            <p className="muted">등록된 화자가 없습니다. 신규 등록으로 시작하세요.</p>
          )}
        </div>
      </main>
    </>
  )
}

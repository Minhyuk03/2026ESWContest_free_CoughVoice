import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'
import StateBlock from '../components/StateBlock'
import { fmtSmart } from '../lib/format'

function lastCoughText(iso) {
  if (!iso) return '기침 기록 없음'
  return `최근 기침 ${fmtSmart(iso)}`
}

export default function Speakers() {
  const navigate = useNavigate()
  const [persons, setPersons] = useState([])
  const [policy, setPolicy] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const reload = useCallback(() => {
    setLoading(true)
    api('/persons')
      .then((r) => { setPersons(r); setError(null) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(reload, [reload])
  useEffect(() => { api('/audio-policy').then(setPolicy).catch(() => {}) }, [])

  async function remove(p) {
    if (!window.confirm(`${p.alias} 화자를 삭제할까요? 이벤트 이력은 미판정으로 남습니다.`)) return
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
          <p className="muted">등록 화자 {persons.length}명</p>
          <button type="button" className="primary" onClick={() => navigate('/speakers/new')}>
            + 신규 등록
          </button>
        </div>

        {/* 내부 용어(임베딩·NFR-06) 대신 사용자가 판단할 수 있는 말로 적는다.
            재생 버튼이 있는 화면과 모순되지 않도록 보관 기간도 함께 밝힌다. */}
        <p className="policy-note">
          <b>개인정보 보호</b> — 이름 대신 별칭만 저장하며, 화자를 구분하는 데 필요한
          목소리 특징 정보만 보관합니다.
          {policy?.enabled && ` 감지된 기침 소리는 ${policy.retention_days}일 뒤 자동으로 지워집니다
            (화자 등록에 사용한 기침은 재등록을 위해 계속 보관합니다).`}
        </p>

        {error && (
          <div className="banner banner-error">
            <span>{error}</span>
            <button type="button" onClick={reload}>다시 시도</button>
          </div>
        )}

        {loading ? (
          <StateBlock kind="loading" title="화자 목록을 불러오는 중입니다" />
        ) : persons.length === 0 ? (
          <StateBlock
            kind="empty"
            title="등록된 화자가 없습니다"
            detail="장치가 감지한 기침 중 같은 사람의 것을 5개 이상 고르면 화자를 등록할 수 있습니다."
            action={<button type="button" className="primary" onClick={() => navigate('/speakers/new')}>
              신규 등록 시작
            </button>}
          />
        ) : (
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
                  등록 샘플 {p.sample_count}개 · {lastCoughText(p.last_cough_at)}
                </p>
                <p className="small">
                  <Link to={`/history?person=${p.id}`}>이 화자의 기침 이력 →</Link>
                </p>
                <div className="speaker-actions">
                  <button type="button" onClick={() => navigate(`/speakers/new?reenroll=${p.id}`)}>재등록</button>
                  <button type="button" className="danger" onClick={() => remove(p)}>삭제</button>
                </div>
              </div>
            ))}
          </div>
        )}

        {persons.length > 0 && (
          <p className="muted small">
            잘못 식별된 기침이 있나요? <Link to="/history?person=unknown">미판정으로 남은 기침 목록</Link>에서
            여러 건을 골라 기존 화자에 연결할 수 있습니다. 다음 기침부터 자동으로 인식하게 하려면
            연결 후 <b>재등록</b>으로 목소리 특징을 갱신하세요.
          </p>
        )}
      </main>
    </>
  )
}

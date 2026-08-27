import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, getServerUrl, saveSession, setServerUrl } from '../api'

export default function Login() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [server, setServer] = useState(getServerUrl())
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    setServerUrl(server)
    try {
      const r = await api('/auth/login', { method: 'POST', body: { username, password } })
      saveSession(r.token, r.username)
      navigate('/')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        {/* 이모지 대신 음성 파형을 그린 표식 — 글꼴·플랫폼에 따라 달라지지 않는다 */}
        <div className="login-logo" aria-hidden="true">
          <svg viewBox="0 0 40 24" width="40" height="24">
            {[6, 12, 20, 14, 8].map((h, i) => (
              <rect key={i} x={i * 8 + 2} y={(24 - h) / 2} width="4" height={h}
                    rx="2" fill="currentColor" />
            ))}
          </svg>
        </div>
        <h1>기침 화자 식별 시스템</h1>
        <p className="muted">관리자 · 보호자 로그인</p>
        <input
          placeholder="아이디"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
        />
        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
        />
        <input
          placeholder="서버 주소"
          value={server}
          onChange={(e) => setServer(e.target.value)}
          spellCheck={false}
        />
        {error && <p className="form-error">{error}</p>}
        <button type="submit" className="primary" disabled={busy}>
          {busy ? '확인 중…' : '로그인'}
        </button>
        <p className="login-hint">보호자 계정은 담당 거주자 이력만 조회 가능 (읽기 전용)</p>
      </form>
    </div>
  )
}

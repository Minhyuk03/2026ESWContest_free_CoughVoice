import { NavLink, Navigate, Outlet, useNavigate } from 'react-router-dom'
import { api, clearSession, getToken, getUsername } from '../api'

const MENU = [
  { to: '/', label: '대시보드', end: true },
  { to: '/history', label: '기침 이력' },
  { to: '/speakers', label: '화자 관리' },
  { to: '/alert-center', label: '알림 센터' },
]

export default function Layout() {
  const navigate = useNavigate()
  if (!getToken()) return <Navigate to="/login" replace />

  async function logout() {
    try {
      await api('/auth/logout', { method: 'POST' })
    } catch { /* 서버 미응답이어도 로컬 세션은 정리 */ }
    clearSession()
    navigate('/login')
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <h1 className="brand">기침 화자 식별</h1>
        <nav>
          {MENU.map((m) => (
            <NavLink key={m.to} to={m.to} end={m.end} className="nav-item">
              {({ isActive }) => <span>{isActive ? '● ' : ''}{m.label}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span>관리자 {getUsername() || ''}</span>
          <button type="button" className="linklike" onClick={logout}>로그아웃</button>
        </div>
      </aside>
      <div className="content">
        <Outlet />
      </div>
    </div>
  )
}

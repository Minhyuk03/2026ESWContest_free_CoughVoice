import { useEffect, useState } from 'react'
import { NavLink, Navigate, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { api, clearSession, getToken, getUsername } from '../api'

const MENU = [
  { to: '/', label: '대시보드', end: true },
  { to: '/history', label: '기침 이력' },
  { to: '/speakers', label: '화자 관리' },
  { to: '/alert-center', label: '알림 센터' },
]

export default function Layout() {
  const navigate = useNavigate()
  const location = useLocation()
  // 좁은 화면에서는 사이드바가 화면을 덮는 서랍이 된다. 경로가 바뀌면 닫는다 —
  // 열린 채로 두면 이동한 화면이 서랍에 가려 아무것도 안 바뀐 것처럼 보인다.
  const [menuOpen, setMenuOpen] = useState(false)
  useEffect(() => { setMenuOpen(false) }, [location.pathname])

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
      {/* 모바일 전용 상단 바 — 데스크톱에서는 숨는다 */}
      <div className="mobile-bar">
        <button
          type="button"
          className="menu-btn"
          aria-label="메뉴 열기"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((v) => !v)}
        >
          메뉴
        </button>
        <span className="mobile-brand">기침 화자 식별</span>
      </div>

      {menuOpen && <div className="drawer-back" onClick={() => setMenuOpen(false)} />}

      <aside className={`sidebar${menuOpen ? ' open' : ''}`}>
        <h1 className="brand">기침 화자 식별</h1>
        <nav>
          {MENU.map((m) => (
            <NavLink key={m.to} to={m.to} end={m.end} className="nav-item">
              {m.label}
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

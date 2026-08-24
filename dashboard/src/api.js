// 서버 API 래퍼 — base URL은 localStorage로 조정 가능 (핫스팟 IP 변동 대응)

const URL_KEY = 'cough-id-server-url'
const TOKEN_KEY = 'cough-id-token'
const USER_KEY = 'cough-id-username'

export function getServerUrl() {
  const saved = localStorage.getItem(URL_KEY)
  if (saved) return saved
  // 서버가 이 대시보드를 직접 서빙하면 같은 출처를 쓴다. 그러면 서버 주소를
  // 입력할 필요가 없고, IP가 바뀌어도 화면이 열린 주소를 그대로 따라간다.
  // vite dev(5173)로 띄운 개발 중에만 기본값 8000으로 넘어간다.
  const { origin, port } = window.location
  return port === '5173' ? 'http://localhost:8000' : origin
}

export function setServerUrl(url) {
  localStorage.setItem(URL_KEY, url.replace(/\/$/, ''))
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function getUsername() {
  return localStorage.getItem(USER_KEY)
}

export function saveSession(token, username) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, username)
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
    options = { ...options, body: JSON.stringify(options.body) }
  }
  const token = getToken()
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${getServerUrl()}${path}`, {
    ...options,
    headers,
    signal: options.signal ?? AbortSignal.timeout(5000),
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const j = await res.json()
      if (j.detail) detail = j.detail
    } catch { /* body 없음 */ }
    throw new Error(detail)
  }
  return res.status === 204 ? null : res.json()
}

export function audioUrl(eventId) {
  return `${getServerUrl()}/events/${eventId}/audio`
}

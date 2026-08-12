// 서버 API 래퍼 — base URL은 localStorage로 조정 가능 (핫스팟 IP 변동 대응)

const URL_KEY = 'cough-id-server-url'
const TOKEN_KEY = 'cough-id-token'
const USER_KEY = 'cough-id-username'

export function getServerUrl() {
  return localStorage.getItem(URL_KEY) || 'http://localhost:8000'
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

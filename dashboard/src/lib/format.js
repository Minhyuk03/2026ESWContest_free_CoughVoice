/* 화면 여러 곳에서 같은 값을 다르게 읽는 일이 없도록, 표시 규칙을 한 곳에 모은다.
   같은 유사도 0.88이 어떤 화면에선 "0.88", 다른 화면에선 "높음"으로 보이면
   사용자는 두 값이 같은 것인지 알 수 없다. */

export const THRESHOLD = 0.75      // 식별 임계치 — 이보다 낮으면 등록 화자로 보지 않는다
export const HIGH_CONFIDENCE = 0.85 // 이 이상이면 그대로 믿어도 되는 수준
// 검토 목록에 담을 기준. 임계치를 겨우 넘긴 건은 오식별일 수 있어 함께 본다.
export const REVIEW_BELOW = HIGH_CONFIDENCE

/** 유사도(0~1)를 "88% · 높음"처럼 읽을 수 있는 형태로 바꾼다. */
export function confidence(sim) {
  if (sim == null) {
    return { pct: null, level: 'none', label: '비교 없음', text: '비교 없음' }
  }
  const clamped = Math.min(1, Math.max(0, sim))
  const pct = Math.round(clamped * 100)
  let level = 'low'
  let label = '낮음'
  if (sim >= HIGH_CONFIDENCE) { level = 'high'; label = '높음' }
  else if (sim >= THRESHOLD) { level = 'mid'; label = '보통' }
  return { pct, level, label, text: `${pct}% · ${label}`, raw: sim }
}

/** 검토가 필요한 이벤트인가. 유사도가 아예 없는 건(비교 대상 없음)과
 *  사람이 이미 판단해 지정한 건은 제외한다. */
export function needsReview(ev) {
  if (ev.person_source === 'manual') return false
  return ev.similarity != null && ev.similarity < REVIEW_BELOW
}

const p2 = (n) => String(n).padStart(2, '0')

export function fmtTime(iso) {
  const d = new Date(iso)
  return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`
}

export function fmtDateTime(iso) {
  const d = new Date(iso)
  return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())} ` +
         `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`
}

/** 오늘이면 시각만, 아니면 날짜까지. 목록에서 줄 길이를 아끼려고 쓴다. */
export function fmtSmart(iso) {
  const d = new Date(iso)
  const today = new Date()
  if (d.toDateString() === today.toDateString()) return fmtTime(iso)
  return `${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}`
}

export function fmtElapsed(seconds) {
  if (seconds == null) return '–'
  if (seconds < 60) return `${Math.round(seconds)}초 전`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}분 전`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}시간 전`
  return `${Math.floor(seconds / 86400)}일 전`
}

/* 화자가 비어 있을 때 '미등록 화자'라고 쓰면 "등록 안 된 사람이 기침했다"로 읽히는데,
   그건 사실이 아니다. 등록본 어느 쪽과도 임계치만큼 닮지 않았다는 뜻일 뿐이고,
   실제 외부인 40건 중 35건(87.5%)은 오히려 등록자 이름을 달았다(2026-08-28 측정).
   있지도 않은 사람을 만들지 않도록 '미판정'으로 쓴다. */
export function speakerLabel(ev) {
  if (!ev.person_alias) return '미판정'
  return ev.person_room ? `${ev.person_alias} (${ev.person_room})` : ev.person_alias
}

/* 알림은 **규칙 종류**를 짧은 말로 함께 보여 준다. 심각도만 쓰면 대부분이 '주의'라
   목록을 훑을 때 무엇 때문에 뜬 알림인지 구분되지 않는다. 종류를 모르면 '기타'. */
const KIND_LABEL = {
  count_window: '횟수',
  night_window: '야간',
  unknown: '미판정',        // 폐기된 규칙 종류 — 낡은 알림 이력 표시용으로만 남긴다
  baseline_delta: '평소 대비',
  duration_days: '지속 기간',
  urgent_symptom: '긴급 증상',
}

export function alertKindLabel(alert) {
  return KIND_LABEL[alert.rule_kind] || '기타'
}

export const SEV_ORDER = ['urgent', 'advisory', 'info']
export const SEV_LABEL = { urgent: '긴급', advisory: '중요', info: '주의' }
export const STATUS_LABEL = { open: '미확인', ack: '확인함', done: '조치 완료' }

/** 규칙 파라미터로 조건 문구를 만든다.
 *  문구를 사용자가 자유롭게 적게 두면 화면엔 "3회/10분"이라 쓰여 있는데 실제로는
 *  기본값대로 도는 사고가 난다. 표시는 항상 파라미터에서 생성한다. */
export function conditionText(kind, v) {
  const span = (min) => (min % 60 === 0 ? `${min / 60}시간` : `${min}분`)
  switch (kind) {
    case 'night_window':
      return `기침 ≥ ${v.threshold_count}회 / ${v.night_start_hour}–${v.night_end_hour}시`
    case 'unknown':
      return '(폐기된 규칙)'
    case 'baseline_delta':
      return `개인 기준선의 ${v.ratio_threshold}배 / 최근 ${v.sustain_hours}시간`
    case 'duration_days':
      return `기침 ${v.duration_days}일 이상 지속`
    case 'urgent_symptom':
      return '긴급 증상 입력 시 즉시'
    default:
      return `기침 ≥ ${v.threshold_count}회 / ${span(v.window_minutes)}`
  }
}

export const RULE_KINDS = [
  { value: 'count_window', label: '횟수 (지정 시간 안에 N회)' },
  { value: 'night_window', label: '야간 횟수 (야간 시간대에 N회)' },
  // '미등록 화자 감지'는 2026-08-28에 뺐다. 근거가 없다 — 외부인 40건 중 35건(87.5%)이
  // 임계치를 넘어 등록자 이름을 달았으므로, 화자가 비는 것은 외부인의 신호가 아니다.
  { value: 'baseline_delta', label: '평소 대비 증가' },
  { value: 'duration_days', label: '기침 지속 기간' },
  { value: 'urgent_symptom', label: '긴급 증상 입력' },
]

/** 알림에서 "이력 보기"로 넘어갈 때 쓰는 링크.
 *
 *  알림 시각 앞뒤 구간과 대상 화자를 이력 화면에 그대로 넘긴다. 필터 없이 넘기면
 *  사용자가 알림이 가리키는 기침을 직접 찾아야 해서, 목록이 길수록 사실상 못 찾는다.
 *  앞 구간을 넉넉히(기본 3시간) 잡는 이유는 알림이 "누적 N회"처럼 지나간 구간을
 *  근거로 뜨기 때문이다. */
export function alertHistoryLink(alert, beforeMinutes = 180, afterMinutes = 30) {
  const t = new Date(alert.created_at).getTime()
  const q = new URLSearchParams({
    from: new Date(t - beforeMinutes * 60000).toISOString(),
    to: new Date(t + afterMinutes * 60000).toISOString(),
    alert: String(alert.id),
  })
  if (alert.person_id) q.set('person', String(alert.person_id))
  else q.set('person', 'unknown')
  return `/history?${q.toString()}`
}

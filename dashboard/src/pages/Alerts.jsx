import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import Topbar from '../components/Topbar'
import StateBlock from '../components/StateBlock'
import {
  RULE_KINDS, SEV_LABEL, STATUS_LABEL, alertHistoryLink, alertKindLabel,
  conditionText, fmtDateTime,
} from '../lib/format'

const STATUS_FLOW = ['open', 'ack', 'done']

// 알림이 쌓이면 목록만으로 페이지가 한없이 길어져 오른쪽 규칙 카드까지 밀린다.
const PAGE_SIZE = 5

const SEV_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'urgent', label: '긴급' },
  { key: 'advisory', label: '중요' },
  { key: 'info', label: '주의' },
]

const STATUS_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'open', label: '미확인' },
  { key: 'ack', label: '확인함' },
  { key: 'done', label: '조치 완료' },
]

/* 규칙 종류마다 실제로 쓰는 파라미터가 다르다. 안 쓰는 값을 입력받으면 화면에는
   보이는데 동작에는 영향이 없는 칸이 생겨, 사용자가 고쳐도 아무 일이 없는 것처럼 보인다. */
const FIELDS_BY_KIND = {
  count_window: ['threshold_count', 'window_minutes', 'cooldown_minutes'],
  night_window: ['threshold_count', 'night_start_hour', 'night_end_hour', 'cooldown_minutes'],
  unknown: ['cooldown_minutes'],
  baseline_delta: ['baseline_days', 'ratio_threshold', 'sustain_hours', 'cooldown_minutes'],
  duration_days: ['duration_days', 'allowed_gap_days', 'cooldown_minutes'],
  urgent_symptom: ['cooldown_minutes'],
}

const FIELD_META = {
  threshold_count: { label: '기준 횟수 (회)', min: 1, step: 1 },
  window_minutes: { label: '관찰 구간 (분)', min: 1, step: 1 },
  night_start_hour: { label: '야간 시작 (시)', min: 0, max: 23, step: 1 },
  night_end_hour: { label: '야간 종료 (시)', min: 0, max: 23, step: 1 },
  cooldown_minutes: { label: '재알림 억제 (분)', min: 0, step: 1 },
  baseline_days: { label: '기준선 학습 (일)', min: 1, step: 1 },
  ratio_threshold: { label: '기준선 대비 배수', min: 1, step: 0.1 },
  sustain_hours: { label: '증가 지속 (시간)', min: 1, step: 1 },
  duration_days: { label: '지속 기간 (일)', min: 1, step: 1 },
  allowed_gap_days: { label: '허용 공백 (일)', min: 0, step: 1 },
}

const BLANK_RULE = {
  name: '새 규칙', kind: 'count_window', target_text: '전체 화자',
  threshold_count: 10, window_minutes: 60, night_start_hour: 22, night_end_hour: 6,
  cooldown_minutes: 30, baseline_days: 7, ratio_threshold: 2, sustain_hours: 24,
  duration_days: 14, allowed_gap_days: 2,
}

function RuleForm({ rule, onClose, onSaved }) {
  const [form, setForm] = useState(() => ({ ...BLANK_RULE, ...rule }))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const fields = FIELDS_BY_KIND[form.kind] || FIELDS_BY_KIND.count_window
  // 조건 문구는 입력값에서 만든다. 사용자가 따로 적게 두면 표시와 동작이 갈라진다.
  const preview = conditionText(form.kind, form)

  function set(k, v) { setForm((f) => ({ ...f, [k]: v })) }

  async function save() {
    setBusy(true)
    setError(null)
    const body = {
      name: form.name.trim() || '새 규칙',
      kind: form.kind,
      target_text: form.target_text.trim() || '전체 화자',
      condition_text: preview,
      ...Object.fromEntries(fields.map((f) => [f, Number(form[f])])),
    }
    try {
      if (rule?.id) await api(`/alert-rules/${rule.id}`, { method: 'PATCH', body })
      else await api('/alert-rules', { method: 'POST', body })
      onSaved()
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal modal-narrow" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{rule?.id ? '알림 규칙 수정' : '알림 규칙 추가'}</h3>
          <button type="button" className="linklike" onClick={onClose}>✕</button>
        </div>

        <label className="field field-block">
          <span className="field-label">규칙 이름</span>
          <input value={form.name} onChange={(e) => set('name', e.target.value)} />
        </label>

        <label className="field field-block">
          <span className="field-label">규칙 종류</span>
          <select value={form.kind} onChange={(e) => set('kind', e.target.value)}>
            {RULE_KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
          </select>
        </label>

        <label className="field field-block">
          <span className="field-label">대상</span>
          <input value={form.target_text} onChange={(e) => set('target_text', e.target.value)}
                 placeholder="전체 화자 또는 화자 별칭" />
        </label>

        <div className="form-grid">
          {fields.map((f) => (
            <label key={f} className="field field-block">
              <span className="field-label">{FIELD_META[f].label}</span>
              <input type="number" value={form[f]} min={FIELD_META[f].min}
                     max={FIELD_META[f].max} step={FIELD_META[f].step}
                     onChange={(e) => set(f, e.target.value)} />
            </label>
          ))}
        </div>

        <p className="muted small">
          저장되는 조건 문구: <b>{preview}</b> — 표시 문구는 입력값에서 자동으로 만들어지므로
          화면에 적힌 조건과 실제 동작이 어긋나지 않습니다.
        </p>
        {error && <p className="form-error">{error}</p>}

        <div className="modal-actions">
          <button type="button" onClick={onClose}>취소</button>
          <button type="button" className="primary" disabled={busy} onClick={save}>
            {busy ? '저장 중…' : '저장'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Alerts() {
  const [alerts, setAlerts] = useState([])
  const [openCount, setOpenCount] = useState(0)
  const [rules, setRules] = useState([])
  const [changes, setChanges] = useState([])
  const [disclaimer, setDisclaimer] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [statusFilter, setStatusFilter] = useState('all')
  const [sevFilter, setSevFilter] = useState('all')
  const [noteFor, setNoteFor] = useState(null)   // 메모를 편집 중인 알림 id
  const [noteText, setNoteText] = useState('')
  const [busyId, setBusyId] = useState(null)
  const [page, setPage] = useState(1)
  const [editing, setEditing] = useState(null)   // 규칙 수정/추가 폼 (null이면 닫힘)
  const [testResult, setTestResult] = useState(null)
  const [showChanges, setShowChanges] = useState(false)   // 규칙 변경 이력 보기

  const reload = useCallback(() => {
    setLoading(true)
    Promise.all([api('/alerts?limit=100'), api('/alert-rules'), api('/alert-rule-changes?limit=20')])
      .then(([a, r, c]) => {
        setAlerts(a.items || [])
        setOpenCount(a.open_count ?? 0)
        setDisclaimer(a.disclaimer || '')
        setRules(r.items || [])
        setChanges(c.items || [])
        setError(null)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(reload, [reload])

  const visible = useMemo(() => alerts.filter((a) => (
    (statusFilter === 'all' || a.status === statusFilter) &&
    (sevFilter === 'all' || a.severity === sevFilter)
  )), [alerts, statusFilter, sevFilter])

  // 필터를 바꾸면 1페이지로 되돌린다 — 3페이지에서 필터를 좁히면 빈 화면이 뜬다.
  useEffect(() => { setPage(1) }, [statusFilter, sevFilter])
  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE))
  const curPage = Math.min(page, pageCount)
  const pageItems = visible.slice((curPage - 1) * PAGE_SIZE, curPage * PAGE_SIZE)

  async function setStatus(alert, status) {
    setBusyId(alert.id)
    try {
      const updated = await api(`/alerts/${alert.id}`, { method: 'PATCH', body: { status } })
      setAlerts((cur) => cur.map((a) => (a.id === updated.id ? updated : a)))
      setOpenCount((c) => c + (status === 'open' ? 1 : (alert.status === 'open' ? -1 : 0)))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusyId(null)
    }
  }

  async function saveNote(alert) {
    setBusyId(alert.id)
    try {
      const updated = await api(`/alerts/${alert.id}`, { method: 'PATCH', body: { note: noteText } })
      setAlerts((cur) => cur.map((a) => (a.id === updated.id ? updated : a)))
      setNoteFor(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusyId(null)
    }
  }

  async function toggleRule(rule) {
    try {
      await api(`/alert-rules/${rule.id}`, { method: 'PATCH', body: { enabled: !rule.enabled } })
      reload()
    } catch (e) {
      setError(e.message)
    }
  }

  async function duplicateRule(rule) {
    try {
      await api(`/alert-rules/${rule.id}/duplicate`, { method: 'POST' })
      reload()
    } catch (e) {
      setError(e.message)
    }
  }

  async function testRule(rule) {
    setTestResult({ ruleId: rule.id, loading: true })
    try {
      const r = await api(`/alert-rules/${rule.id}/test`, { method: 'POST' })
      setTestResult({ ruleId: rule.id, data: r })
    } catch (e) {
      setTestResult({ ruleId: rule.id, error: e.message })
    }
  }

  return (
    <>
      <Topbar title="알림 센터 & 규칙 설정" />
      <main className="page">
        {error && (
          <div className="banner banner-error">
            <span>{error}</span>
            <button type="button" onClick={reload}>다시 시도</button>
          </div>
        )}
        {disclaimer && <p className="disclaimer">{disclaimer}</p>}

        <div className="alerts-cols">
          {/* 두 열은 각자 스크롤한다. 한 덩어리로 두면 알림이 쌓일수록 규칙 카드가
              화면 밖으로 밀려나 규칙을 보려면 알림을 전부 지나쳐 내려가야 한다. */}
          <div className="card col-card">
            <div className="card-head">
              <h3>알림 이력 {openCount > 0 && <span className="count-badge">미확인 {openCount}</span>}</h3>
              <button type="button" onClick={reload}>새로고침</button>
            </div>

            <div className="chip-filters">
              <span className="field-label">상태</span>
              {STATUS_FILTERS.map((f) => (
                <button key={f.key} type="button"
                        className={`chip${statusFilter === f.key ? ' on' : ''}`}
                        onClick={() => setStatusFilter(f.key)}>
                  {f.label}
                </button>
              ))}
              <span className="field-label">위험도</span>
              {SEV_FILTERS.map((f) => (
                <button key={f.key} type="button"
                        className={`chip${sevFilter === f.key ? ' on' : ''}`}
                        onClick={() => setSevFilter(f.key)}>
                  {f.label}
                </button>
              ))}
            </div>

            <div className="alert-list scroll-area">
              {loading && <StateBlock kind="loading" title="알림을 불러오는 중입니다" />}
              {!loading && visible.length === 0 && (
                <StateBlock
                  kind="empty"
                  title={alerts.length === 0 ? '아직 발생한 알림이 없습니다' : '이 조건에 해당하는 알림이 없습니다'}
                  detail={alerts.length === 0
                    ? '규칙이 켜져 있어도 조건을 넘기기 전까지는 알림이 생기지 않습니다. 오른쪽에서 규칙을 시험 실행해 지금 기준으로 울리는지 확인할 수 있습니다.'
                    : '상태·위험도 필터를 전체로 바꿔 보세요.'}
                />
              )}
              {pageItems.map((a) => (
                <div key={a.id} className={`alert-item sev-${a.severity || 'info'} st-${a.status}`}>
                  <div className="alert-main">
                    <p className="alert-title">
                      <span className={`sev-pill sev-${a.severity}`}>{SEV_LABEL[a.severity] || '주의'}</span>
                      <span className="kind-chip">{alertKindLabel(a)}</span>
                      {a.rule}
                    </p>
                    <p className="small">{a.message}</p>
                    <p className="muted small">
                      대상: {a.person_alias
                        ? `${a.person_alias}${a.person_room ? ` (${a.person_room})` : ''}`
                        : '미판정'} · {fmtDateTime(a.created_at)}
                    </p>
                    {/* 경계값의 출처를 함께 보여 준다. 임상 지침에서 온 값과
                        사용자가 정한 관찰 기준이 화면에서 구분되어야 한다. */}
                    {a.source && <p className="muted small">근거: {a.source}</p>}
                    {(a.assignee || a.acked_at) && (
                      <p className="muted small">
                        담당 {a.assignee || '—'}
                        {a.acked_at && ` · 확인 ${fmtDateTime(a.acked_at)}`}
                      </p>
                    )}
                    {a.note && (
                      <p className="alert-note"><span className="note-label">메모</span> {a.note}</p>
                    )}

                    {noteFor === a.id ? (
                      <div className="note-editor">
                        <textarea value={noteText} rows={2} placeholder="조치 내용을 적어 두세요"
                                  onChange={(e) => setNoteText(e.target.value)} />
                        <div className="row-actions">
                          <button type="button" className="primary" disabled={busyId === a.id}
                                  onClick={() => saveNote(a)}>저장</button>
                          <button type="button" onClick={() => setNoteFor(null)}>취소</button>
                        </div>
                      </div>
                    ) : (
                      <button type="button" className="linklike small"
                              onClick={() => { setNoteFor(a.id); setNoteText(a.note || '') }}>
                        {a.note ? '메모 수정' : '+ 메모 남기기'}
                      </button>
                    )}
                  </div>

                  <div className="alert-side">
                    <div className="status-switch" role="group" aria-label="알림 처리 상태">
                      {STATUS_FLOW.map((s) => (
                        <button
                          key={s}
                          type="button"
                          className={`status-opt${a.status === s ? ' on' : ''}`}
                          disabled={busyId === a.id}
                          onClick={() => setStatus(a, s)}
                        >
                          {STATUS_LABEL[s]}
                        </button>
                      ))}
                    </div>
                    <Link to={alertHistoryLink(a)} className="small">해당 시간 이력 보기 →</Link>
                  </div>
                </div>
              ))}
            </div>

            {visible.length > PAGE_SIZE && (
              <div className="pagination">
                <span className="spacer" />
                <button type="button" disabled={curPage <= 1}
                        onClick={() => setPage(curPage - 1)}>‹ 이전</button>
                <span>{curPage} / {pageCount} 페이지 · 총 {visible.length}건</span>
                <button type="button" disabled={curPage >= pageCount}
                        onClick={() => setPage(curPage + 1)}>다음 ›</button>
                <span className="spacer" />
              </div>
            )}
          </div>

          <div className="card col-card">
            <div className="card-head">
              <h3>알림 규칙</h3>
              <span className="head-actions">
                <button type="button" onClick={() => setShowChanges(true)}>변경 이력</button>
                <button type="button" className="primary" onClick={() => setEditing({})}>+ 규칙 추가</button>
              </span>
            </div>
            <div className="rule-list scroll-area">
              {loading && <StateBlock kind="loading" title="규칙을 불러오는 중입니다" />}
              {rules.map((r) => (
                <div key={r.id} className={`rule-card${r.enabled ? '' : ' off'}`}>
                  <div className="rule-head">
                    <p className="alert-title">{r.name}</p>
                    <button
                      type="button"
                      className={`toggle ${r.enabled ? 'on' : ''}`}
                      onClick={() => toggleRule(r)}
                    >
                      {r.enabled ? 'ON' : 'OFF'}
                    </button>
                  </div>
                  <p className="muted small">조건: {r.condition_text || '—'}</p>
                  {/* 평가 설명은 서버가 만든 문자열을 그대로 쓴다. 화면에서 다시
                      조립하면 규칙 종류가 늘어날 때마다 표시와 동작이 어긋난다. */}
                  <p className="muted small">
                    실제 평가: {r.evaluation_text}
                    {r.cooldown_minutes > 0 && ` · 재알림 ${r.cooldown_minutes}분 억제`}
                  </p>
                  <p className="muted small">
                    {r.clinical
                      ? '근거: 임상 지침 기준'
                      : '근거: 사용자 지정 관찰 기준 (의학적 경계값 아님)'}
                  </p>
                  <p className="muted small">대상: {r.target_text} · 수신: {r.channels_text}</p>

                  <div className="row-actions">
                    <button type="button" onClick={() => setEditing(r)}>수정</button>
                    <button type="button" onClick={() => duplicateRule(r)}>복제</button>
                    <button type="button" onClick={() => testRule(r)}>시험 실행</button>
                  </div>

                  {testResult?.ruleId === r.id && (
                    <div className="test-result">
                      {testResult.loading && <p className="muted small">시험 실행 중…</p>}
                      {testResult.error && <p className="form-error">{testResult.error}</p>}
                      {testResult.data && (
                        <>
                          <p className={testResult.data.would_fire ? 'test-fire' : 'test-quiet'}>
                            {testResult.data.would_fire
                              ? '지금 기준이면 알림이 울립니다'
                              : '지금 기준으로는 울리지 않습니다'}
                          </p>
                          {testResult.data.results.map((x) => (
                            <p key={String(x.person_id)} className="muted small">
                              {x.label}: {x.would_fire ? x.message : '조건 미달'}
                            </p>
                          ))}
                          <p className="muted small">{testResult.data.note}</p>
                          <button type="button" className="linklike small"
                                  onClick={() => setTestResult(null)}>결과 닫기</button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <p className="muted small note">
              ※ 알림의 “해당 시간 이력 보기”를 누르면 그 알림이 가리키는 시간 구간으로
              필터가 걸린 기침 이력이 열립니다.
            </p>
          </div>
        </div>
      </main>

      {editing && (
        <RuleForm
          rule={editing.id ? editing : null}
          onClose={() => setEditing(null)}
          onSaved={() => { setEditing(null); reload() }}
        />
      )}

      {/* 변경 이력은 평소에 볼 일이 없지만 "왜 안 울렸지"를 따질 때 반드시 필요하다.
          규칙 목록 옆에서 자리를 차지하지 않도록 눌러서 여는 방식으로 둔다. */}
      {showChanges && (
        <div className="modal-back" onClick={() => setShowChanges(false)}>
          <div className="modal modal-narrow" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>규칙 변경 이력</h3>
              <button type="button" className="linklike" onClick={() => setShowChanges(false)}>✕</button>
            </div>
            <p className="muted small">
              규칙을 켜고 끄거나 기준을 바꾼 기록입니다. 알림이 줄어든 이유가 기침이 줄어서인지,
              규칙이 바뀌어서인지 구분하는 데 씁니다.
            </p>
            {changes.length === 0 ? (
              <StateBlock kind="empty" title="아직 변경 이력이 없습니다"
                          detail="규칙을 수정·복제하거나 ON/OFF를 바꾸면 누가 무엇을 바꿨는지 여기에 쌓입니다." />
            ) : (
              <div className="change-list scroll-area">
                {changes.map((c) => (
                  <div key={c.id} className="change-item">
                    <p className="small"><b>{c.rule_name}</b> — {c.summary}</p>
                    <p className="muted small">{c.actor} · {fmtDateTime(c.created_at)}</p>
                  </div>
                ))}
              </div>
            )}
            <div className="modal-actions">
              <button type="button" className="primary"
                      onClick={() => setShowChanges(false)}>닫기</button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

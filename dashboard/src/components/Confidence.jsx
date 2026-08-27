import { confidence } from '../lib/format'

/** 신뢰도(코사인 유사도)를 "88% · 높음" + 막대로 보여 준다.
 *  숫자 0.88만으로는 그것이 좋은 값인지 알 수 없어 등급을 함께 붙인다. */
export default function Confidence({ value, source, bar = true }) {
  // 사람이 손으로 지정한 화자에는 확신도가 없다. 저장된 similarity는 지정 이전
  // 모델 점수라 지금 라벨과 무관하므로, 숫자를 그대로 보여 주면 거짓말이 된다.
  if (source === 'manual') {
    return (
      <span className="conf conf-manual"
            title="사람이 직접 지정한 화자입니다. 모델 점수는 지정 이전 값이라 지금 라벨의 확신도가 아닙니다.">
        사람이 지정
      </span>
    )
  }
  const c = confidence(value)
  if (c.pct == null) {
    return <span className="conf conf-none" title="비교할 등록 화자가 없었습니다">비교 없음</span>
  }
  return (
    <span className={`conf conf-${c.level}`}>
      {bar && (
        <span className="conf-track" aria-hidden="true">
          <span className="conf-fill" style={{ width: `${c.pct}%` }} />
        </span>
      )}
      <span className="conf-text">{c.pct}% · {c.label}</span>
    </span>
  )
}

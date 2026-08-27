/** 로딩·오류·빈 상태를 서로 다르게 보여 준다.
 *
 *  셋을 같은 "데이터가 없습니다"로 뭉뚱그리면, 서버가 죽은 것과 오늘 기침이 없는 것이
 *  화면에서 구분되지 않는다. 빈 상태에는 다음에 할 일을 함께 적는다.
 */
export default function StateBlock({ kind, title, detail, action }) {
  return (
    <div className={`state-block state-${kind}`}>
      <p className="state-title">{title}</p>
      {detail && <p className="state-detail">{detail}</p>}
      {action}
    </div>
  )
}

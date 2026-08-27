# 외부 데이터셋 — 출처와 라이선스

대회 제출물에 이 내용을 그대로 인용할 것. **제출 전 대회 규정에서 외부 데이터셋 사용 가부를 반드시 확인해야 한다** (아직 미확인).

## Coswara

| 항목 | 내용 |
|---|---|
| 제공 | LEAP Lab, Indian Institute of Science (IISc), Bangalore, India |
| 라이선스 | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| 저작권 | Copyright (c) 2021 LEAP Lab, Indian Institute of Science, Bangalore, India |
| 저장소 | https://github.com/iiscleap/Coswara-Data |
| 규모 | 참가자 2,746명, 1인당 9종 녹음(호흡 2·기침 2·모음 3·숫자세기 2) |
| 수집 기간 | 2020-04 ~ 2022-02 |
| 우리가 쓰는 부분 | `cough-heavy.wav`, `cough-shallow.wav` 만 사용 (화자 ID로 묶인 기침 2종) |

CC BY 4.0은 출처 표기를 조건으로 복제·변형·상업적 이용을 허용한다. 보고서에는 위 저작권 표기와 저장소 링크를 함께 싣는다.

### 왜 이 데이터가 필요한가

VoxCeleb 사전학습 ECAPA를 **그대로** 쓰면 기침에서 화자를 구분하지 못한다 (2026-08-24 자체 측정: EER 55%, 동전 던지기 수준). 선행 연구는 같은 ECAPA를 **기침 데이터로 적응시켜** EER 13.39%를 얻었고, 그때 사용한 데이터가 Coswara다.

우리 데이터는 화자 3명(s01·s02·x01)뿐이라 적응 학습이 불가능하다. Coswara의 수백~수천 화자가 그 역할을 한다.

### ⚠ 구조적 한계 — 세션이 하나뿐이다 (2026-08-27 확인)

내려받은 11개 날짜 폴더에서 **고유 참가자 980명 / 폴더 980개**로, 한 참가자가 한 날짜에만
등장한다. 참가자 폴더의 `cough-heavy.wav`와 `cough-shallow.wav`는 같은 녹음 세션에서 연속
녹음된 것이다.

따라서 이 데이터로는 **"같은 화자의 다른 날 녹음"이라는 양성 쌍을 만들 수 없다.** 투영층
학습이 heavy/shallow 쌍을 양성으로 썼으므로, 결과적으로 "같은 세션이면 가깝게"를 보상하도록
학습됐고 세션 불변성은 배울 수 없었다. 우리 실사용 데이터에서 임베딩이 화자가 아니라 녹음
블록을 인코딩한 현상(`보고서_기술자료_수치모음.md` 6절 ⑤)과 직접 연결된다.

선행 연구의 EER 13.39%도 같은 데이터셋에서 같은 쌍 구성을 썼다면 동일한 교란을 안고 있을
수 있다. **인용할 때 세션 분리 여부를 함께 확인할 것.**

### 참고 문헌

Li, H., Wang, S., He, L. et al. *Beyond fingerprint and voice print: cough sequence sound for identity verification.* npj Health Systems 3, 60 (2026).
https://www.nature.com/articles/s44401-026-00111-1

- Coswara 사용, VoxCeleb 사전학습 ECAPA-TDNN을 기침에 맞게 적응
- **EER 13.39%**, F1 0.8676. 관련 연구 성능 범위는 EER 10~15%
- 음성 기반 화자 인증(VoxCeleb, EER 2.87~3.22%)보다 크게 어려운 문제임을 명시

**기대치**: 최고 수준 연구가 EER 13% 대다. 7~8회 중 1회는 틀린다는 뜻이고, 그마저 Coswara 자체 도메인 기준이다. 우리 I2S 마이크·50cm 환경으로의 전이는 별도 검증이 필요하다.

## 로컬 보관 위치

`~/datasets/Coswara/` (저장소 밖, iCloud 밖)
- `coughs/<날짜>/<참가자ID>/cough-{heavy,shallow}.wav` — 사용 대상
- `combined_data.csv` — 참가자 메타데이터
- `LICENSE.md` — 원본 라이선스 전문
- `fetch_subset.py` — 부분 다운로드 스크립트 (전체 16GB 중 필요분만)

원본 오디오는 저장소에 커밋하지 않는다. 용량 문제이기도 하고, 재배포 시 라이선스 표기 의무가 따라붙기 때문이다.

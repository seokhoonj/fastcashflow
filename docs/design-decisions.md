# Design decisions

가정 워크북 구조와 명명 규칙에 대해 결정된 내용과 그 이유. 각 항목은
나중에 다시 같은 토론을 반복하지 않도록 **근거**까지 기록합니다.

## 1. 단일 워크북 (assumptions.xlsx)

**결정**: rate tables (7시트) + segments + riders = 총 9시트를 한 워크북에
담는다. 이전 v1 reader가 썼던 두 파일 분리 (`sample_tables.xlsx` +
`sample_basis.xlsx`)는 폐기.

**근거**:
- 두 파일 분리를 정당화하는 시나리오 — (1) 한 tables를 여러 basis가 공유,
  (2) tables는 분기 갱신/basis는 월 갱신, (3) 거대 워크북 회피 — 가 현
  프로젝트 규모에서 모두 약함.
- 현실: 한 portfolio = 한 valuation 입력 묶음 → 한 파일에 같이 두는 게
  자연.
- 사용자 인터페이스 단순: `read_assumptions(path)` 단일 경로.

**대안**:
- 외부 table 라이브러리 (대형 valuation 플랫폼들이 갖는 inventory 패턴)
  는 fastcashflow 규모를 넘는다. yagni.

---

## 2. 시트명 `segments` (이전엔 `basis`)

**결정**: (product, channel) 매핑 시트의 이름은 `segments`. `basis`는
쓰지 않음.

**근거**:
- 보험계리에서 "basis" (산출기초율) 는 mortality·lapse·expense·discount
  까지 묶은 **valuation 입력 전체**를 가리키는 용어. 한 시트의 이름으로
  들어올리면 의미 충돌.
- 그 시트의 실제 역할은 "(product, channel) → 어느 table을 쓸지 + 스칼라
  파라미터 배정" 즉 **매핑 / configuration**. 기초율 아님.
- 후보들:
  - `mapping_tables`: `_tables` 접미사가 "여러 named table의 registry"
    의미로 일관되게 쓰여 (mortality_tables 등) 단일 매핑 시트엔 의미 부정확
  - `mapping_table` (단수): 워크북 내 유일한 단수형 — 패턴 일관성 깨짐
  - `segments`: `riders`와 같은 패턴 (복수형 일반명사, 행의 정체를 직시).
    "각 행 = 한 segment". 채택.

**관련 코드**: 리더 내부에서도 `segments`라는 변수명을 이미 사용 중.

---

## 3. 파일명 `assumptions.xlsx` (이전엔 `basis.xlsx` 후보)

**결정**: 워크북 파일명은 `assumptions.xlsx`. Python 클래스 `Assumptions`,
모듈 `assumptions.py`, 함수 `read_assumptions()`와 이름 일치.

**근거**:
- 산출기초율 (좁은 의미 "basis") 은 시트 단위 → 파일 이름으로 들어올리면
  자기참조 발생 (`basis.xlsx` 안에 `basis` 시트가 9개 중 1개로 들어가는
  꼴, 오해 유발).
- "assumptions" 는 더 넓은 단어로 9시트 전부를 자연스럽게 덮음.
- 클래스/모듈/함수 이름과의 일관성이 우선.

---

## 4. 컬럼 값 명명 규칙

**결정**:
| 컬럼 | 규칙 | 예 |
|---|---|---|
| `product` | snake_case 소문자 | `term_a`, `whole_life` |
| `channel` | ALL UPPERCASE 약어 | `GA`, `FC`, `BANCA` |
| `table_id` | SCREAMING_SNAKE_CASE | `MORT_STD`, `LAPSE_GA` |
| `rider_code` | snake_case 소문자 | `dth_main`, `hosp` |
| `type` | snake_case 소문자 | `death_main`, `morbidity` |
| 컬럼 헤더 | snake_case 소문자 | `expense_acquisition`, `mortality_cv` |

**근거**:
- channel은 업계 관용 약어 (General Agency, Financial Consultant)
  대문자 표기 보존이 정보 손실 적음.
- table_id는 코드 상수와 비슷한 named reference이라
  SCREAMING_SNAKE_CASE가 데이터 값과 시각적으로 구분됨.
- product / rider_code는 model points / 코드 enum과 매핑되므로 같은
  snake_case 규칙.

---

## 5. Column semantics (`rate` / `amount` / `factor`)

**결정**: 컬럼 이름이 값의 의미를 표시.

| 컬럼명 | 의미 | 단위 | 시트 |
|---|---|---|---|
| `rate` | 확률 / 발생률 / 환산률 | 무차원 (0~1) | mortality, rider_rate, waiver, lapse, discount, inflation |
| `amount` | 화폐 금액 | 원 (또는 portfolio 통화) | maintenance |
| `factor` | 곱셈자 (multiplier) | 무차원 (보통 ~1.0) | **현재 없음**; 향후 A/E layer 도입 시 (Task #8) |

**근거**:
- 의미가 다른 값에 같은 컬럼명을 쓰면 reader가 단위 처리를 잘못할
  소지. `rate` 가정 → 0~1 검증 / `amount` 가정 → 통화 처리.
- 기존 `maintenance_tables.rate` (값이 60,000 같은 원 단위)는 명칭
  오류. `amount`로 rename 확정.

---

## 6. 위험률 vs 발생률 가정 (best estimate) — IFRS 17 정합

**결정**: 워크북의 rate table들은 **best-estimate 발생률 가정**으로
간주한다. "위험률" (pricing 기초율) 이라는 표현은 docs/naming에서 제거.

**근거**:
- IFRS 17 Sec. 33, B37: BEL은 편의 없는 (unbiased) 최선추정으로 측정.
  pricing 기초율 (보수적 마진 포함) 을 그대로 쓰면 BEL 과대 / CSM 부풀림.
- 한국 산업 실무: 가정관리시스템에서 `위험률 × A/E ratio = best estimate`
  를 외부에서 계산한 결과가 valuation 엔진의 입력. fastcashflow는 그
  결과를 받는다.
- pricing 위험률 마진은 `level_premium` 안에 이미 녹아 있어서, `premium_cf
  > E(claim_cf)` 가 자연스럽게 발생 — 이 차이가 CSM의 원천.

---

## 7. A/E factor 레이어 — 패턴 1 vs 2

**결정**: 워크북에 A/E factor 시트를 **선택적 레이어로 추가** (Task #8,
미구현). 기본은 없음 (factor 1.0). 사용자가 (a) 외부에서 위험률 × A/E를
미리 곱해 base rate에 넣거나 (b) base + ae_factors 시트 둘 다 채워서
엔진 런타임 곱셈 — 둘 다 지원.

**근거**:
- 두 워크플로우:
  - **패턴 1 (외부 calibration)**: 가정관리시스템에서 사전 곱셈, 엔진엔
    final best-estimate만 입력. 한국 자체개발 시스템에 많은 형태.
  - **패턴 2 (런타임 곱셈)**: 엔진이 base + multiplier 둘 다 받음.
    상용 valuation 플랫폼들의 기본 동작.
- 둘 다 산업에 존재 → 한쪽에 치우치지 않게 양쪽 다 지원.
- 단, scalar A/E는 (age × duration × channel) 패턴을 못 잡으므로 ae_factors
  시트도 axis-flex 설계로 (Task #7과 같은 schema-detection).

**선행 토론 / 폐기 안**:
- 처음엔 "fastcashflow의 (product × channel × age) per-segment table 설계가
  이미 A/E를 함의" 라고 했으나, 산업 정합 / 운영 편의 (sensitivity 한 칸
  수정) 측면에서 별도 레이어 지원이 가치 있다는 결론.

---

## 8. Maximize 설계 원칙 — Layered, optional

**결정**: 엔진의 callable은 최대 차원 (sex, issue_age, duration,
calendar_year)을 받는다. 워크북 입력층은 base table (필수) + A/E factor
(optional) + age_shift (optional) + improvement (optional) 의 **4 레이어
조합**으로 표현. 사용자가 채운 레이어만 활성, 안 채우면 no-op (1.0 / 0).

**근거**:
- "엔진은 풍부하게, 입력은 단순~복잡까지 사용자 선택" 원칙.
- 각 레이어 의미:
  - **base** (Task #7): 발생률의 main shape. axis-flex로 1차원 (scalar)
    부터 4차원 ((sex, issue_age, duration)) 까지.
  - **A/E factor** (Task #8): 런타임 multiplier. base와 같은 axis-flex.
  - **age_shift** (Task #9): 정수 shift. table 재사용 시 cohort 보정.
  - **improvement** (Task #10): 연도별 mortality improvement scale (산업 표준 패턴).
- 4개 layer 각각이 실무에서 흔히 보이는 워크어라운드 (위험률 × A/E를
  외부 Excel에서 미리 곱셈, cohort마다 테이블 복사, improvement scale
  사전 곱셈, select-and-ultimate 별도 시트 분리) 를 시스템 내부 1급
  표현으로 끌어들임. 입력 안 채우면 단순 base 모델과 동일하게 동작.

---

## 9. Sample identifier — generic placeholder

**결정**: 샘플 워크북의 table_id는 generic placeholder.
- `MORT_STD` (기존 `KMT_STD`)
- `DISC_STD` (기존 `RFR_2025`)
- `WVR_STD`, `MAINT_STD`, `INFL_STD`, `LAPSE_GA`, `LAPSE_FC`,
  `HOSP_STD`, `CANCER_STD`, `ADB_STD` 등은 이미 generic

**근거**:
- `KMT_STD` (Korean Mortality Table Standard 패턴) 는 placeholder인데
  실제 한국 산업 표준 약어 (예: KIDI 경험생명표 9회) 와 혼동 가능. "이게
  KELT-9인가" 오해 회피.
- `RFR_2025` 의 "2025"는 vintage 표시지만 sample에 박혀 있으면 시간
  지나서 오래된 데이터로 보임. `DISC_STD` 로 단순화.
- 실 사용 시엔 회사 경험분석 / 감독원 발표 자료의 정확한 식별자로 교체
  (docs에 명시).

---

## 10. 도입 결정 (확정) vs 향후 검토 (보류)

**확정**:
- 1~9 위 항목 전부
- 워크북 단일화, `segments` 시트명, snake_case 규칙, column semantics,
  best-estimate 정의, layered optional 설계 골격

**보류 (Task로 등록, 별도 구현)**:
- Task #7: base table axis-flex 리더 (schema-detecting)
- Task #8: A/E factor 레이어
- Task #9: age_shift 컬럼
- Task #10: improvement 레이어
- Task #1: discount / inflation / maintenance curve화
- Task #5: curves.py / numerics.py 레이어 분리 (완료)
- Task #2: ModelPoints product/channel + value_segmented helper

**검토 안 함 (yagni)**:
- 워크북 두 파일 분리 (#1 결정)
- "위험률" 그대로 입력 (#6 결정 — best-estimate이 표준)
- 손해율 테이블 입력 (별도 토론, GMM은 발생률 × 금액 직접 계산)
- VFA 파라미터 (`fund_fee`, `guaranteed_credit_rate`) Assumptions 잔류
  ([[vfa-param-relocation]] 메모 — 별도 refactor)

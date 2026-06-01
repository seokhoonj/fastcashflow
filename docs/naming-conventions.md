# Naming conventions

`fastcashflow`의 입력 워크북 (`assumptions.xlsx`)과 그에 매핑되는 코드의
명명 규칙. 보험계리 용어와 충돌이 일어나지 않도록 굳혀 둔 약속. 각 결정의
**근거**는 `docs/design-decisions.md` 참조.

## File & sheet

| 항목 | 규칙 | 예 | 비고 |
|---|---|---|---|
| Workbook 파일명 | `assumptions.xlsx` | `sample_assumptions.xlsx` | 단일 파일 (segments + coverages + 7 rate tables 통합) |
| Rate-table registry 시트 | `<kind>_tables` (복수) | `mortality_tables`, `lapse_tables`, `discount_tables` | 한 시트에 같은 종류의 named table 여러 개 (`table_id` 컬럼으로 그룹) |
| Mapping/configuration 시트 | 단수·복수 일반명사, 접미사 없음 | `segments`, `coverages` | 각 행이 한 설정 entry |

## Sheets in `assumptions.xlsx`

| 시트 | 역할 |
|---|---|
| `segments` | `(product_code, channel_code)` 별 — 어느 rate table을 쓸지 + 스칼라 파라미터 (`ra_confidence`, `*_cv`, optional `*_age_shift`, `expense_table` 등). `defaults` 행이 fallback |
| `coverages` | rate-driven 담보 registry: `coverage_code → rate_table`. 모든 상품 공통 (product 별로 다른 calibration 필요시 `CANCER_HEALTH`, `CANCER_WHOLELIFE` 처럼 code 분리). `coverage_name` / `calculation_method` 은 별도 `calculation_methods.csv` (담보별 산출방식) 에 |
| `mortality_tables` | 사망 발생률 가정 (`table_id` × `sex` × `age` → `rate`) |
| `incidence_rate_tables` | 특약 발생률 가정 (구조 동일) |
| `waiver_tables` | 납입면제 발생률 가정 (구조 동일) |
| `lapse_tables` | 해지율 가정 (`table_id` × `duration` → `rate`) |
| `expense_tables` | 사업비 ledger (`table_id` × `expense_type` × `basis` × `value`). `basis` 가 alpha_fixed / alpha_pro_rata / beta_pro_rata / gamma_fixed / lae_pro_rata 등 kernel-side primitive 를 결정 |
| `discount_tables` | 할인율 곡선 (`table_id` × `year` → `rate`; locked-in, Sec. 36) |
| `inflation_tables` | 사업비 인플레이션 곡선 (`table_id` × `year` → `rate`) |
| `surrender_value_tables` (optional) | 해약환급금 비율 곡선 (`table_id` × `duration_month` → `factor`) |
| `ae_factors` (optional) | A/E factor — `(product_code × channel_code × coverage_code)` + 옵션 axes → `factor`. base rate에 런타임 곱셈 |
| `improvement_tables` (optional) | mortality improvement 곡선 (`table_id` × `year` → `factor`). `segments`의 `mortality_improvement_table` 컬럼이 참조 |

별도 파일:

| 파일 | 역할 |
|---|---|
| `calculation_methods.csv` | 담보별 산출방식 (`coverage_code → calculation_method` 분류). 5종 fixed pattern (DEATH / MORBIDITY / DIAGNOSIS / ANNUITY / MATURITY). `assumptions.xlsx` 와 분리 — 신담보 추가 시에만 손댐 |
| `inforce_state.csv` (optional) | 결산 시점 보유계약 상태 (`mp_id`, `elapsed_months`, `count`, `prior_csm`, `lock_in_rate`) |

## Column headers

전부 **소문자 snake_case**. 예: `product_code`, `channel_code`,
`coverage_code`, `rate_table`, `mortality_table`, `expense_table`,
`ra_confidence`, `mortality_cv`, `table_id`, `sex`, `age`, `duration`,
`year`, `rate`, `amount`, `factor`, `expense_type`, `basis`, `value`.

## Column semantics (`rate` / `amount` / `factor`)

같은 워크북 안에 의미가 다른 값들을 섞지 않기 위해, 컬럼 이름이 값의
종류와 단위를 표시합니다.

| 컬럼명 | 의미 | 단위 | 값 범위 | 사용 시트 |
|---|---|---|---|---|
| `rate` | 확률 / 발생률 / 환산률 | 무차원 (decimal) | 0~1 (또는 작은 양수) | mortality, incidence_rate, waiver, lapse, discount, inflation |
| `amount` | 화폐 금액 | 원 (또는 portfolio 통화) | 양의 실수 | maintenance |
| `factor` | 곱셈자 (multiplier) | 무차원 | 보통 ~1.0 | **현재 없음** (장래 A/E factor 레이어 도입 시 예약) |

리더가 `rate`는 확률 검증, `amount`는 통화 처리, `factor`는 곱셈자 처리를
할 수 있도록 컬럼명이 의미를 운반합니다.

## Value conventions

| 컬럼 | 규칙 | 예 | 이유 |
|---|---|---|---|
| `product_code` | SCREAMING_SNAKE_CASE | `TERM_LIFE_A`, `WHOLE_LIFE_A`, `HEALTH_A` | enum-like 외부 식별자 |
| `channel_code` | ALL UPPERCASE 약어 | `GA`, `FC`, `TM` | 업계 관용 약어 (General Agency, Financial Consultant, Telemarketing) |
| `table_id` | SCREAMING_SNAKE_CASE 풀네임 | `MORTALITY_STD`, `LAPSE_GA`, `DISCOUNT_STD`, `INPATIENT_STD`, `ADB_STD` | named reference. 줄임말 안 씀 (`MORT_STD` 같은 abbreviation 지양). 단 industry-universal abbr 인 `ADB` 같은 매우 짧은 것은 예외 |
| `coverage_code` | SCREAMING_SNAKE_CASE 풀네임 | `DEATH`, `INPATIENT`, `CANCER`, `MATURITY`, `ANNUITY`, `ADB` | enum-like 식별자. 사용자 카탈로그 — 엔진 reserved 코드 없음 |
| `calculation_method` | SCREAMING_SNAKE_CASE 풀네임 | `DEATH`, `MORBIDITY`, `DIAGNOSIS`, `ANNUITY`, `MATURITY` | **engine 의 cash flow 산출방식 routing key**. 5 개 고정. 자세한 각 산출방식별 계산은 `assumptions-format.md` 의 coverages 시트 섹션 참조 |
| `state` | SCREAMING_SNAKE_CASE | `ACTIVE`, `WAIVER`, `PAIDUP` | enum-like, 정책 status |
| `defaults` (특수 product 값) | 소문자 단어 | `defaults` | segments 시트의 fallback 행 marker (값 아닌 keyword) |

## Python code

| 항목 | 규칙 | 예 |
|---|---|---|
| 모듈 | snake_case | `assumptions.py`, `projection.py`, `engine.py` |
| 클래스 | PascalCase | `Basis`, `ModelPoints`, `Cashflows`, `Measurement` |
| 함수 / 변수 | snake_case | `read_basis`, `discount_monthly_curve`, `n_time` |
| 모듈 private | leading underscore | `_project_kernel`, `_norm_ppf`, `_axis_tables` |
| 상수 | UPPER_SNAKE_CASE | `RISK_MORTALITY`, `RISK_MORBIDITY`, `STATE_ACTIVE`, `STATE_MODELS` (registry dict) |

## 데이터 ID와 Python 코드 enum 의 일관성

워크북 값 (예: `coverage_code = "DEATH"`, `calculation_method = "MORBIDITY"`) 이
Python 상수 / enum (예: `CalculationMethod.MORBIDITY == "MORBIDITY"`)
과 **bit-exact** 일치합니다. 모두 SCREAMING_SNAKE_CASE 풀네임.
`coverage_code` 는 사용자 카탈로그 — 엔진이 reserved 코드를 가지고
있지 않으므로 회사가 자유롭게 짓습니다 (`DEATH` 은 샘플 관용).

`enum-like 식별자 family`:

- `product_code`, `channel_code`, `table_id`, `coverage_code`, `calculation_method`,
  `state`, `state_model` — 모두 외부 식별자 / 코드 상수 family. SCREAMING_SNAKE_CASE.
- 줄임말 안 씀 (`MORT` 가 아닌 `MORTALITY`, `HOSP` 가 아닌 `INPATIENT` 등).
  단 industry-universal 한 매우 짧은 abbr 인 `ADB` 정도 예외.

`벤더 데이터 / 컬럼명` (소문자 snake_case):

- 컬럼 헤더 (`product_code`, `coverage_code`, `rate_table`, `level_premium`,
  `count`, `mp_id` 등) — 표/스키마 식별자, 행 안의 값들과 시각적 구분 위해
  소문자.

## Sample workbook의 식별자는 generic placeholder

번들 sample의 `MORTALITY_STD`, `DISCOUNT_STD`, `LAPSE_GA`, `INPATIENT_STD` 같은
ID는 **generic placeholder** 입니다. 실제 한국 산업 표준 (예: 보험개발원 KIDI
경험생명표 9회) 의 식별자가 아닙니다.

실 사용 시에는 회사가 채택한 위험률 / 발생률 가정의 정확한 식별자로
교체하세요 (예: 회사 경험분석 결과 기반의 `MORTALITY_2024_M_STD`,
감독원 발표 RFR 곡선 기반의 `RFR_2025_12_KOR` 등).

## 보험계리 용어와의 관계

| 워크북 표현 | 보험계리 의미 | 주의 |
|---|---|---|
| `assumptions.xlsx` 파일 전체 | 산출기초율 (valuation basis) | "basis"는 좁은 시트명으로 쓰지 않음 |
| `*_tables` 시트들 (mortality, lapse 등) | **best-estimate 발생률 가정** (IFRS 17 Sec. 33, B37) | pricing 위험률 (산출기초율의 보수적 측면) 그대로 입력하면 BEL 과대 / CSM 부풀림 |
| `segments` 시트 | 가정 매핑 / 상품·채널 배정 | 발생률 자체 아님 — 어떤 테이블을 쓸지의 indirection |

IFRS 17 GMM의 BEL은 편의 없는 최선추정으로 측정해야 합니다 (Sec. 33).
워크북에 들어가는 mortality/morbidity/lapse 등은 회사가 외부에서
경험분석·A/E 보정을 거쳐 만든 **best-estimate 발생률 가정**입니다.
pricing 마진은 `level_premium`에 이미 녹아 있어, BEL의 입력이 best
estimate일 때 `premium_cf > E(claim_cf)`가 자연스럽게 발생하고, 이 차이가
CSM의 원천이 됩니다.

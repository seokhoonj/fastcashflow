---
orphan: true
---

# Naming conventions

`fastcashflow`의 입력 워크북 (`basis.xlsx`)과 그에 매핑되는 코드의
명명 규칙. 보험계리 용어와 충돌이 일어나지 않도록 굳혀 둔 약속.

## File & sheet

| 항목 | 규칙 | 예 | 비고 |
|---|---|---|---|
| Workbook 파일명 | `basis.xlsx` | `sample_basis.xlsx` | 단일 파일 (segments + coverages + 7 rate tables 통합) |
| Rate-table registry 시트 | `<kind>_tables` (복수) | `mortality_tables`, `lapse_tables`, `discount_tables` | 한 시트에 같은 종류의 named table 여러 개 (`table_id` 컬럼으로 그룹) |
| Mapping/configuration 시트 | 단수·복수 일반명사, 접미사 없음 | `segments`, `coverages` | 각 행이 한 설정 entry |

## Sheets in `basis.xlsx`

| 시트 | 역할 |
|---|---|
| `segments` | `(product, channel)` 별 — 어느 rate table을 쓸지 + 스칼라 파라미터 (`ra_confidence`, `*_cv`, optional `*_age_shift`, `expense_table` 등). `_DEFAULTS` 행이 fallback |
| `coverages` | rate-driven 담보 registry: `coverage → rate_table`. 모든 상품 공통 (product 별로 다른 calibration 필요시 `CANCER_HEALTH`, `CANCER_WHOLELIFE` 처럼 코드 분리). `calculation_method` 은 별도 `calculation_methods.csv` (담보별 산출방법) 에 |
| `mortality_tables` | 사망 발생률 가정 (`table_id` × `sex` × `age` → `rate`) |
| `incidence_rate_tables` | 특약 발생률 가정 (구조 동일) |
| `waiver_tables` | 납입면제 발생률 가정 (구조 동일) |
| `lapse_tables` | 해지율 가정 (`table_id` × `duration` → `rate`) |
| `expense_tables` | 사업비 ledger (`table_id` × `expense_type` × `basis` × `value`). `basis` 가 alpha_fixed / alpha_pro_rata / beta_pro_rata / gamma_fixed / lae_pro_rata / surrender_value_pro_rata 등 kernel-side primitive 를 결정 |
| `discount_tables` | 할인율 곡선 (`table_id` × `year` → `rate`; locked-in, 문단 36) |
| `inflation_tables` | 사업비 인플레이션 곡선 (`table_id` × `year` → `rate`) |
| `surrender_value_tables` (optional) | 해약환급금 곡선 (`table_id` × `duration_month` → `factor` 환급률 또는 `amount` 금액; segments 의 `surrender_value_basis` 가 해석) |
| `ae_factors` (optional) | A/E factor — `(product × channel × coverage)` + 옵션 axes → `factor`. base rate에 런타임 곱셈 |
| `improvement_tables` (optional) | mortality improvement 곡선 (`table_id` × `year` → `factor`). `segments`의 `mortality_improvement_table` 컬럼이 참조 |

별도 파일:

| 파일 | 역할 |
|---|---|
| `calculation_methods.csv` | 담보별 산출방법 (`coverage → calculation_method` 분류). 5종 fixed pattern (DEATH / MORBIDITY / DIAGNOSIS / ANNUITY / MATURITY). `basis.xlsx` 와 분리 — 신담보 추가 시에만 손댐 |
| `inforce_state.csv` (optional) | 결산 시점 보유계약 상태 (`mp_id`, `elapsed_months`, `count`, `prior_csm`, `lock_in_rate`, `prior_count`; 선택 `prior_loss_component`, VFA는 `account_value` / `prior_account_value`) |

## Column headers

전부 **소문자 snake_case**. 예: `product`, `channel`,
`coverage`, `rate_table`, `mortality_table`, `expense_table`,
`ra_confidence`, `mortality_cv`, `table_id`, `sex`, `age`, `duration`,
`year`, `rate`, `factor`, `expense_type`, `basis`, `value`.

## Column semantics (`rate` / `amount` / `factor`)

같은 워크북 안에 의미가 다른 값들을 섞지 않기 위해, 컬럼 이름이 값의
종류와 단위를 표시합니다.

| 컬럼명 | 의미 | 단위 | 값 범위 | 사용 시트 |
|---|---|---|---|---|
| `rate` | 확률 / 발생률 / 환산률 | 무차원 (decimal) | 0~1 (또는 작은 양수) | mortality, incidence_rate, waiver, lapse, discount, inflation |
| `factor` | 곱셈자 (multiplier) | 무차원 | 보통 ~1.0 | surrender_value (factor 모드), ae_factors, improvement_tables |
| `amount` | 통화 금액 | 통화 | 0 이상 | surrender_value (amount 모드) |
| `value` | 통화 금액 (ledger) | 통화 | 0 이상 | expense_tables |

리더가 `rate`는 확률 검증, `amount`는 통화 처리, `factor`는 곱셈자 처리를
할 수 있도록 컬럼명이 의미를 운반합니다.

## 측정 결과의 필드 접미사 — `_cf` / `_path` / `_cv`

위는 *입력* 워크북 컬럼이고, 아래는 측정 *결과* 객체 (`gmm.measure` /
`paa.measure` / `vfa.measure` 가 돌려주는 `Cashflows`·`Measurement`) 의 필드
접미사다. **`_cf` 와 `_cv` 는 글자가 비슷하지만 완전히 다른 것**이라 먼저 못을
박는다:

| 접미사 | 무엇 | 단위 | 예 |
|---|---|---|---|
| `_cf` | **cash flow** — 그 달의 현금흐름 (돈) | 통화 / 월 | `premium_cf`, `mortality_cf`, `morbidity_cf` |
| `_path` | **stock** — 매월 롤포워드되는 잔액 궤적 | 통화 (잔액) | `lic_path`, `lrc_path`, `csm_path`, `ra_path` |
| (접미사 없음) | **count** — 사람/계약 수 (돈 아님) | 명 / 건 | `inforce`, `deaths` |
| `_cv` | **coefficient of variation** — RA 가 쓰는 변동계수 (위험의 불확실성) | 무차원 | `mortality_cv`, `morbidity_cv`, `longevity_cv` |

### `_cf` — 현금흐름 레그 (flow)

`Cashflows` 의 `(n_mp, n_time)` 배열들. 각 월에 발생하는 돈의 유입/유출이다
(부채 관점, 유출 양수):

- `premium_cf` — 보험료 유입 (부채 감소)
- `mortality_cf` / `morbidity_cf` — 사망위험 / 질병위험 청구 (RA risk class
  로 갈림; 아래 "위험조정 (RA) 와 `_cv`" 참조). 진단 (`DIAGNOSIS`) 청구는 자기
  레그가 없고 morbidity 위험으로 `morbidity_cf` 에 합쳐진다
- `annuity_cf` / `disability_cf` — 생존연금 / 장해소득 지급
- `maturity_cf` — 만기보험금 (만기 1회), `surrender_cf` — 해약환급, `expense_cf` — 사업비

계산은 "in-force × 율 × 보장금액" 꼴이다. 예: 어느 달의 `mortality_cf` =
유지 건수 × 그 달 사망률 × 사망보험금. BEL 은 이 레그들의 현가 합:
`BEL = PV(claims) - PV(premiums)`.

### `_path` — 잔액 궤적 (stock)

flow 가 아니라 *잔액*이다 — 매월 롤포워드되는 부채/마진의 시점별 값
(`(n_mp, n_time+1)`, 마지막 열은 잔존 tail). `lic_path` (발생사고부채),
`lrc_path` (잔여보장부채), `csm_path` (보험계약마진), `ra_path` (위험조정 잔액).
flow 를 쌓아 만든 누적량이라 `_cf` 와 단위는 같아도 (돈) 성격이 다르다.

### 위험조정 (RA) 와 `_cv`

`_cv` 는 현금흐름이 *아니라* 각 보험위험의 **변동계수** (coefficient of
variation = 표준편차/평균, 청구의 불확실성) 다. `Basis` 의 입력값이고
(`mortality_cv` / `morbidity_cv` / `longevity_cv` / `disability_cv`, VFA 는
`expense_cv` 도), IFRS 17 위험조정 (RA, 비금융위험 보상) 을 만드는 데 쓰인다.
RA 는 청구 레그의 현가를 **risk class 별로** 그 class 의 cv 로 가중한다:

```
z  = norm_ppf(ra_confidence)              # 신뢰수준 -> 정규분위수
RA = z * ( mortality_cv  * PV(mortality_cf)     # 사망위험
         + morbidity_cv  * PV(morbidity_cf)     # 질병위험 (진단 포함)
         + longevity_cv  * PV(annuity + maturity)
         + disability_cv * PV(disability_cf) )
```

그래서 청구 레그가 risk class 로 쪼개져 있는 것 (`mortality_cf` vs
`morbidity_cf`) 이다 — RA 가 각 class 에 자기 cv 를 곱하려면 PV 를 class 별로
따로 들고 있어야 하기 때문. (cost-of-capital 방식 RA 는 cv 대신 자본비용율을
쓰지만 risk-class 분리는 동일하다.)

## Value conventions

| 컬럼 | 규칙 | 예 | 이유 |
|---|---|---|---|
| `product` | SCREAMING_SNAKE_CASE | `TERM_LIFE_A`, `WHOLE_LIFE_A`, `HEALTH_A` | enum-like 외부 식별자 |
| `channel` | ALL UPPERCASE 약어 | `GA`, `FC`, `TM` | 업계 관용 약어 (General Agency, Financial Consultant, Telemarketing) |
| `table_id` | SCREAMING_SNAKE_CASE 풀네임 | `MORTALITY_STD`, `LAPSE_TERM_GA`, `DISCOUNT_STD`, `INPATIENT_STD`, `ADB_STD` | named reference. 줄임말 안 씀 (`MORT_STD` 같은 abbreviation 지양). 단 industry-universal abbr 인 `ADB` 같은 매우 짧은 것은 예외 |
| `coverage` | SCREAMING_SNAKE_CASE 풀네임 | `DEATH`, `INPATIENT`, `CANCER`, `MATURITY`, `ANNUITY`, `ADB` | enum-like 식별자. 사용자 카탈로그 — 엔진 reserved 코드 없음 |
| `calculation_method` | SCREAMING_SNAKE_CASE 풀네임 | `DEATH`, `MORBIDITY`, `DIAGNOSIS`, `ANNUITY`, `MATURITY` | **engine 의 cash flow 산출방법 routing key**. 5 개 고정. 자세한 각 산출방법별 계산은 `basis-format.md` 의 coverages 시트 섹션 참조 |
| `state` | SCREAMING_SNAKE_CASE | `ACTIVE`, `WAIVER`, `PAIDUP` | enum-like, 정책 status |
| `_DEFAULTS` (특수 product 값) | 대문자 + 밑줄 (예약 마커) | `_DEFAULTS` | segments 시트의 fallback 행 marker (값 아닌 keyword) |

라우팅 / 그룹 축 (`product`, `channel`, `coverage`) 은 **접미사 없는 bare 키**
입니다. 엔진은 이 값을 **불투명 키 (opaque key=내용을 해석하지 않는 식별자)**
로만 다루므로, 코드를 넣든 이름을 넣든 임의 분석그룹을 넣든 사용자 자유입니다
(그래서 `_code` 접미사를 붙이지 않습니다). 사람이 읽기 좋은 `product_name`
같은 라벨 컬럼을 자기 워크북에 둘 수는 있으나 표시용일 뿐 — 엔진은 무시합니다
(샘플 데이터에는 넣지 않습니다).

## 데이터 ID와 Python 코드 enum 의 일관성

워크북 값 (예: `coverage = "DEATH"`, `calculation_method = "MORBIDITY"`) 이
Python 상수 / enum (예: `CalculationMethod.MORBIDITY == "MORBIDITY"`)
과 **bit-exact** 일치합니다. 모두 SCREAMING_SNAKE_CASE 풀네임.
`coverage` 는 사용자 카탈로그 — 엔진이 reserved 코드를 가지고
있지 않으므로 회사가 자유롭게 짓습니다 (`DEATH` 은 샘플 관용).

`enum-like 식별자 family`:

- `product`, `channel`, `table_id`, `coverage`, `calculation_method`,
  `state`, `state_model` — 모두 외부 식별자 / 코드 상수 family. SCREAMING_SNAKE_CASE.
- 줄임말 안 씀 (`MORT` 가 아닌 `MORTALITY`, `HOSP` 가 아닌 `INPATIENT` 등).
  단 industry-universal 한 매우 짧은 abbr 인 `ADB` 정도 예외.

`벤더 데이터 / 컬럼명` (소문자 snake_case):

- 컬럼 헤더 (`product`, `coverage`, `rate_table`, `premium`,
  `count`, `mp_id` 등) — 표/스키마 식별자, 행 안의 값들과 시각적 구분 위해
  소문자.

## Sample workbook의 식별자는 generic placeholder

번들 sample의 `MORTALITY_STD`, `DISCOUNT_STD`, `LAPSE_TERM_GA`, `INPATIENT_STD` 같은
ID는 **generic placeholder** 입니다. 실제 한국 산업 표준 (예: 보험개발원 KIDI
경험생명표 9회) 의 식별자가 아닙니다.

실 사용 시에는 회사가 채택한 위험률 / 발생률 가정의 정확한 식별자로
교체하세요 (예: 회사 경험분석 결과 기반의 `MORTALITY_2024_M_STD`,
감독원 발표 RFR 곡선 기반의 `RFR_2025_12_KOR` 등).

## 보험계리 용어와의 관계

| 워크북 표현 | 보험계리 의미 | 주의 |
|---|---|---|
| `basis.xlsx` 파일 전체 | 산출기초 (valuation basis) | "basis"는 좁은 시트명으로 쓰지 않음 |
| `*_tables` 시트들 (mortality, lapse 등) | **best-estimate 발생률 가정** (IFRS 17 문단 33, B37) | pricing 위험률 (보험료 산출기초의 보수적 측면) 그대로 입력하면 BEL 과대 / CSM 부풀림 |
| `segments` 시트 | 가정 매핑 / 상품·채널 배정 | 발생률 자체 아님 — 어떤 테이블을 쓸지의 indirection |

IFRS 17 GMM의 BEL은 편의 없는 최선추정으로 측정해야 합니다 (문단 33).
워크북에 들어가는 mortality/morbidity/lapse 등은 회사가 외부에서
경험분석·A/E 보정을 거쳐 만든 **best-estimate 발생률 가정**입니다.
pricing 마진은 `premium`에 이미 녹아 있어, BEL의 입력이 best
estimate일 때 `premium_cf > E(mortality_cf)`가 자연스럽게 발생하고, 이 차이가
CSM의 원천이 됩니다.

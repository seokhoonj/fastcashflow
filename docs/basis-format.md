---
orphan: true
---

# 산출기초 (basis) 입력 포맷

fastcashflow 엔진에 들어가는 **계리 가정**을 정의하는 입력 포맷입니다.
사용자가 이 스펙에 맞춰 요율·파라미터를 채워 넣습니다. 이 워크북
(`basis.xlsx`) 을 `read_basis` 가 읽어 **산출기초** (`Basis` — 개별
가정들의 묶음) 개체로 조립합니다.

명명 규칙 전반은 `docs/naming-conventions.md` 참조.

---

## 1. 개요 — 단일 워크북

가정 입력 전체는 **하나의 워크북** `basis.xlsx` 안의 여러 시트에
담깁니다 (아래 표가 전체 목록):

| 시트 | 역할 |
|---|---|
| `segments` | (상품 × 채널) 세그먼트별 어느 표를 쓸지 + 스칼라 파라미터 + `expense_table` 참조 |
| `coverages` | 담보코드 → 위험률표 전역 레지스트리 (coverage, rate_table). calculation_method 는 별도 `calculation_methods` 파일 |
| `mortality_tables` | 사망 발생률 가정 (named tables) |
| `incidence_rate_tables` | 특약 발생률 가정 |
| `waiver_tables` | 납입면제 발생률 가정 |
| `lapse_tables` | 해지율 가정 |
| `discount_tables` | 할인율 곡선 (locked-in, Sec. 36) |
| `expense_tables` | item-form 사업비 ledger (basis dispatch — §3.3) |
| `inflation_tables` | 사업비 인플레이션 곡선 (`table_id` × `year` → `rate`) |
| `surrender_value_tables` (optional) | 해약환급금 곡선 |
| `ae_factors` (optional) | A/E factor — base rate에 런타임 곱셈 (생략 시 1.0) |
| `improvement_tables` (optional) | mortality improvement 곡선 (`table_id` × `year` → `factor`) |

가정은 **세그먼트** (상품 × 채널) 단위로 다릅니다. 한국 시장에서 해지율은
채널(GA / FC 등)별로 크게 다르고, 사망률·사업비도 상품별로 갈립니다.
`segments` 시트가 세그먼트별로 어떤 발생률 표를 쓸지 매핑하고, 엔진은
세그먼트별로 평가한 뒤 IFRS 17 그룹으로 합산합니다.

reader 호출:

```python
import fastcashflow as fcf

fcf.samples.export("samples", template="gmm", quiet=True)   # 견본 한 세트 (본인 파일 있으면 생략)
basis = fcf.read_basis("samples/basis.xlsx")  # BasisRouter: {(product, channel): Basis}
basis = basis.resolve(("TERM_LIFE_A", "GA"))  # 한 세그먼트
```

---

## 2. 공통 규약

- **모든 요율은 연(annual)** 단위. 엔진이 월 요율로 변환합니다
  (constant-force 가정 — 12회 적용 시 연 요율이 정확히 복원).
- `sex`: `0` = 남, `1` = 여.
- 모든 시트의 **1행은 헤더**, 2행부터 데이터입니다. 컬럼은 헤더 **이름**으로
  읽습니다 (순서 무관).
- 표는 **조회 범위 밖이면 끝값을 유지**합니다. 따라서 균일한 가정은 **한 줄만**
  넣으면 전 구간에 적용됩니다 (flat = 1행).
- 두 가지 시간축을 구분합니다:
  - **경과연수(`duration`)** — 계약 발행 후 경과 연수, `0`부터. lapse·
    maintenance에 사용. (mortality는 도달연령 = 가입연령 + 경과연수.)
  - **투영연도(`year`)** — 평가일로부터의 연수, `0`부터. discount·inflation에
    사용.

### Column semantics

| 컬럼명 | 의미 | 단위 | 사용 시트 |
|---|---|---|---|
| `rate` | 확률 / 발생률 / 환산률 | 무차원 (0~1) | mortality, incidence_rate, waiver, lapse, discount |
| `factor` | 무차원 비율 / 환급률 | 무차원 | surrender_value (cum_premium_factor), ae_factors, improvement |
| `amount` | 통화 금액 | 통화 | surrender_value (amount_per_policy / amount_per_unit), expense |

---

## 3. Rate-table 시트들

각 시트는 여러 표를 `table_id`로 구분해 담습니다. 한 시트 안에서 `table_id`가
다르면 다른 표입니다.

### 3.1 Axis-flex 공통 규약 (rate 시트 4종)

`mortality_tables`, `incidence_rate_tables`, `waiver_tables`, `lapse_tables` 네
시트는 **schema-detecting axis-flex** 입니다. 다음 축 중 **사용자가 채운 것만**
컬럼으로 두면 reader가 자동 인식, 빠진 축은 broadcast (모든 값에 동일 적용)
합니다. 모두 `table_id` + `rate` (확률) 컬럼은 필수.

| 축 (선택) | 의미 |
|---|---|
| `sex` | 0 남 / 1 여 |
| `age` | **도달연령** (issue_age + duration); select 축들과 배타 |
| `issue_age` | **가입연령**; `duration`과 함께 select-and-ultimate 표현 |
| `duration` | 경과연수 (0부터); lapse·select 등에 사용 |

지원 schema 조합 예 (`mortality_tables` 기준):

| columns | 의미 | 전형 용도 |
|---|---|---|
| `table_id, rate` | flat 스칼라 | 모든 인구에 같은 발생률 (테스트·단순 모델) |
| `table_id, age, rate` | 도달연령만 | sex/duration 변동 없음 |
| `table_id, sex, age, rate` | 성별 × 도달연령 (현 기본 형식) | 일반적 KIDI 류 |
| `table_id, duration, rate` | 경과연수만 | lapse 기본 |
| `table_id, sex, issue_age, duration, rate` | full select grid | select-and-ultimate 사망률, 채널·duration 결합 해지율 등 |

규칙:
- `age` (도달연령) 과 `issue_age` / `duration` (select 축) 은 **동시 사용 금지** — 둘 다 도달연령 / select 효과 표현이라 의미 중복. reader가 reject.
- 한 시트 안 각 `table_id`의 데이터는 자기 축 카르테시안 곱을 **빠짐없이** 채워야 함 (구멍이 있으면 reject).
- 조회 범위 밖이면 끝값을 유지 (clip-to-edge).
- 모든 4개 시트가 같은 규약. lapse가 sex/age에 의존하는 표를 쓰고 싶으면 그 컬럼을 lapse 시트에 추가하면 됨.

### 3.2 시트별 의미

- `mortality_tables` — 주계약 사망 발생률 (in-force 감소도 함께 구동).
- `waiver_tables` — 납입면제 개시율 (ACTIVE → 면제 상태 전이율).
- `incidence_rate_tables` — 요율 기반 특약 (사망형·질병형·진단형) 발생률.
- `lapse_tables` — 해지율. 보통 상품 × 채널별로 다릅니다 (`LAPSE_TERM_GA`, `LAPSE_HEALTH_FC` 처럼 상품×채널).

### 3.3 사업비 — `expense_tables` 시트 (권장) 와 segments 스칼라 (legacy)

사업비는 두 가지 형태로 받을 수 있습니다. 신규 / 권장 경로는
`expense_tables` 시트의 **row 형식** (basis dispatch), 기존 / 호환 경로는
`segments` 시트의 `alpha_*` / `beta_pct` / `gamma_flat` 스칼라 컬럼.
segments 의 `expense_table` 컬럼이 채워지면 row 형식이 우선이며,
스칼라는 reader 가 0 으로 만들어 두 경로가 동시에 살아있지 않습니다.

#### `expense_tables` (권장)

각 row 는 하나의 사업비 항목. 같은 `table_id` 의 모든 row 가 한 segment 의
사업비 ledger 를 구성합니다.

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `EXPENSE_TERM_FC`) |
| `expense_type` | 자유 라벨 (`acquisition` / `maintenance` / `collection` / `LAE` / `overhead` 등). engine 은 무시; 리포트 / gmm.trace 가 echo. |
| `basis` | 엔진 dispatch 키. 아래 5 종 중 하나 |
| `value` | 값 (`*_pro_rata` 는 비율 0..1, `*_fixed` 는 계약당 정액) |

`basis` vocabulary -- 한국 actuarial α / β / γ 분류 + LAE
(Loss Adjustment Expense, 손해사정비):

| basis | 의미 | 적용 시점 |
|---|---|---|
| `alpha_pro_rata` | α 신계약비 -- 보험료 비례 (`annualized_premium × value`) | t = 0 |
| `alpha_fixed` | α' 신계약비 -- 계약당 정액 (`inforce[0] × value`) | t = 0 |
| `beta_pro_rata` | β 유지/수금비 -- 보험료 비례 (`premium × value`) | 매월 (납입 중) |
| `gamma_fixed` | γ 유지비 -- 계약당 정액 (`inforce × value / 12`) | 매월 |
| `lae_pro_rata` | LAE 손해사정비 -- 청구 비례 (`(claim + morbidity) × value`) | 매월 |

글로벌 `expense_inflation` (segments 의 `inflation_table` 참조) 은
`gamma_fixed` 와 `lae_pro_rata` 에만 적용 -- 두 alpha basis 는 t=0
일회성, `beta_pro_rata` 는 이미 보험료 자체가 시간 변동이라 이중
적용 회피.

segments 시트의 `expense_table` (선택) 컬럼이 segment 별 어느 table_id 를
쓸지 결정. 빈 셀이면 expense 없음 (no-expense basis).

### 3.4 `surrender_value_tables` (optional)

해약환급금 (surrender value) 곡선. 시트 자체가 없거나 segments 의
`surrender_value_table` 컬럼이 비어 있으면 lapse 가 cash flow 없이
in-force 만 제거 (해약환급금 = 0 가정, 무해약환급금형 protection).

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `SURRENDER_STD`) |
| `duration_month` | 가입 후 경과 월 (0 부터, contiguous) |
| `factor` **또는** `amount` | 해약 시 값 — 둘 중 **한 컬럼만**. 의미는 아래 |

해약환급금의 **값 컬럼은 두 가지** 중 하나이고, 어느 의미로 읽을지는 segments
시트의 `surrender_value_basis` 컬럼이 정합니다 (생략 시 `cum_premium_factor`):

| `surrender_value_basis` | 값 컬럼 | 엔진 해석 |
|---|---|---|
| `cum_premium_factor` (기본) | `factor` | 누적 납입보험료에 곱하는 환급률 |
| `amount_per_policy` | `amount` | 계약당 해약환급금 (통화 금액) |
| `amount_per_unit` | `amount` | 단위당 금액 × 정책의 `surrender_base_amount` (가입금액 / 기본보험료 등) |

값 컬럼 종류와 `surrender_value_basis` 가 어긋나면 (`factor` 컬럼을 amount 로,
또는 그 반대로) reader 가 거부합니다 — 한쪽을 다른 쪽으로 잘못 읽으면 조용히
오측정되기 때문. 값 컬럼은 **시트 단위로 하나** 입니다 (`factor` 또는 `amount`):
한 워크북의 surrender 시트 안에 factor 표와 amount 표를 섞을 수 없으므로,
모든 segment 의 `surrender_value_basis` 가 같은 컬럼 종류를 가리켜야 합니다.

엔진 계산 (post-projection):
```
lapse_flow[mp, t]   = inforce[mp, t] x lapse_monthly[mp, year(t)]   # 해약 건수

# cum_premium_factor  (factor 컬럼)
cum_premium[mp, t]  = cumsum(premium_cf[mp, :t+1])
surrender_cf[mp, t] = lapse_flow x cum_premium x factor[duration_month]

# amount_per_policy   (amount 컬럼)
surrender_cf[mp, t] = lapse_flow x amount[duration_month]

# amount_per_unit     (amount 컬럼 + policies 의 surrender_base_amount)
surrender_cf[mp, t] = lapse_flow x amount[duration_month] x surrender_base_amount[mp]
```

BEL에 future outflow 로 포함. 환급률이 substantial 한 한국 상품 (단기납
종신, 경영인 정기, 저해지환급금형 등) 의 BEL 정확도 ↑. 계약 명세서의 해약환급금
표를 그대로 쓰려면 `amount` 컬럼 (amount_per_policy), 가입금액 비례면
`amount_per_unit`, 환급률만 있으면 `factor` (cum_premium_factor).

**제약 (v1)**:
- `cum_premium_factor` 의 기준금액 = **누적 납입보험료** 만 (책임준비금 / 적립금
  기준은 amount 모드로 직접 금액을 주거나 cum premium 기준으로 환산). `amount`
  모드는 평가일 시점 in-force 에 선형이라 보유계약 rescale 이 정확하지만,
  `cum_premium_factor` 는 평가일 이전 납입에 path-dependent 라 rescale 이 근사.
- `lapse_flow` 는 state-machine 의 실제 해약 exit (occupancy x state 별 해약률).
  단일 active state 에서는 `inforce x lapse` 의 historical 식과 같음.

### 3.5 `discount_tables` · `inflation_tables`

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `DISCOUNT_STD`, `INFLATION_STD`) |
| `year` | 투영연도 (평가일로부터, 0부터) |
| `rate` | `discount`: 연 할인율 · `inflation`: 연 비용 인플레이션 |

- 경제 가정 — 시장 기준, **평가일**별로 다르며 상품·채널과 무관.
- `discount_tables` — 할인율 기간구조(커브). 평면 할인율이면 한 줄.
- reader 는 `discount_tables` / `inflation_tables` 의 **연도별 곡선 전체** 를
  그대로 통과시킵니다 (`year` 한 행만 있으면 그 값이 전 구간 평탄 적용).

---

## 4. `segments` 시트

`_DEFAULTS` 행 하나 + 세그먼트(상품 × 채널)별 행들. **빈 칸은 `_DEFAULTS` 행의
값을 상속**하고, 채우면 그 행에서 override합니다.

**키 컬럼**: `product`, `channel`.

**표 참조 컬럼** (값 = rate-table 시트의 `table_id`):
`mortality_table`, `lapse_table`, `waiver_table`, `discount_table`,
`inflation_table`, `surrender_value_table` (선택), `expense_table`.
- `waiver_table` — 납입면제 모델이 없는 상품이면 비웁니다.
- `surrender_value_table` — 해약환급금이 없는 무해약형 protection 이면 비웁니다.
- `expense_table` — `expense_tables` 시트의 `table_id` 참조 (3.3 §).
  채워지면 item-form 사업비 ledger 가 적용되고 그 안의 row 별
  `expense_type` / `basis` / `value` 가 kernel-side primitive 로 들어감
  (alpha_fixed / alpha_pro_rata / beta_pro_rata / gamma_fixed / lae_pro_rata).
  비어 있으면 사업비 zero.

**스칼라 컬럼** (값 = 숫자/문자):
`ra_confidence`, `mortality_cv`, `morbidity_cv`,
`longevity_cv`, `disability_cv`, `expense_cv`, `cost_of_capital_rate`,
`ra_method`, `investment_return`, `fund_fee`.

**Optional age-shift 컬럼** (정수, 기본 0):
`mortality_age_shift`, `morbidity_age_shift`, `waiver_age_shift` —
해당 base table의 `issue_age` 인자를 정수만큼 이동시켜 같은 표를 다른
cohort에 재사용하게 함. 양수면 더 늙게, 음수면 더 젊게 lookup. 모든
coverage rate는 `morbidity_age_shift` 한 값을 공유함. 컬럼이 없거나 0이면
no-op.

**Optional `state_model` 컬럼** (문자열, 기본 None):
세그먼트가 어떤 상태기계 (StateModel) 를 쓰는지 enum 으로 선택.
`fastcashflow.STATE_MODELS` 사전의 키 (문자열) 를 값으로 넣음.

| 값 | 의미 |
|---|---|
| (빈 셀) | `Basis.state_model = None`. 다중 상태 mechanic 이 트리거되면 (납입면제 / 모델포인트의 state 컬럼 비-0) 자동으로 `WAIVER_MODEL` 사용 |
| `WAIVER` | 2-state Markov (ACTIVE / WAIVER). 가장 흔한 한국 protection 상품 형태 |
| `WAIVER_PAIDUP` | 3-state (ACTIVE / WAIVER / PAIDUP). 완납(paid-up) 상태가 있는 보유계약 평가 ([3.2](cookbook/markov/paid-up)) |

향후 라이브러리 버전업에서 추가 등록 가능 (`CANCER_REINCIDENCE`,
`DISABILITY`, `LTC_GRADES` 등). 등록되지 않은 키를 적으면 read_basis
가 `ValueError` 와 등록된 키 목록을 함께 반환.

자유 형태 (사전에 없는 topology) 가 필요하면 Excel 대신 Python 으로
`StateModel` 개체를 직접 만들어 `Basis(state_model=...)` 에 주입.

샘플 워크북의 `_DEFAULTS` 행은 `state_model = WAIVER` 로 설정되어 있어
모든 세그먼트가 명시적으로 WAIVER_MODEL 을 사용.

예시 (`sample_basis.xlsx` 발췌):

```
product      channel  mortality_table  lapse_table    ...  expense_table    ra_confidence  mortality_cv
_DEFAULTS             MORTALITY_STD                   ...                   0.75           0.10
TERM_LIFE_A  GA                        LAPSE_TERM_GA  ...  EXPENSE_TERM_GA
TERM_LIFE_A  FC                        LAPSE_TERM_FC  ...  EXPENSE_TERM_FC
```

- `TERM_LIFE_A / GA` — `mortality_table` 등 빈 칸은 `_DEFAULTS`에서 상속
  (`MORTALITY_STD`, `ra_confidence` 0.75, `mortality_cv` 0.10). `lapse_table`과
  `expense_table`만 행에서 지정.
- `TERM_LIFE_A / FC` — 같은 패턴, lapse 와 expense 만 다름.

전사 고정값 (`ra_confidence` 등) 은 `_DEFAULTS`에 한 번만 적습니다 — 바꿀 때
한 칸만 고치면 됨.

---

## 4.1 Optional `ae_factors` 시트 — A/E (Actual / Expected) 곱셈자

산업 패턴 중 **base rate × A/E factor = best estimate** 형태를 워크북에서
명시적으로 표현하려면 `ae_factors` 시트를 추가합니다. 시트가 없으면 모든
factor가 1.0 (no-op).

필수 컬럼: `product`, `channel`, `coverage`, `factor`.

선택 axis 컬럼 (rate table과 같은 schema 규약): `sex`, `age`, `issue_age`,
`duration` — 채우면 factor가 그 차원으로 변동.

A/E는 `coverages` 시트에 등록된 rate-driven 담보에만 적용됩니다 —
in-force 감쇠용 `mortality_annual` 은 coverage 가 아니므로 A/E 키를
가지지 않습니다 (calibration 은 mortality_table 자체로). lapse / WAIVER 는
v1 에서 A/E factor 미지원.

예시:

```
product   channel  coverage   factor
TERM_LIFE_A    GA       INPATIENT         1.5         # GA의 입원 손해율 150%
TERM_LIFE_A    FC       INPATIENT         1.2         # FC는 좀 더 보수적
TERM_LIFE_A    GA       CANCER            1.0         # 암 발생률은 위험률과 일치
TERM_LIFE_A    GA       DEATH     0.8         # 사망 보장 자체 calibration
```

age 차원 추가 — 20대 anti-selection 패턴:

```
product   channel  coverage   age   factor
TERM_LIFE_A    GA       INPATIENT         25    3.0
TERM_LIFE_A    GA       INPATIENT         26    2.8
...                                          # 각 age마다 한 줄 (dense 요구)
TERM_LIFE_A    GA       INPATIENT         60    1.0
```

엔진 평가 식:
```
final_rate(sex, ia, dur) = base_rate(sex, ia + age_shift, dur)
                          × ae_factor(sex, ia, dur)
                          × (improvement factor — Task #10)
```

A/E sheet 도입의 두 가지 이득:
- **Audit trail** — base table (KIDI 류, 회사 표준) 과 calibration (회사 경험)
  이 분리됨
- **Sensitivity** — factor 한 칸 수정으로 시나리오 재실행 (base 재계산 불필요)

## 4.2 Optional `improvement_tables` 시트 — mortality improvement 곡선

장기 계약에서 사망률은 매년 일정 비율로 개선되는 trend가 있어 그 효과를
미래 사망률에 반영하려면 `improvement_tables` 시트와 `segments`의 optional
`mortality_improvement_table` 컬럼을 사용합니다.

`improvement_tables` 컬럼:

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `IMPR_STD`) |
| `year` | 정수 년수 (0부터, = policy duration year) |
| `factor` | 그 year의 누적 improvement multiplier (보통 1.0 →감소) |

`segments` 컬럼:

| 컬럼 | 의미 |
|---|---|
| `mortality_improvement_table` (optional) | 위 시트의 `table_id` 참조. 빈칸 = 개선 없음 (factor 1.0) |

예시 — 1.5% 연 mortality improvement:

```
improvement_tables:
  table_id    | year | factor
  IMPR_STD    | 0    | 1.000
  IMPR_STD    | 1    | 0.985
  IMPR_STD    | 2    | 0.970
  IMPR_STD    | 3    | 0.956
  ...

segments:
  product | channel | mortality_table | mortality_improvement_table | ...
  TERM_LIFE_A  | GA      | MORTALITY_STD        | IMPR_STD                    | ...
```

엔진 적용:
```
final_mortality(year=d) = base × improvement_factor[d]
```

장년 만기 (60년) 계약에서 `(1 - improvement)^60 ≈ 60%` 정도 감소 — 유효한
크기. 종신·100세 만기 상품 valuation에 의미 있음.

V1 한계: mortality 한 layer만 적용. MORBIDITY / lapse improvement는 별도
확장 (필요 시 같은 패턴으로 추가).

## 5. `coverages` 시트

**rate-driven 담보 registry** — 각 `coverage` 의 요율표. 모든 상품이
같은 담보 정의를 공유합니다 (HEALTH_A 의 `CANCER` 와 WHOLE_LIFE_A 의
`CANCER` 가 동일 incidence rate 사용).

| 컬럼 | 의미 |
|---|---|
| `coverage` | 담보 코드 (model_point 의 coverage row 와 매칭) |
| `rate_table` | `incidence_rate_tables` 또는 `mortality_tables` 의 `table_id`. 빈칸 불가 (rate 없는 담보는 등록 안 함) |

> **분리된 두 파일**: `coverage` 의 `calculation_method` (계산 routing) 과
> `coverage_name` (사람용 메모) 는 별도 **`calculation_methods.csv`**
> (담보별 산출방법) 에 있습니다 — `basis.xlsx` 는 actuarial basis 만,
> 매핑은 신담보 추가 시에만 손대는 분류. 자세한 건
> `docs/cookbook/basics/calculation-methods.md` 참조.

### calculation_method 의 5 종 (`calculation_methods.csv` 에서 분류)

**중요: `calculation_method` 값이 engine 의 cash flow 산출방법을 결정합니다.**
사용자는 각 담보가 어떤 `calculation_method` 인지 매핑에서 선언만 하고,
engine 이 그 산출방법 (몇 회 지급 / depleting pool / inforce 곱 etc.)
을 hardcoded 로 적용.

| `calculation_method` | engine 산출방법 | 사용 case |
|---|---|---|
| `DEATH` | incidence (coverage 의 `rate_table` lookup) × benefit, **non-decrementing** | 일반사망 / 질병사망 / 재해사망. 사망 종류 별 자체 `rate_table` (보통 mortality table 가리킴) |
| `MORBIDITY` | incidence × benefit, **반복 발생** (in-force 안 감소) | 입원 / 수술 / 통원 등 |
| `DIAGNOSIS` | incidence × benefit, **1회 지급** (depleting "not yet diagnosed" pool) | 암진단 / CI / 진단 일시금 |
| `ANNUITY` | in-force × `annuity_payment` (model_point scalar), `annuity_frequency_months` 주기 | 연금 |
| `MATURITY` | term 도달 시점 in-force × `maturity_benefit` (model_point scalar). lump sum | 만기환급금 |

값은 **반드시 위 5 개 중 하나**. 다른 값은 reader error. 새 산출방법이 필요하면
engine 의 새 `CalculationMethod` 멤버 추가 + projection kernel 의 routing branch
추가 작업 (library maintainer 의 영역).

`ANNUITY` / `MATURITY` 는 rate 없이 survival benefit — `coverages` 시트가
아니라 `calculation_methods.csv` 에만 등록하고, model_points 의 `annuity_payment`
/ `maturity_benefit` 스칼라로 금액을 줍니다.

In-force 감쇠 (decrement) 는 항상 `Basis.mortality_annual` 한 곳에서
모든 상품에 적용됩니다 — 사망 보장이 등록되어 있든 아니든, 사람이 죽으면
계약이 종료되니까. 사망 보장 (DEATH 산출방법 coverage) 의 지급률은 그 담보의
자체 `rate_table` 이 결정합니다.

**product 별 다른 정의가 필요한 경우** (예: HEALTH_A 의 cancer 와 WHOLE_LIFE_A 의
cancer 가 진짜 다른 base rate calibration) → 다른 `coverage` 로 분리
(`CANCER_HEALTH`, `CANCER_WHOLELIFE`). 같은 code 이면 같은 정의.

---

## 6. reader 해소 순서

reader (`read_basis(path)`) 가 워크북을 읽어 세그먼트별 `Basis`를
만드는 순서:

1. 각 rate-table 시트의 모든 표를 `table_id`로 적재.
2. `segments` 시트의 `_DEFAULTS` 행을 식별하고, 각 세그먼트 행에서 빈 칸을
   `_DEFAULTS`로 채움.
3. 표 참조 컬럼의 `table_id`를 실제 표로 해소.
4. `coverages` 시트에서 그 상품의 특약 목록을 붙임.
5. → `{(product, channel): Basis}` 반환. 엔진이 세그먼트별로 평가.

---

## 7. 범위 밖 — 참고

- **상품 구조** (상태기계 — 납입면제·장해 상태 전이, 어떤 특약을 갖는지의
  구조) 는 가정이 아니라 상품 정의입니다. 이 워크북은 **요율과 파라미터**만
  담습니다.
- 변액의 `fund_fee` / `investment_return` 은 계리 가정이 아니라 계약조건이라
  본래 상품 정의 쪽이 맞지만, 현재는 segments의 스칼라 컬럼에 들어갑니다 —
  추후 VFA 파라미터를 상품 정의 쪽으로 이전할 예정. 한편 최저보증이율
  (`minimum_crediting_rate`) 과 GMDB / GMAB 보증액은 이미 **모델포인트** 필드
  (`minimum_crediting_rate` / `minimum_death_benefit` / `minimum_accumulation_benefit`) 입니다.
- 계약별 정보 (가입연령·보험금·보험료·계좌가치 등) 는 가정이 아니라
  **모델포인트** 파일입니다.

---

## 7.5 Economic scenarios (별도 파일)

확률론 평가용 시나리오 (할인율 경로 · 투자수익률 경로 등) 은 워크북이
아니라 **별도 파일**로 받습니다. 큰 시나리오 집합 (수천 path × 수백 month)
은 xlsx 보다 binary 형식이 훨씬 효율적이라 분리.

```python
import fastcashflow as fcf
import numpy as np
import polars as pl

mp    = fcf.samples.model_points()
basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))

# 견본 시나리오 파일 (보통은 회사가 만든 파일; 본인 파일 있으면 이 블록 생략)
# wide-format 2-D table: 한 행 = 한 scenario, 한 열 = 한 projection month
n_time = int(mp.term_months.max())
rng = np.random.default_rng(0)
pl.DataFrame(0.03 + rng.normal(0, 0.01, (256, n_time))).write_parquet("samples/discount_scenarios.parquet")

scenarios = fcf.read_scenarios("samples/discount_scenarios.parquet")  # shape (n_scenarios, n_time)
result = fcf.gmm.stochastic(mp, basis, scenarios)             # gmm.stochastic / vfa.tvog 에 직접 전달
```

지원 형식: `.parquet`, `.csv`, `.xlsx`, `.feather`. 한 열짜리 파일은
flat-rate 시나리오로 해석되어 `(n_scenarios,)` 로 반환.

시나리오 생성 (Hull-White, Vasicek, regime-switching, climate path 등)
은 fastcashflow 범위 밖 — 별도 ESG 도구로 만든 결과를 파일로 받는 구조.

## 8. 입력 layer의 향후 확장 (Task #7~#10)

현재 워크북은 **단순 입력 layer**입니다. 엔진은 더 풍부한 callable signature를
갖고 있어 `(sex, issue_age, duration, issue_class, elapsed)`, 사용자가 원할 때
워크북에 다음 레이어를 선택적으로 추가할 수 있도록 확장 예정:

| Layer | 시트 / 컬럼 | 상태 |
|---|---|---|
| Base rate axis-flex | `mortality_tables` 등에 `issue_age` / `duration` 컬럼 옵션 (select-and-ultimate) | Task #7 완료 |
| A/E factor | 새 `ae_factors` 시트 — base rate에 런타임 곱셈 | Task #8 완료 |
| `age_shift` | `segments`에 정수 컬럼 — 테이블 재사용 시 cohort 보정 | Task #9 완료 |
| Mortality improvement | 새 `improvement_tables` 시트 — 연도별 개선 곡선 | Task #10 완료 |

원칙은 **엔진 = 최대 차원, 입력 = 단순~복잡까지 선택**. 안 채우면 no-op
(곱셈자 1.0, shift 0).

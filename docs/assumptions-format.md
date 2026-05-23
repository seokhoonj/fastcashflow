# 계리 가정 입력 포맷 (assumptions input)

fastcashflow 엔진에 들어가는 **계리 가정**을 정의하는 입력 포맷입니다. 현업
담당자가 이 스펙에 맞춰 요율·파라미터를 채워 넣습니다.

명명 규칙 전반은 `docs/naming-conventions.md`, 각 결정의 근거는
`docs/design-decisions.md` 참조.

---

## 1. 개요 — 단일 워크북

가정 입력 전체는 **하나의 워크북** `assumptions.xlsx` 안의 9개 시트에
담깁니다:

| 시트 | 역할 |
|---|---|
| `segments` | (상품 × 채널) 세그먼트별 어느 표를 쓸지 + 스칼라 파라미터 |
| `riders` | 상품별 특약 부착 (rider_code, type, rate_table) |
| `mortality_tables` | 사망 발생률 가정 (named tables) |
| `rider_rate_tables` | 특약 발생률 가정 |
| `waiver_tables` | 납입면제 발생률 가정 |
| `lapse_tables` | 해지율 가정 |
| `maintenance_tables` | 유지비 (maintenance expense) |
| `discount_tables` | 할인율 곡선 (locked-in, Sec. 36) |
| `inflation_tables` | 유지비 인플레이션 곡선 |
| `ae_factors` (optional) | A/E factor — base rate에 런타임 곱셈 (생략 시 1.0) |
| `improvement_tables` (optional) | mortality improvement 곡선 (`table_id` × `year` → `factor`) |

가정은 **세그먼트** (상품 × 채널) 단위로 다릅니다. 한국 시장에서 해지율은
채널(GA / FC 등)별로 크게 다르고, 사망률·사업비도 상품별로 갈립니다.
`segments` 시트가 세그먼트별로 어떤 발생률 표를 쓸지 매핑하고, 엔진은
세그먼트별로 평가한 뒤 IFRS 17 그룹으로 합산합니다.

reader 호출:

```python
basis = read_assumptions("assumptions.xlsx")
# basis: dict[(product, channel), Assumptions]
asmp = basis[("term_a", "GA")]
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
| `rate` | 확률 / 발생률 / 환산률 | 무차원 (0~1) | mortality, rider_rate, waiver, lapse, discount, inflation |
| `amount` | 화폐 금액 | 통화 | maintenance |

---

## 3. Rate-table 시트들

각 시트는 여러 표를 `table_id`로 구분해 담습니다. 한 시트 안에서 `table_id`가
다르면 다른 표입니다.

### 3.1 Axis-flex 공통 규약 (rate 시트 4종)

`mortality_tables`, `rider_rate_tables`, `waiver_tables`, `lapse_tables` 네
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
- `waiver_tables` — 납입면제 개시율 (active → 면제 상태 전이율).
- `rider_rate_tables` — 요율 기반 특약 (사망형·질병형·진단형) 발생률.
- `lapse_tables` — 해지율. 보통 상품 × 채널별로 다릅니다 (`LAPSE_GA`, `LAPSE_FC` 처럼 채널 suffix).

### 3.3 `maintenance_tables`

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `MAINT_STD`) |
| `duration` | 경과연수 (0부터 연속) |
| `amount` | 계약당 연 유지비 (실질 단가, 통화) |

- 경과연수별로 다르면 여러 줄, 단일 단가면 한 줄.
- **실질 단가**입니다 — 인플레이션은 `inflation_tables`가 따로 키웁니다.
- 컬럼이 `rate`가 아니라 `amount`인 이유: 값이 확률이 아니라 화폐 금액이기
  때문 (column semantics 참조).

### 3.4 `discount_tables` · `inflation_tables`

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `DISC_STD`, `INFL_STD`) |
| `year` | 투영연도 (평가일로부터, 0부터) |
| `rate` | `discount`: 연 할인율 · `inflation`: 연 비용 인플레이션 |

- 경제 가정 — 시장 기준, **평가일**별로 다르며 상품·채널과 무관.
- `discount_tables` — 할인율 기간구조(커브). 평면 할인율이면 한 줄.
- **v1 한계**: 현재 reader는 `discount_tables[0]` / `inflation_tables[0]`을
  flat 스칼라로 사용합니다 (커브 전체를 쓰는 작업은 Task #1).

---

## 4. `segments` 시트

`defaults` 행 하나 + 세그먼트(상품 × 채널)별 행들. **빈 칸은 `defaults` 행의
값을 상속**하고, 채우면 그 행에서 override합니다.

**키 컬럼**: `product`, `channel`.

**표 참조 컬럼** (값 = rate-table 시트의 `table_id`):
`mortality_table`, `lapse_table`, `maintenance_table`, `waiver_table`,
`discount_table`, `inflation_table`.
- `waiver_table` — 납입면제 모델이 없는 상품이면 비웁니다.

**스칼라 컬럼** (값 = 숫자/문자):
`expense_acquisition`, `ra_confidence`, `mortality_cv`, `morbidity_cv`,
`longevity_cv`, `disability_cv`, `expense_cv`, `cost_of_capital_rate`,
`ra_method`, `investment_return`, `fund_fee`, `guaranteed_credit_rate`.

**Optional age-shift 컬럼** (정수, 기본 0):
`mortality_age_shift`, `morbidity_age_shift`, `waiver_age_shift` —
해당 base table의 `issue_age` 인자를 정수만큼 이동시켜 같은 표를 다른
cohort에 재사용하게 함. 양수면 더 늙게, 음수면 더 젊게 lookup. 모든
rider rate는 `morbidity_age_shift` 한 값을 공유함. 컬럼이 없거나 0이면
no-op.

예시 (`sample_assumptions.xlsx` 발췌):

```
product   channel  mortality_table  lapse_table  ...  expense_acquisition  ra_confidence  mortality_cv
defaults           MORT_STD                      ...                       0.75           0.10
term_a    GA                        LAPSE_GA     ...  150000
term_a    FC                        LAPSE_FC     ...   80000
```

- `term_a / GA` — `mortality_table` 등 빈 칸은 `defaults`에서 상속
  (`MORT_STD`, `ra_confidence` 0.75, `mortality_cv` 0.10). `lapse_table`과
  `expense_acquisition`만 행에서 지정.
- `term_a / FC` — 같은 패턴, lapse와 acquisition만 다름.

전사 고정값 (`ra_confidence` 등) 은 `defaults`에 한 번만 적습니다 — 바꿀 때
한 칸만 고치면 됨.

---

## 4.1 Optional `ae_factors` 시트 — A/E (Actual / Expected) 곱셈자

산업 패턴 중 **base rate × A/E factor = best estimate** 형태를 워크북에서
명시적으로 표현하려면 `ae_factors` 시트를 추가합니다. 시트가 없으면 모든
factor가 1.0 (no-op).

필수 컬럼: `product`, `channel`, `rider_code`, `factor`.

선택 axis 컬럼 (rate table과 같은 schema 규약): `sex`, `age`, `issue_age`,
`duration` — 채우면 factor가 그 차원으로 변동.

main mortality는 `rider_code = "dth_main"` 으로 표기 (`riders` 시트의
death_main type 코드와 일치). lapse · waiver 는 v1에서 A/E factor 미지원.

예시:

```
product   channel  rider_code   factor
term_a    GA       hosp         1.5         # GA의 입원 손해율 150%
term_a    FC       hosp         1.2         # FC는 좀 더 보수적
term_a    GA       cancer       1.0         # 암 발생률은 위험률과 일치
term_a    GA       dth_main     0.8         # CI 80%
```

age 차원 추가 — 20대 anti-selection 패턴:

```
product   channel  rider_code   age   factor
term_a    GA       hosp         25    3.0
term_a    GA       hosp         26    2.8
...                                          # 각 age마다 한 줄 (dense 요구)
term_a    GA       hosp         60    1.0
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
  term_a  | GA      | MORT_STD        | IMPR_STD                    | ...
```

엔진 적용:
```
final_mortality(year=d) = base × improvement_factor[d]
```

장년 만기 (60년) 계약에서 `(1 - improvement)^60 ≈ 60%` 정도 감소 — 유효한
크기. 종신·100세 만기 상품 valuation에 의미 있음.

V1 한계: mortality 한 layer만 적용. morbidity / lapse improvement는 별도
확장 (필요 시 같은 패턴으로 추가).

## 5. `riders` 시트

각 상품이 어떤 특약을 갖는지, 특약별 유형과 요율표를 정의합니다.

| 컬럼 | 의미 |
|---|---|
| `product` | 상품 |
| `rider_code` | 특약 코드 (모델포인트가 담보를 이 코드로 지칭) |
| `rider_name` | 특약명 (사람용 메모 — 엔진은 안 읽음) |
| `type` | `death_main` / `death` / `morbidity` / `diagnosis` / `annuity` / `maturity` |
| `rate_table` | 요율 기반(`death`·`morbidity`·`diagnosis`)이면 `rider_rate_tables`의 `table_id`; 나머지는 빈칸 |

- `death_main` — 주계약 사망. 요율은 segments의 `mortality_table`이 구동하므로
  `rate_table`은 빈칸.
- `annuity` / `maturity` — 생존급부. 요율 없음 (`maturity_benefit`,
  `annuity_payment` 모델포인트 스칼라로 구동).

---

## 6. reader 해소 순서

reader (`read_assumptions(path)`) 가 워크북을 읽어 세그먼트별 `Assumptions`를
만드는 순서:

1. 각 rate-table 시트의 모든 표를 `table_id`로 적재.
2. `segments` 시트의 `defaults` 행을 식별하고, 각 세그먼트 행에서 빈 칸을
   `defaults`로 채움.
3. 표 참조 컬럼의 `table_id`를 실제 표로 해소.
4. `riders` 시트에서 그 상품의 특약 목록을 붙임.
5. → `{(product, channel): Assumptions}` 반환. 엔진이 세그먼트별로 평가.

---

## 7. 범위 밖 — 참고

- **상품 구조** (상태기계 — 납입면제·장해 상태 전이, 어떤 특약을 갖는지의
  구조) 는 가정이 아니라 *상품 정의*입니다. 이 워크북은 **요율과 파라미터**만
  담습니다.
- 변액 계약조건 (`fund_fee`, 최저보증이율) 은 계리 가정이 아니라 계약조건이라
  본래 모델포인트 / 상품 정의 쪽이 맞지만, 현재는 segments의 스칼라 컬럼에
  들어갑니다 — 추후 [[vfa-param-relocation]] 작업으로 이전 예정.
- 계약별 정보 (가입연령·보험금·보험료·계좌가치 등) 는 가정이 아니라
  **모델포인트** 파일입니다.

---

## 7.5 Economic scenarios (별도 파일)

확률론 평가용 시나리오 (할인율 경로 · 투자수익률 경로 등) 은 워크북이
아니라 **별도 파일**로 받습니다. 큰 시나리오 집합 (수천 path × 수백 month)
은 xlsx 보다 binary 형식이 훨씬 효율적이라 분리.

```python
import fastcashflow as fcf

# wide-format 2-D table: 한 행 = 한 scenario, 한 열 = 한 projection month
scenarios = fcf.read_scenarios("discount_scenarios.parquet")
# shape (n_scenarios, n_time) 의 numpy array

# value_stochastic / measure_tvog 에 직접 전달
result = fcf.value_stochastic(model_points, assumptions, scenarios)
```

지원 형식: `.parquet`, `.csv`, `.xlsx`, `.feather`. 한 열짜리 파일은
flat-rate 시나리오로 해석되어 `(n_scenarios,)` 로 반환.

시나리오 생성 (Hull-White, Vasicek, regime-switching, climate path 등)
은 fastcashflow 범위 밖 — 별도 ESG 도구로 만든 결과를 파일로 받는 구조.

## 8. 입력 layer의 향후 확장 (Task #7~#10)

현재 워크북은 **단순 입력 layer**입니다. 엔진은 더 풍부한 callable signature를
갖고 있어 (sex, issue_age, duration, calendar_year), 사용자가 원할 때
워크북에 다음 레이어를 선택적으로 추가할 수 있도록 확장 예정:

| Layer | 시트 / 컬럼 | 상태 |
|---|---|---|
| Base rate axis-flex | `mortality_tables` 등에 `issue_age` / `duration` 컬럼 옵션 (select-and-ultimate) | Task #7 완료 |
| A/E factor | 새 `ae_factors` 시트 — base rate에 런타임 곱셈 | Task #8 완료 |
| `age_shift` | `segments`에 정수 컬럼 — 테이블 재사용 시 cohort 보정 | Task #9 완료 |
| Mortality improvement | 새 `improvement_tables` 시트 — 연도별 개선 곡선 | Task #10 완료 |

원칙은 **엔진 = 최대 차원, 입력 = 단순~복잡까지 선택**. 안 채우면 no-op
(곱셈자 1.0, shift 0). 자세한 설계 근거는 `docs/design-decisions.md` §8 참조.

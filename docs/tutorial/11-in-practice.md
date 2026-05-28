# 11장. 실무에서의 활용 (1)

```{admonition} 이 장에서 배우는 것
:class: tip

- 가정 파일과 모델포인트 파일의 구조
- 파일을 읽어 평가하고 결과를 저장하기
- 메모리를 넘는 대규모 포트폴리오
```

8장에서 엔진으로 일반모형을 측정하고, 9~10장에서 PAA와 VFA까지
봤습니다. 모두 입력을 코드로 만들거나 샘플 데이터로 불러와 돌린
것이었죠. 실제 업무에서는 회사의 계약과 가정을 다뤄야 하고,
그것은 보통 엑셀·CSV 파일로 들어옵니다. 이 장은 그 파일들의 구조와,
파일을 읽어 평가하는 흐름을 다룹니다.

## 11.1 입력 파일

실무에서 fastcashflow 가 입력으로 받는 파일은 **네 개**입니다. 각각
*누가 / 언제 / 왜* 만드는지가 다릅니다:

```{list-table}
:header-rows: 1
:widths: 16 34 50

* - 입력 개체
  - 파일
  - 무엇
* - **Assumptions**
  - `assumptions.xlsx`
  - 계리적 가정 (사망률 · 해지율 · 할인율 · 사업비 · 위험조정)
* - **ModelPoints**
  - `policies.csv` 또는 `inforce_2026Q1.csv`
  - 보유 계약 (영구 spec + 결산시 상태)
* - **ModelPoints**
  - `coverages.csv`
  - 각 계약에 붙은 담보 (특약) 목록과 가입금액
* - **ModelPoints**
  - `benefit_patterns.csv`
  - 담보 카탈로그 — 담보 코드 → 청구 패턴
```

이 절은 네 파일을 `assumptions` → `policies` → `coverages` →
`benefit_patterns` 순서로 봅니다. 코드에서 reader 가 도는 순서가
그대로입니다 — `read_assumptions` 가 먼저, 그 다음 `read_model_points`
가 세 파일을 ModelPoints 개체로 묶습니다.

### 가정 파일 — `assumptions.xlsx`

엑셀 워크북 한 권. 사망률 / 해지율 / 할인율 / 사업비 / 위험조정 같은
**계리적 가정** 을 시트별로 정리합니다. 결산일마다 calibration 을 갱신
하는 자리.

가장 중심이 되는 두 시트:

**`segments`** — `(상품, 채널)` 매 조합에 어떤 위험률 / 해지율 / 할인율
테이블을 쓸지 한 줄씩 매핑.

| product_code | channel_code | mortality_table | lapse_table | discount_table |
|---|---|---|---|---|
| TERM_LIFE_A | GA | MORTALITY_STD | LAPSE_GA | DISCOUNT_STD |
| TERM_LIFE_A | FC | MORTALITY_STD | LAPSE_FC | DISCOUNT_STD |
| HEALTH_A | TM | MORTALITY_STD | LAPSE_TM_HEALTH | DISCOUNT_STD |

**`mortality_tables`** — 위에서 가리키는 사망률 테이블의 실제 값.

| table_id | sex | age | rate |
|---|---|---|---|
| MORTALITY_STD | 0 | 40 | 0.00088 |
| MORTALITY_STD | 0 | 41 | 0.00097 |
| MORTALITY_STD | 1 | 40 | 0.00045 |

위와 같은 패턴으로 `lapse_tables`, `discount_tables`, `expense_tables`
시트가 자기 ID 별 곡선을 담습니다. `coverages` 시트 한 장이 각 담보
코드를 어느 *위험률 테이블* 에 잇는지 한 줄씩 적습니다 — 담보 카탈로그가
*코드 → 패턴* 매핑이라면 이쪽은 *코드 → 위험률* 매핑.

전체 시트 구조는 [`assumptions-format`](../assumptions-format) 에 정리되어
있습니다. 8장에서 `lambda` 로 적었던 사망률을 여기서는 엑셀 표에 채워
넣는다고 생각하면 됩니다.

### 계약 파일 — `policies.csv`

한 줄이 한 계약. **가입 시점의 영구 spec** 만 들어갑니다 — 가입 후 안
바뀌는 값들입니다.

| mp_id | product_code | channel_code | issue_age | sex | term_months | premium_term_months | count |
|---|---|---|---|---|---|---|---|
| P001 | TERM_LIFE_A | FC | 35 | 0 | 240 | 240 | 1 |
| P002 | HEALTH_A | GA | 38 | 1 | 240 | 240 | 1 |
| P003 | TERM_LIFE_A | GA | 42 | 0 | 240 | 240 | 1 |

- `mp_id` — 계약 식별자
- `product_code` / `channel_code` — `segments` 시트와 맞물려 어느 가정
  세트를 적용할지
- 나머지 — 가입연령 / 성별 / 보험기간 / 납입기간 / 계약 수

**결산 모드** (보유계약 평가) 는 같은 파일에 *상태 컬럼 네 개* 가 더
들어옵니다. 정책관리 시스템이 매 분기 끝에 떨어뜨리는 "보유계약
마감파일" 형태:

| ... | elapsed_months | count | prior_csm | lock_in_rate |
|---|---|---|---|---|
| ... | 36 | 0.92 | 55000 | 0.03 |
| ... | 48 | 0.88 | 42000 | 0.03 |

- `elapsed_months` — 가입 후 경과 개월수
- `count` — 결산일 기준 잔존 (사망 / 해지 빠진 후)
- `prior_csm` — 직전 분기 종가 CSM (이번 분기로 carry-forward)
- `lock_in_rate` — 가입 시점에 잠긴 할인율

이 결합 파일을 보통 `inforce_2026Q1.csv` 같은 분기명으로 부릅니다.
신계약 평가는 영구 spec 만 있는 `policies.csv`, 결산 평가는 spec + 상태가
합쳐진 `inforce_*.csv` — 같은 reader 가 둘 다 받습니다 (11.2 절).

### 담보 파일 — `coverages.csv`

한 줄이 한 (계약, 담보). 주계약도 특약도 모두 한 줄씩이고 `mp_id` 로
계약 파일과 묶입니다.

| mp_id | coverage_code | amount | premium |
|---|---|---|---|
| P001 | DEATH | 80000000 | 45000 |
| P001 | MATURITY | 10000000 | 18000 |
| P002 | DEATH | 50000000 | 28000 |
| P002 | CANCER | 30000000 | 22000 |
| P002 | INPATIENT | 1000000 | 9000 |

- `coverage_code` — 담보 카탈로그 (`benefit_patterns.csv`) 에 등록된 담보
  코드. 그 매핑을 따라 엔진이 청구 알고리즘을 고름.
- `amount` — 가입금액 (사망보험금 / 진단금 / 입원 일당 등)
- `premium` — 그 담보 몫의 월 보험료 (선택, 없으면 0)

담보에 *면책기간* / *감액기간* 이 있으면 `waiting` (면책 개월수) /
`reduction_end` / `reduction_factor` 컬럼을 더합니다. 없는 담보는 비워
둡니다.

P001 은 두 줄 (주계약 사망 + 만기환급), P002 는 세 줄. 계약마다 담보
수가 다르니 행 수도 다릅니다 — **long-form** 입니다. 작은 동질
포트폴리오는 한 행에 담보를 모두 펼친 *wide-form* 도 가능합니다 (한
행 한 계약, `<code>_benefit` 컬럼들).

### 담보 카탈로그 — `benefit_patterns.csv`

회사가 다루는 모든 담보 코드를 모아둔 **목록 파일** 입니다. 각 담보가
엔진 안에서 어떤 *청구 패턴* 으로 계산될지 사용자가 직접 매핑합니다.
새로운 담보가 생길 때 한 줄 추가해주면 됩니다.

| coverage_code | coverage_name | benefit_pattern |
|---|---|---|
| DEATH | 일반사망 | DEATH |
| ADB | 재해사망 | DEATH |
| INPATIENT | 입원특약 | MORBIDITY |
| CANCER | 암진단특약 | DIAGNOSIS |
| ANNUITY | 생존연금 | ANNUITY |
| MATURITY | 만기환급 | MATURITY |

`benefit_pattern` 칸의 값이 엔진의 **청구 알고리즘** 을 결정합니다.
다섯 가지가 전부:

```{list-table}
:header-rows: 1
:widths: 22 78

* - 청구 패턴
  - 엔진의 동작
* - `DEATH`
  - 사망형 담보 (일반사망 / 질병사망 / 재해사망 등). 자기 위험률로 지급, 보유계약은 그대로 (in-force 감쇠는 별도 사망률 입력이 담당)
* - `MORBIDITY`
  - 입원 · 수술 등 반복지급 담보. 매번 지급하고 계약은 유지
* - `DIAGNOSIS`
  - 진단 등 1회 지급 담보. 미진단 풀에서 차감
* - `ANNUITY`
  - 매월 지급하는 생존급부
* - `MATURITY`
  - 보험기간 끝에 지급하는 생존급부
```

같은 사망 사건 (일반 / 재해 / 질병) 도 자기 *위험률 테이블* 은 따로
갖되 청구 패턴은 모두 `DEATH` — 자세한 한국 상품 매핑은 쿡북의
[BenefitPattern 결정 가이드](../cookbook/basics/benefit-patterns-catalog) 참조.

### 네 파일을 한 번에 그림으로

```
assumptions.xlsx      ──┐   계리적 가정 (위험률 / 할인 / 사업비)
policies.csv          ──┤   보유 계약 spec
coverages.csv         ──┤   담보 가입금액 (mp_id 로 policies 와 join)
benefit_patterns.csv  ──┘   담보 카탈로그 (코드 → 패턴 매핑)
                        │
                        ▼
              엔진 평가 (BEL / RA / CSM)
```

코드 매핑은 두 갈래로 갈라집니다 — *패턴* (DEATH/MORBIDITY/...) 은
담보 카탈로그에서, *위험률* (실제 숫자) 은 가정 워크북의 `coverages`
시트에서. 한 자리에 모으지 않고 분리한 이유는 두 매핑이 다른 일을
하기 때문 — 카탈로그는 "어느 알고리즘을 쓸지", 가정은 "어떤 숫자
값을 넣을지".

## 11.2 결산 워크플로 — 매 분기 한 파일

실무의 IFRS17 평가는 보통 분기마다 도는 **결산 사이클** 입니다. 정책관리
시스템이 매 분기 끝에 "보유계약 마감파일" 한 장을 떨어뜨리고 — 그 안에
계약의 영구 spec (가입연령 / 보험기간 / 보험금) 과 직전 분기 종가의
상태 (경과월수 / 잔존 / 직전 분기 CSM / lock-in 할인율) 가 함께
들어 있습니다.

fastcashflow 는 그 한 파일을 그대로 받습니다. `read_inforce_policies`
한 번의 호출이 spec 과 state 를 동시에 읽고 평가에 필요한 두 개체
(`ModelPoints`, `InforceState` 클래스의 개체) 를 돌려줍니다.

```python
import fastcashflow as fcf

# (1) 샘플 파일을 현재 폴더에 생성 (한 번만 — 이미 자기 파일이 있으면 생략)
fcf.save_sample_assumptions("assumptions.xlsx")              # .xlsx 만 (multi-sheet 워크북)
fcf.save_sample_inforce_policies("inforce_2026Q1.csv")       # .csv / .xlsx / .parquet / .feather
fcf.save_sample_coverages("coverages.csv")                   # .csv / .xlsx / .parquet / .feather
fcf.save_sample_benefit_patterns("benefit_patterns.csv")     # .csv / .xlsx / .parquet / .feather
# .xlsx 는 시트당 ~ 1M row 한계 -- 대형 portfolio 는 .parquet / .feather 권장

# (2) 결산 평가 — 한 분기의 inforce 한 파일을 그대로 읽어 in-force 측정
basis       = fcf.read_assumptions("assumptions.xlsx")       # {(product_code, channel_code): Assumptions}
assumptions = basis[("TERM_LIFE_A", "GA")]                   # 한 세그먼트 선택

model_points, state = fcf.read_inforce_policies(
    "inforce_2026Q1.csv", assumptions,                       # 결산 1-파일 (spec + state 결합)
    coverages="coverages.csv",                               # 담보 파일 (long-form)
    benefit_patterns="benefit_patterns.csv",                 # 담보 카탈로그
)
val = fcf.value_in_force(
    model_points, assumptions, period_months=3,              # 다음 분기 (3 개월) 까지의 평가
    prior_csm    = state.prior_csm,                          # 직전 분기 종가 CSM
    lock_in_rate = state.lock_in_rate,                       # 가입 시점 lock-in 된 할인율
)
fcf.write_valuation(val, "results_2026Q1.csv")               # 결과 파일
```

각 함수의 역할:

- `save_sample_*` — 패키지 내장 샘플 파일을 디스크에 복사합니다. Excel /
  텍스트 에디터로 열어 fastcashflow 의 입력 파일이 어떻게 생겼는지 직접
  들여다 볼 수 있습니다. 자기 데이터를 쓸 땐 이 줄을 빼고 그 자리에
  자기 파일이 있다고 보면 됩니다.
- `read_assumptions` — 가정 엑셀을 읽어 `{(product_code, channel_code):
  Assumptions}` 딕셔너리로 돌려줍니다. 한 워크북에 여러 세그먼트
  (상품 × 채널) 를 함께 관리하기 위함입니다. 한 세그먼트만 쓰려면
  키로 골라냅니다.
- `read_inforce_policies` — 결산 1-파일을 읽어 **`(ModelPoints, InforceState)`
  튜플** 을 돌려줍니다. ModelPoints 에는 `elapsed_months` / `count` 가
  이미 fold 되어 있고, InforceState 는 `prior_csm` / `lock_in_rate` 을
  carry — 다음 줄의 `value_in_force` 에 그대로 넘깁니다.
- `value_in_force` — 결산 평가. 신계약 평가의 `value` 와 다른 점은:
  (a) 가입 시 lock-in 된 할인율을 명시적으로 받음, (b) 직전 분기의 CSM
  을 출발점으로 carry-forward, (c) `period_months` 로 이번 분기에만
  release 될 부분을 잘라냄.
- `write_valuation` — BEL·RA·CSM·손실요소를 모델포인트마다 한 줄씩
  파일로 저장합니다.

```{admonition} 신계약 평가는 어떻게?
:class: note

위는 *결산* (보유계약) 평가입니다. 새로 인수한 계약은 *결산 상태가
없으니* `inforce_state` 컬럼이 없는 보통의 policies 파일로:

​    `model_points = fcf.read_model_points("new_business.csv", assumptions, coverages=..., benefit_patterns=...)`
​    `val = fcf.value(model_points, assumptions)`

`read_model_points` 와 `value` 의 흐름. 8 장에서 이미 본 형태와 같습니다.
신계약과 보유계약은 같은 엔진이지만 입력 파일 / 함수가 다른 두 *모드*.
```

자기 데이터가 *두 파일* (영구 spec 의 `policies.csv` + 분기별 갱신의
`inforce_state.csv`) 로 분리되어 들어오는 ETL 환경이라면, 그대로 둘로
받는 path 도 있습니다 — `read_model_points("policies.csv", ...)` +
`read_inforce_state("inforce_state.csv")` + `apply_inforce_state(mp,
state)`. 결과는 위 1-파일과 동일.

## 11.3 메모리를 넘는 규모

포트폴리오가 너무 커서 메모리에 한꺼번에 올리기 어렵다면
`value_file()`을 씁니다. wide 형식의 parquet 파일을 조각조각 나눠
읽고, 평가하고, 결과를 쓰는 일을 한 조각씩 차례로 처리해, 메모리에는
한 번에 한 조각만 올립니다.

```python
# 시연용 셋업 -- long-form 샘플을 wide parquet 로 한 번 변환
# (자기 데이터를 쓸 때는 이미 wide parquet 형태로 갖고 있다고 가정)
model_points.to_wide(assumptions).write_parquet("portfolio.parquet")

# 스트리밍 평가 -- 한 줄. 결과는 results/ 폴더에 분할 저장
fcf.value_file(
    "portfolio.parquet", "results/", assumptions,
    benefit_patterns="benefit_patterns.csv",
)
```

이 방식이면 포트폴리오 크기는 메모리가 아니라 디스크가 허락하는
만큼까지 늘어납니다.

## 11.4 다음 장

여기까지 입력 파일을 읽어 평가하고 저장하는 흐름을 봤습니다. 다음 장은
같은 측정 결과를 그림으로 보고, 기간별 변동을 분석하고, 손익 리포트로
정리하는 법을 다룹니다.

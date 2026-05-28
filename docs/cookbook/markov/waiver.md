# 3.1 보험료 납입면제 (waiver)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 보험료 납입면제가 *상태 전이* (active → waiver) 로 모델링되는 이유 —
  cookbook 의 첫 상태(Markov) 챕터
- `state_model` 과 `waiver_incidence_annual` 두 자리의 wiring
- waiver 상태에서 바뀌는 것 — *보험료는 멈추고 보장은 계속*
- 납입면제가 BEL 을 어떻게 키우는지 (보험료 수입이 면제되니 부채 증가)
```

지금까지 (2 장) 의 정액형 상품은 보유계약이 사망 / 해지로 *줄기만* 했습니다.
이 챕터는 계약이 한 상태에서 다른 상태로 *옮겨 가는* 첫 사례입니다.

## 상품 소개 — 납입면제

**보험료 납입면제** (waiver of premium) 는 피보험자가 장해 등 정해진 사유에
해당하면, 그 이후의 보험료 납입을 **면제** 하면서 보장은 그대로 유지하는
특약입니다. 보험료를 더 안 내도 사망보험금 · 진단금 등은 계속 보장됩니다 —
면제된 보험료는 사실상 보험사가 부담하는 셈입니다.

엔진 관점에서 한 계약은 두 상태 중 하나에 있습니다:

- **active** — 정상 납입 중. 매월 보험료를 내고, 사망 / 해지 / *납입면제 진입*
  의 위험에 노출.
- **waiver** — 납입면제 상태. 보험료를 안 내지만 보장은 계속. 사망으로만
  빠져나감 (해지는 보통 없음).

매월 active 계약의 일부가 납입면제 사유로 **waiver 로 전이** 합니다. 이
전이율이 `waiver_incidence_annual` 입니다.

## 모델링 매핑 — 2-state

```{list-table}
:header-rows: 1
:widths: 32 68

* - 자리
  - 무엇
* - `Assumptions.state_model`
  - `STATE_MODELS["WAIVER"]` — active / waiver 2-state 모델
* - `Assumptions.waiver_incidence_annual`
  - active → waiver 연 전이율 callable `(sex, issue_age, duration)`
* - `ModelPoints.state`
  - 각 계약의 시작 상태. 신계약은 `STATE_ACTIVE` (납입 중)
```

핵심은 **두 상태에서 보험료와 보장이 다르게 작동** 한다는 점:

- **보험료** 는 *active 점유* 에만 곱해집니다 — waiver 로 옮겨 간 계약은 더
  안 냅니다.
- **사망보험금** 은 *전체 보유계약* (active + waiver) 에 곱해집니다 — waiver
  계약도 보장은 살아 있으니까.

## 한 계약 — 손계산과 엔진

납입면제의 효과를 또렷이 보려고, **보험료가 사망보험금을 정확히 상쇄** 하도록
설정합니다 (월 보험료 1,000 = 월 사망보험금 기대값 1% × 100,000). 납입면제가
없으면 BEL = 0 이고, 납입면제가 들어오면 그만큼 부채가 생깁니다.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 3개월, active 로 시작
- 월 사망률 1%, 해지 없음, 월 납입면제 발생률 10%
- 사망보험금 100,000, 월 보험료 1,000
- 월 할인율 0 (상태 전이에 집중)
```

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import STATE_MODELS, STATE_ACTIVE

# 사망률 함수 -- 월 1% 의 연 환산 (평탄)
death_fn  = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)
# 해지율 함수 -- 해지 없음
lapse_fn  = lambda s, a, d: np.full(d.shape, 0.0)
# 납입면제 발생률 함수 -- 월 10% 의 연 환산 (active → waiver 전이율)
waiver_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.10) ** 12)

# 계리적 가정
asmp = fcf.Assumptions(
    mortality_annual        = death_fn,   # 보유계약 감쇠용 사망률 (월 1%)
    lapse_annual            = lapse_fn,   # 해지율 (해지 없음)
    waiver_incidence_annual = waiver_fn,  # active → waiver 전이율 (월 10%)
    discount_annual         = 0.0,        # 연 할인율 0 (검증 단순화)
    ra_confidence           = 0.75,       # 위험조정 신뢰수준 75%
    mortality_cv            = 0.10,       # 사망률 변동계수 10%
    state_model             = STATE_MODELS["WAIVER"],  # 2-state: active / waiver
    coverages               = (
        fcf.CoverageRate("DEATH", death_fn),  # 사망 보장 1종 (청구 rate = death_fn)
    ),
)

# 모델 포인트 (계약 하나, active 로 시작)
mp = fcf.ModelPoints.single(
    issue_age     = 40,            # 가입연령 40세
    sex           = 0,             # 성별 (0=남, 1=여)
    benefits      = {0: 100_000},  # 0번 보장 (= DEATH) 의 보험금 100,000
    level_premium = 1_000,         # 월납 보험료 1,000
    term_months   = 3,             # 보험기간 3개월
    state         = STATE_ACTIVE,  # 시작 상태 (active = 납입 중)
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},
)

m = fcf.measure(mp, asmp)
print(f"inforce    = {m.cashflows.inforce[0, :3]}")     # 보유계약 (active + waiver)
print(f"premium_cf = {m.cashflows.premium_cf[0, :3]}")  # 보험료 (active 만 납입)
print(f"claim_cf   = {m.cashflows.claim_cf[0, :3]}")    # 사망보험금 (전체 inforce)
print(f"BEL        = {m.bel[0, 0]:.2f}")                # 최선추정부채
```

출력:

```
inforce    = [1.     0.99   0.9801]
premium_cf = [1000.    891.    793.881]
claim_cf   = [1000.    990.    980.1]
BEL        = 285.22
```

손계산으로 상태 점유를 따라갑니다. active 는 매월 사망 (1%) *과* 납입면제
전이 (10%) 로 둘 다 빠지고, waiver 는 사망 (1%) 으로만 빠집니다:

| t | active | waiver | 전체 inforce | 보험료 (active × 1,000) | 사망보험금 (inforce × 1% × 100,000) |
|---|---|---|---|---|---|
| 0 | 1.000000 | 0.000000 | 1.000000 | 1,000.00 | 1,000.00 |
| 1 | 0.891000 | 0.099000 | 0.990000 |   891.00 |   990.00 |
| 2 | 0.793881 | 0.186219 | 0.980100 |   793.88 |   980.10 |

- `active[t] = (0.99 × 0.90)^t` — 사망 0.99 와 납입면제 0.90 를 *둘 다* 곱함
- `전체 inforce[t] = 0.99^t` — 사망으로만 줄어듦 (waiver 는 in-force 유지)
- BEL = PV(사망보험금) − PV(보험료) = (1,000 + 990 + 980.1) −
  (1,000 + 891 + 793.88) = 2,970.1 − 2,684.88 = **285.22**

엔진의 `premium_cf` / `claim_cf` 가 표의 두 열과 정확히 일치합니다.

```{admonition} 납입면제가 없으면 BEL = 0
:class: note

`waiver_incidence_annual` 을 0 으로 두면 active 점유가 전체 inforce 와
같아져 보험료 (1,000 × inforce) 가 사망보험금 (inforce × 1% × 100,000 =
1,000 × inforce) 을 매월 정확히 상쇄 → **BEL = 0**. 납입면제가 들어오는
순간 보험료 수입만 빠지고 보장은 그대로라, 그 차액이 곧 부채 285.22 로
나타납니다. 이것이 납입면제의 비용입니다.
```

## 결과 읽기 — 보험료와 보장의 비대칭

납입면제 모델의 한 줄 요약: **보험료는 active 에만, 보장은 전체 inforce 에.**

- `premium_cf` 가 `claim_cf` 보다 빠르게 줄어드는 게 핵심 신호입니다. 위
  예제에서 t=2 의 보험료 793.88 vs 사망보험금 980.10 — active 점유가
  전체 inforce 보다 작아서 생기는 격차.
- 이 격차의 현재가치 합이 BEL 을 양수로 (= 손실 쪽으로) 밀어 올립니다.
- 납입면제 발생률이 높을수록 active 가 빨리 빠져 보험료 수입이 줄고 BEL 이
  커집니다.

```{admonition} 상태 점유는 trajectory 에 직접 노출되지 않음
:class: note

`Measurement` 는 active / waiver 의 분리 점유를 따로 내주지 않습니다 —
`inforce` 는 두 상태의 합입니다. 위 손계산처럼 active 점유는
`premium_cf / level_premium` 로 역산할 수 있습니다 (보험료가 active 에만
곱해지므로). 상태별 점유가 필요한 정밀 검증은 [검증 패턴](../workflow/validation)
의 `show_trace` 로.
```

## 변형 — 발생률 축과 워크북 wiring

### 발생률을 연령 / 경과에 의존시키기

`waiver_incidence_annual` 도 다른 rate 함수와 같은 `(sex, issue_age, duration)`
시그니처라, 실무에서는 평탄 상수 대신 경험률표 룩업을 넣습니다. 워크북으로
평가할 때는 `assumptions.xlsx` 의 `waiver_tables` 시트에 표를 채우고
`segments` 시트의 `waiver_table` 컬럼이 그 `table_id` 를 가리키게 합니다
(자세한 시트 구조는 [튜토리얼 11장](../../tutorial/11-in-practice)).

### paid-up 분리

납입후 (paid-up) 상태를 active / waiver 와 별도로 추적하려면 3-state 모델이
필요합니다 — 별도 챕터 (작성 예정). 본 챕터의 `STATE_MODELS["WAIVER"]` 는
active / waiver 2-state 만 다룹니다.

## 함정

### 함정 1 — `state_model` 을 안 거는 경우

`waiver_incidence_annual` 만 넣고 `state_model` 을 생략하면 엔진이
`WAIVER_MODEL` 을 암묵적으로 적용합니다. 동작은 같지만, **명시적으로
`state_model = STATE_MODELS["WAIVER"]` 를 거는 것** 이 의도를 드러내고
다른 state 모델로 바꿀 때 한 자리만 고치면 되는 안전한 패턴입니다.

### 함정 2 — 시작 상태를 안 줌

`ModelPoints` 의 `state` 를 생략하면 `STATE_ACTIVE` (납입 중) 로 시작합니다.
신계약은 그게 맞지만, 결산 시점에 *이미 납입면제 중인* 보유계약을 평가할
때는 `state = STATE_WAIVER` 로 시작 상태를 명시해야 합니다.

### 함정 3 — waiver 에서 보험료가 계속 잡힘

직접 2-state 를 안 쓰고 보험료를 그냥 0 으로 만들어 흉내 내면, *언제부터*
면제되는지의 전이 동학이 사라져 BEL 이 틀립니다. 납입면제는 "보험료를 0
으로" 가 아니라 "active 점유가 매월 waiver 로 빠져나가는" 상태 전이로
모델링해야 정확합니다.

## 인접 레시피

- [2.1 정기보험](../simple/term-life) — 상태 전이 없는 정액형, 본 챕터의
  출발점.
- [1.3 사망률의 두 역할](../basics/mortality-roles) — decrement 와 보장 청구의
  분리. 납입면제도 같은 결 (active 감쇠 vs 전체 보장).
- [2.3 다종 진단 + 면책 / 감액](../simple/diagnosis-rules) — 담보 룰 축.
  상태 (납입면제) 와 직교하며 한 계약에 공존.
- paid-up 분리 (3-state) (작성 예정) — active / waiver / paidup 을 각각 별도
  state 로.
- [검증 패턴](../workflow/validation) — `show_trace` 로 상태별 점유와 cash
  flow 를 한 줄씩 확인.
```

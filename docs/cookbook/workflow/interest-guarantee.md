# 8.4 전통형 금리보증 비용 (TVOG)

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 전통형 / 금리연동형 상품의 **최저보증이율** 이 회사에 지우는 비용을, 결정론
  **intrinsic value** 와 변동성이 만드는 **시간가치 (TVOG)** 로 분해해 측정하는 법
  (`pricing.interest_tvog`, 별칭 `gmm.interest_tvog`)
- 이 측정이 앞서 만든 두 조각 — `statutory_reserve` (통계준비금 `V_t`) 와
  `esg.simulate` (위험중립 금리시나리오) — 을 **엮어서** 나온다는 것
- **시간가치가 곧 금리 변동성의 비용** 이라는 점 (`rate_vol` 을 키우면 intrinsic 은
  그대로, 시간가치만 커짐)
- 왜 고정현금흐름의 확률론 재할인 (`gmm.stochastic`) 으로는 이 비용이 **보이지
  않는지** (비대칭이 없으면 TVOG 가 0)
:::

전통형 / 금리연동형 상품은 적립 준비금에 **최저보증이율 `i_g`** 를 부리합니다.
회사의 운용 earned rate `r` 이 `i_g` 아래로 떨어지면 회사가 그 부족분을 메워야
합니다. `max(i_g - r, 0)` 은 볼록 (convex) 이라, 중앙(결정론) 시나리오 하나로
보면 **intrinsic value** (중앙 경로에서의 비용) 만 보이고, 금리가 흔들릴 때 추가로
드는 **시간가치 (TVOG = Time Value of Options and Guarantees)** 는 여러 시나리오에서만
드러납니다. 둘의 합이 보증의 전체 비용입니다:

    전체 비용  =  intrinsic value  +  시간가치 (TVOG)

이것은 [5.2 변액보험 최저보증의 시간가치](../variable/gmdb-gmab-tvog) 의 **일반계정
판** 입니다 — 그쪽은 분리계정 (계좌가치) 의 크레딧 floor, 이쪽은 일반계정 준비금의
이율 보증.

## 모델링 매핑 — 두 조각을 엮는다

:::{list-table}
:header-rows: 1
:widths: 36 64

* - 조각
  - 출처
* - 준비금 궤적 `V_t`
  - `pricing.statutory_reserve(mp, stat)` — `i_g` 로 부리하는 NLP 통계준비금
    ([8.3 수익성 분석](profit-testing)).
* - 금리시나리오 `r_{s,t}`
  - `esg.simulate(...).rates` — 위험중립 `(n_scen, n_time)` 연율
    ([경제적 가정 — 시나리오 생성](../basics/scenario-generation)).
* - 보증율 `i_g`
  - `statutory_basis.discount_annual` (스칼라) 기본값.
* - 중앙 경로
  - 기본 = `initial_prices` 의 내재 forward 곡선 (무변동성 baseline).
:::

시나리오 `s`, 월 `t` 의 비용 = 부족분의 현재가치
`cost_s = sum_t D_s(t) * max(i_g_m - r_m[s,t], 0) * V_t` 이고, intrinsic / 시간가치로
분해해 `TVOGResult` 로 돌려줍니다.

## 최소 작동 예제 — intrinsic 과 시간가치

2.5% 보증이율의 전통형 양로보험. 곡선이 `i_g` 아래에서 출발하도록 잡아 보증이
실제로 뭅니다.

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import pricing

# a traditional endowment with a 2.5% guaranteed crediting rate
endow = fcf.ModelPoints.single(
    issue_age=40, premium=0.0, term_months=120,
    benefits={"DEATH": 100_000_000.0}, maturity_benefit=100_000_000.0,
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
stat = fcf.Basis(mortality_annual=0.01, lapse_annual=0.0, discount_annual=0.025,
                 ra_confidence=0.75, mortality_cv=0.0,
                 coverages=(fcf.CoverageRate("DEATH", 0.01),))

# risk-neutral interest scenarios from the ESG; the curve starts below i_g
maturities = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0])
rates      = np.array([0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.030])
scen = fcf.esg.simulate(maturities, rates, ufr=0.035, alpha=0.1,
    mean_reversion=0.05, rate_vol=0.015, equity_vol=0.0, correlation=0.0,
    n_scenarios=2000, n_time=120, seed=20240601)

res = pricing.interest_tvog(endow, stat, scen.rates,
                                      initial_prices=scen.initial_prices)
print(f"intrinsic value = {res.intrinsic_value:>13,.0f}")
print(f"time value      = {res.time_value:>13,.0f}")
print(f"total cost      = {res.total_value:>13,.0f}")
```

출력:

```text
intrinsic value =        28,390
time value      =     3,972,095
total cost      =     4,000,485
```

intrinsic 은 작습니다 — 중앙(forward) 경로는 `i_g` 를 살짝만 밑돌기 때문입니다.
비용의 대부분 (3,972,095) 은 **시간가치** — 금리가 흔들리며 어떤 경로에서는 보증이
깊이 무는 데서 오는 비용입니다. 결정론 평가 하나로는 이 부분이 통째로 안 보입니다.

## 시간가치 = 변동성의 비용

`rate_vol` 만 바꿔 가며 재면, intrinsic 은 (중앙 경로가 같으니) **그대로** 이고
**시간가치만 변동성에 따라 커집니다** — 시간가치가 곧 금리 변동성이 매기는
값이라는 뜻입니다.

```python
print(f"{'rate_vol':>9}{'intrinsic':>13}{'time value':>14}")
for vol in (0.005, 0.015, 0.030):
    s = fcf.esg.simulate(maturities, rates, ufr=0.035, alpha=0.1,
        mean_reversion=0.05, rate_vol=vol, equity_vol=0.0, correlation=0.0,
        n_scenarios=2000, n_time=120, seed=20240601)
    r = pricing.interest_tvog(endow, stat, s.rates,
                                        initial_prices=s.initial_prices)
    print(f"{vol:>9.3f}{r.intrinsic_value:>13,.0f}{r.time_value:>14,.0f}")
```

출력:

```text
 rate_vol    intrinsic    time value
    0.005       28,390       887,925
    0.015       28,390     3,972,095
    0.030       28,390     8,786,206
```

## 결과 해석 — 비용의 부호와 netting

`total_value` 는 회사가 지는 **추가 비용** 이라 항상 `>= 0` 입니다 (`max(...) >= 0`).
가치에 반영할 때는 **BEL 에 더하고 CSM / CSM+RA 에서 뺍니다** — 보증이 깊으면 계약을
onerous 로 밀 수 있습니다. `intrinsic_value` 는 결정론 중앙이율 평가가 이미 보는
부분, `time_value` 는 확률론에서만 드러나는 추가분입니다. `guarantee_cost` 는
`(n_scenarios,)` 비용 분포라 백분위수 (꼬리 위험) 도 읽을 수 있습니다.

## 함정 / 검증

- **`gmm.stochastic` 로는 안 보임** — `measure_stochastic` 는 고정현금흐름을 시나리오별
  재할인할 뿐이라 비대칭이 없고, 시나리오 평균이 결정론 BEL 과 같아 TVOG 가 0 입니다.
  보증의 시간가치는 준비금 부리에 걸리는 floor 에서만 나옵니다 — 이 함수가 그
  자리입니다.
- **중앙 경로는 forward 가 기본** — `initial_prices` (= `scen.initial_prices`) 를 주면
  곡선의 내재 forward 를 중앙 경로로 씁니다. 위험중립에서 forward 는 같은 `P(0,t)` 를
  재현하는 무변동성 경로라, intrinsic / 시간가치 분해의 baseline 으로 맞습니다.
  `central_rates` 로 직접 줄 수도 있고, 둘 다 없으면 명시적 에러 (조용한 평균 fallback
  없음).
- **스칼라 `i_g` (v1)** — 보증율은 스칼라입니다. `statutory_basis.discount_annual` 이
  per-year 곡선이면 `guaranteed_rate=` 로 스칼라를 명시하세요.
- **horizon 일치** — `rate_scenarios` 의 열 수는 `n_time` (계약경계 horizon =
  `statutory_reserve(...)[0].shape[1] - 1`) 과 같아야 합니다. ESG 를 같은 `n_time` 로
  생성하세요.
- **몬테카를로 잡음** — 2,000 경로는 seed / 경로수에 따라 흔들립니다. 보고용 수치는
  경로를 늘리고 (antithetic 기본 on), 분포의 백분위수까지 함께 보세요.

## 인접 레시피

- [5.2 변액보험 최저보증의 시간가치](../variable/gmdb-gmab-tvog) — 분리계정 (계좌형)
  의 TVOG. 이 챕터의 일반계정 대응.
- [8.3 수익성 분석 / profit-testing](profit-testing) — `statutory_reserve` 와 NLP
  준비금. 이 보증비용은 그 준비금 위에 얹힙니다.
- [경제적 가정 — 시나리오 생성](../basics/scenario-generation) — `esg.simulate` 로
  위험중립 금리시나리오를 만드는 법.

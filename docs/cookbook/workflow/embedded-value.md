# 8.5 임베디드밸류 / 신계약가치 (VNB)

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 이익 시그니처에서 **신계약가치 (VNB = Value of New Business)** 를 뽑는 법 --
  `VNB = PVFP - CoC - TVOG` (`pricing.embedded_value`)
- `nbv` (= -BEL) 가 **필요자본 차감 전** 가치인 데 비해, VNB 는 **요구자본 보유비용
  (CoC)** 과 **보증의 시간가치 (TVOG)** 까지 차감한 주주가치라는 것
- 앞 세 챕터의 조각이 한 식으로 합성되는 자리 -- 이익 시그니처 (8.3) +
  요구자본 보유비용 + 금리보증 TVOG (8.4)
:::

[8.3 수익성 분석](profit-testing) 의 `nbv` 는 **세전 · 필요자본 차감 전** 가치였습니다.
실제 주주가치는 거기서 두 가지를 더 뺍니다 -- 계약 뒤에 **요구자본을 묶어 두는 비용**
과, **금리보증 같은 옵션의 시간가치**. 그 결과가 **신계약가치 (VNB)** 입니다:

    VNB = PVFP - CoC - TVOG

- **PVFP** -- 미래 주주이익의 현재가치. 이익 시그니처
  (`statutory_profit_signature` 또는 IFRS17 `signature`) 의 현가.
- **CoC** -- 요구자본 보유의 마찰비용. 연 `frictional_spread` 를 요구자본에 매월
  (1/12) 부과해 현가.
- **TVOG** -- 보증의 시간가치 ([8.4](interest-guarantee) 의 금리보증 TVOG 등).

`embedded_value` 는 이 셋을 합성만 합니다 -- 어느 것도 새로 투영하지 않습니다.

## 최소 작동 예제 -- PVFP 에서 VNB 로

전통형 양로보험 (10% 로딩) 의 통계 이익 시그니처에서 출발합니다.

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import pricing
from fastcashflow.curves import discount_monthly_curve

endow = fcf.ModelPoints.single(
    issue_age=40, premium=0.0, term_months=120,
    benefits={"DEATH": 100_000_000.0}, maturity_benefit=100_000_000.0,
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
stat = fcf.Basis(mortality_annual=0.01, lapse_annual=0.0, discount_annual=0.025,
                 ra_confidence=0.75, mortality_cv=0.0,
                 coverages=(fcf.CoverageRate("DEATH", 0.01),))

reserve, net = pricing.statutory_reserve(endow, stat)
gross = fcf.ModelPoints.single(
    issue_age=40, premium=float(net[0]) * 1.10, term_months=120,   # 10% loading
    benefits={"DEATH": 100_000_000.0}, maturity_benefit=100_000_000.0,
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
sig = pricing.statutory_profit_signature(gross, stat, stat)

# no capital, no TVOG -- VNB is just the PVFP
ev0 = pricing.embedded_value(sig, reference_rate=0.05)
print(f"PVFP = {ev0.pvfp:,.0f}   VNB = {ev0.value:,.0f}")
```

출력:

```text
PVFP = 7,087,977   VNB = 7,087,977
```

## 요구자본 보유비용 (CoC) 차감

요구자본을 준비금의 5% 로 잡고 (stylized), 연 6% 의 마찰스프레드를 매깁니다.
요구자본 경로는 명시 배열로 줄 수도 있고 (예: confidence-level `ra_path.sum(0)`
위험자본, 또는 규제자본 경로), 여기처럼 **준비금 x factor** 로 줄 수도 있습니다.

```python
n_time = reserve.shape[1] - 1
dm = discount_monthly_curve(stat, n_time)
V  = reserve.sum(axis=0)                     # (n_time+1,) portfolio reserve

ev1 = pricing.embedded_value(sig, reference_rate=0.05, discount_monthly=dm,
                             required_capital=0.05, reserve=V, frictional_spread=0.06)
print(f"PVFP {ev1.pvfp:,.0f}  CoC {ev1.cost_of_capital:,.0f}  VNB {ev1.value:,.0f}")
```

출력:

```text
PVFP 7,087,977  CoC 1,129,822  VNB 5,958,155
```

요구자본을 묶어 두는 마찰비용 1,129,822 만큼 가치가 깎입니다. `frictional_spread` 는
**연율** 이라 매월 1/12 로 부과됩니다 -- 같은 관례로, confidence-level 위험자본
경로에 `cost_of_capital_rate` 를 스프레드로 넣으면 엔진의 cost-of-capital RA 와
정확히 일치합니다.

## 금리보증 TVOG 까지 -- 완성된 VNB

[8.4](interest-guarantee) 의 금리보증 TVOG 를 마지막 차감으로 얹습니다.

```python
mats = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0])
rts  = np.array([0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.030])
scen = fcf.esg.simulate(mats, rts, ufr=0.035, alpha=0.1, mean_reversion=0.05,
    rate_vol=0.015, equity_vol=0.0, correlation=0.0, n_scenarios=2000, n_time=n_time, seed=20240601)
tv = pricing.interest_guarantee_tvog(endow, stat, scen.rates, initial_prices=scen.initial_prices)

ev2 = pricing.embedded_value(sig, reference_rate=0.05, discount_monthly=dm,
    required_capital=0.05, reserve=V, frictional_spread=0.06, tvog=tv.total_value)
print(f"PVFP {ev2.pvfp:,.0f}  CoC {ev2.cost_of_capital:,.0f}  "
      f"TVOG {ev2.tvog:,.0f}  VNB {ev2.value:,.0f}")
```

출력:

```text
PVFP 7,087,977  CoC 1,129,822  TVOG 4,000,485  VNB 1,957,670
```

PVFP 7,087,977 에서 자본비용 1,129,822 와 보증의 시간가치 4,000,485 를 빼면 신계약가치
1,957,670 -- 세 챕터의 조각이 한 식으로 모입니다.

## 결과 해석 / 변형

- **이익 기준 선택** -- `embedded_value` 는 `ProfitSignature` 만 보므로, 전통형
  `statutory_profit_signature` 든 IFRS17 `signature` 든 넘기는 대로 씁니다. 전통형
  (이익에 RA 가 없음) + 임의 요구자본이 깔끔한 기본 조합입니다.
- **요구자본 정의** -- v1 은 caller 공급 (명시 경로 또는 `factor x reserve`).
  실제 K-ICS 요구자본 (SCR) 은 후속 단계입니다.
- **할인율** -- 단일 `reference_rate` 가 이익·자본비용 스트림을 모두 할인하는 전통형
  EV 단일금리식. MCEV 의 reference + CRNHR 분해는 후속입니다.

## 함정 / 검증

- **이중계상 주의** -- IFRS17 `signature` (이익에 RA release 포함) 에
  confidence-level RA 경로를 요구자본으로 넣으면 이중계상은 아니나 관점이 섞입니다
  (RA release 는 가치, CoC 는 자본 보유의 마찰드래그 -- 다른 양). 전통형 시그니처
  조합이 더 깔끔합니다.
- **스칼라 factor 는 reserve 필요** -- `required_capital` 을 스칼라로 주면
  `reserve=` 를 함께 줘야 합니다 (명시적 에러).
- **세전 (v1)** -- `nbv` / `signature` 와 같이 세전입니다. 세금은 후속.

## 인접 레시피

- [8.3 수익성 분석 / profit-testing](profit-testing) -- PVFP 의 출발점인 이익
  시그니처와 `nbv` (필요자본 차감 전).
- [8.4 전통형 금리보증 비용 (TVOG)](interest-guarantee) -- VNB 의 TVOG 차감항.
- [경제적 가정 -- 시나리오 생성](../basics/scenario-generation) -- TVOG 가 쓰는
  금리시나리오.

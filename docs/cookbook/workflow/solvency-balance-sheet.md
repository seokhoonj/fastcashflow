# 8.8 자산 - 가용자본 - 지급여력비율

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **자산 포트폴리오** 를 시가평가하고 (`fcf.AssetPortfolio`, `fcf.Bond` / `Equity` /
  `Property` / `Cash`), **가용자본** (자산 - 부채) 을 산출 (`fcf.available_capital`)
- 자산과 부채를 같은 곡선충격으로 재평가하는 **순금리 SCR** (`fcf.net_interest_scr`)
- 한 번에 **지급여력비율** 까지 조립 (`fcf.assess_solvency`)
- v1 의 정직한 한계 -- 주식/부동산은 가용자본을 올리되 아직 SCR 에 안 들어가
  비율이 **상한** 이라는 점
:::

지급여력비율 = 가용자본 / SCR. fastcashflow 는 부채측 (BEL, SCR) 을 내고, 이 챕터는
**자산측을 정적 (t=0) 으로 시가평가** 해 비율의 분자를 완성합니다. SCR 은 순간충격-재평가라
자산을 t=0 에 한 번 평가하면 충분합니다 (동적 자산투영 불필요). 채권은 할인곡선으로 가격이
매겨지고 ([8.7 ALM](alm-duration) 의 `bond_value`), 주식·부동산·현금은 시가로 담습니다.

## 자산 + 가용자본 + 비율 -- 한 번에

부채 DV01 에 맞춘 채권 + 현금으로 백업한 보장성 계약. `assess_solvency` 가 전체 그림을
냅니다.

```python
import fastcashflow as fcf
from fastcashflow import alm

mp = fcf.ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 100_000_000.0},
                            calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
basis = fcf.Basis(mortality_annual=0.012, lapse_annual=0.0, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.10,
                  coverages=(fcf.CoverageRate("DEATH", 0.012),))

# a bond sized to the liability DV01, plus cash
liab_dv01 = alm.liability_dv01(mp, basis)
per_face = alm.bond_duration(alm.Bond(100.0, 0.03, 10, 1), 0.03).dv01
bond = alm.Bond(face=liab_dv01 / per_face * 100.0, coupon_rate=0.03, maturity_years=10, frequency=1)
port = fcf.AssetPortfolio(holdings=(bond, fcf.Cash(5_000_000.0)))

a = fcf.assess_solvency(port, mp, basis, regime=fcf.SOLVENCY2)
print(f"portfolio value   = {a.portfolio_value:>14,.0f}")
print(f"BEL + risk margin = {a.bel + a.risk_margin:>14,.0f}")
print(f"available capital = {a.available_capital:>14,.0f}")
print(f"insurance SCR     = {a.insurance_scr:>14,.0f}")
print(f"net interest SCR  = {a.net_interest_scr:>14,.0f}")
print(f"total SCR         = {a.total_scr:>14,.0f}")
print(f"solvency ratio    = {a.solvency_ratio:>13.1%}")
```

출력:

```text
portfolio value   =      7,649,930
BEL + risk margin =      5,359,741
available capital =      2,290,189
insurance SCR     =      1,423,820
net interest SCR  =         43,467
total SCR         =      1,467,287
solvency ratio    =        156.1%
```

가용자본 = 자산 (7,649,930) - 기술준비금 (BEL+위험마진 5,359,741) = 2,290,189. SCR 은
보험위험 + **순금리** (채권이 부채 DV01 에 매칭돼 43,467 로 작음 -- 면역에 가까움). 비율
156.1% 는 가용자본 / 총 SCR. 채권이 부채 금리민감도를 헤지하니 순금리 SCR 이 작습니다.

## v1 의 한계 -- 주식은 분자만 올린다

주식/부동산은 v1 에서 **시가 carrier** 입니다 -- 가용자본 (분자) 은 올리되 아직 시장위험
SCR (분모) 에 안 들어갑니다. 그래서 주식이 많으면 비율이 **과대** 됩니다.

```python
port2 = fcf.AssetPortfolio(holdings=(bond, fcf.Cash(5_000_000.0), fcf.Equity(3_000_000.0)))
b = fcf.assess_solvency(port2, mp, basis, regime=fcf.SOLVENCY2)
print(f"+3,000,000 equity -> available capital {b.available_capital:>14,.0f}")
print(f"                     total SCR          {b.total_scr:>14,.0f}  (unchanged)")
print(f"                     solvency ratio     {b.solvency_ratio:>13.1%}  (overstated -- no equity SCR yet)")
```

출력:

```text
+3,000,000 equity -> available capital      5,290,189
                     total SCR               1,467,287  (unchanged)
                     solvency ratio            360.5%  (overstated -- no equity SCR yet)
```

주식 3,000,000 을 더하면 가용자본은 5,290,189 로 오르지만 total SCR 은 그대로라 비율이
360.5% 로 뜁니다 -- 주식하락 SCR (SII -35%/-48%, 부동산 -25%) 이 아직 안 들어간
**상한** 입니다. 자산측 시장위험 SCR 은 후속 단계 (규제 충격수치 추출 필요) 입니다.

## 함정 / 검증

- **주식/부동산 SCR 미반영 (v1)** -- 위 한계. 채권 (금리) 과 부채는 완전하지만, 주식/
  부동산이 많은 책은 SCR 과소 -> 비율 과대. 채권 백업 책에서는 비율이 정확합니다.
- **순금리 SCR 은 자산+부채 net** -- 같은 곡선충격으로 둘 다 재평가, worst-of up/down.
  매칭 (DV01) 책은 0 에 가깝고, 미스매치는 양(+). K-ICS 는 `interest_curves` 가 없어
  (곡선 caller 공급) 부채측 금리값으로 fallback.
- **가용자본은 자산-기술준비금** -- 기타 대차대조표 부채가 있으면 포트폴리오 값에서 미리
  차감해 넘기세요. 계층화 (기본/보완자본) 는 v1 단순화 (순자산 총액).
- **정적 t=0** -- 동적 자산투영 (롤·재투자) = 동적 ALM 은 범위 밖. 표준공식 비율엔 불필요.

## 인접 레시피

- [8.6 요구자본 (Solvency II / K-ICS)](required-capital) -- 분모인 부채 SCR.
- [8.7 ALM -- 듀레이션 / DV01](alm-duration) -- 채권 가격·DV01, 부채 DV01 매칭.
- [8.5 임베디드밸류](embedded-value) -- 요구자본을 자본비용으로 받는 또 다른 소비처.

# 8.8 자산 - 가용자본 - 지급여력비율

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **자산 포트폴리오** 를 시가평가하고 (`fcf.AssetPortfolio`, `fcf.Bond` / `Equity` /
  `Property` / `Cash`), **가용자본** (자산 - 부채) 을 산출 (`fcf.available_capital`)
- 자산과 부채를 같은 곡선충격으로 재평가하는 **순금리 SCR** (`fcf.net_interest_scr`)
- **주식 / 부동산 시장위험 SCR** (선진 -35% / 신흥 -48% / 부동산 -25%) 과 시장모듈 집계
- 한 번에 **지급여력비율** 까지 조립 (`fcf.assess_solvency`) -- 보험위험 + 시장위험을
  BSCR 로 묶어 (K-ICS 는 상관집계, Solvency II 는 단순합)
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
print(f"operational SCR   = {a.operational_scr:>14,.0f}")
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
operational SCR   =         23,868
total SCR         =      1,491,155
solvency ratio    =        153.6%
```

가용자본 = 자산 (7,649,930) - 기술준비금 (BEL+위험마진 5,359,741) = 2,290,189. SCR 은
보험위험 + **순금리** (채권이 부채 DV01 에 매칭돼 43,467 로 작음 -- 면역에 가까움) +
**운영위험** (보험료·BEL 의 factor, 23,868). 비율 153.6% 는 가용자본 / 총 SCR. 채권이
부채 금리민감도를 헤지하니 순금리 SCR 이 작습니다.

## 주식 / 부동산 -- 시장위험 SCR

주식·부동산은 가용자본 (분자) 을 올리는 동시에 **시장위험 SCR (분모)** 도 매깁니다. 주식
3,000,000 (선진시장) 을 더하면:

```python
port2 = fcf.AssetPortfolio(holdings=(bond, fcf.Cash(5_000_000.0),
                                     fcf.Equity(3_000_000.0, "developed")))
b = fcf.assess_solvency(port2, mp, basis, regime=fcf.SOLVENCY2)
print(f"+3,000,000 equity -> equity SCR      {b.equity_scr:>14,.0f}")
print(f"                     market module    {b.market_module_scr:>14,.0f}")
print(f"                     BSCR             {b.bscr:>14,.0f}")
print(f"                     operational SCR  {b.operational_scr:>14,.0f}")
print(f"                     total SCR        {b.total_scr:>14,.0f}")
print(f"                     available capital{b.available_capital:>14,.0f}")
print(f"                     solvency ratio   {b.solvency_ratio:>13.1%}")
```

출력:

```text
+3,000,000 equity -> equity SCR           1,050,000
                     market module         1,061,701
                     BSCR                  2,485,521
                     operational SCR          23,868
                     total SCR             2,509,389
                     available capital     5,290,189
                     solvency ratio          210.8%
```

주식하락 충격 (선진시장 -35%) 으로 주식 SCR 1,050,000 이 잡히고, 금리 (43,467) 와 함께
시장모듈 (1,061,701, 상관 0.25) 로 묶입니다. BSCR 은 보험위험 (1,423,820) 과 시장모듈을
top-level 집계 -- Solvency II 는 단순합 (2,485,521), K-ICS 는 0.25 상관집계. 거기에
운영위험 (23,868) 을 더해 총 SCR 2,509,389. 가용자본은
주식만큼 올라 5,290,189 지만 SCR 도 같이 올라 비율은 210.8% 로 **분자만 오르던 과대가
사라졌습니다**.

## 함정 / 검증

- **운영위험은 부채 factor** -- 보험료·BEL 에 위험계수 (K-ICS max(보험료 3.5%, BEL 0.4%) /
  SII min(0.3 BSCR, max(0.04 보험료, 0.0045 BEL))) 를 적용해 BSCR 위에 가산합니다.
- **주식 세분·자산군은 일부만** -- 주식은 선진/신흥, 부동산 단일까지 반영했습니다.
  인프라/장기보유/우선주/기타 주식 세분, 외환·자산집중·**신용** 위험액은 후속 -- 그런
  자산이 큰 책은 아직 SCR 과소입니다.
- **SII top-level 은 단순합** -- Solvency II 의 모듈간 상관행렬 (Directive Annex IV) 은
  여기서 추출 못 해, 보험 + 시장을 분산효과 없이 단순합 (보수적). K-ICS 는 0.25 상관집계.
- **순금리 SCR 은 자산+부채 net** -- 같은 곡선충격으로 둘 다 재평가, worst-of up/down.
  매칭 (DV01) 책은 0 에 가깝고, 미스매치는 양(+). K-ICS 는 `interest_curves` 가 없어
  (곡선 caller 공급) 순금리 성분이 0 -- 주식/부동산은 그대로 잡힙니다.
- **가용자본은 자산-기술준비금** -- 기타 대차대조표 부채가 있으면 포트폴리오 값에서 미리
  차감해 넘기세요. 계층화 (기본/보완자본) 는 v1 단순화 (순자산 총액).
- **정적 t=0** -- 동적 자산투영 (롤·재투자) = 동적 ALM 은 범위 밖. 표준공식 비율엔 불필요.

## 인접 레시피

- [8.6 요구자본 (Solvency II / K-ICS)](required-capital) -- 분모인 부채 SCR.
- [8.7 ALM -- 듀레이션 / DV01](alm-duration) -- 채권 가격·DV01, 부채 DV01 매칭.
- [8.5 임베디드밸류](embedded-value) -- 요구자본을 자본비용으로 받는 또 다른 소비처.

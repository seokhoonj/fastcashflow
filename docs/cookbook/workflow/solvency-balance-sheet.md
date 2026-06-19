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

## 신용위험 SCR -- K-ICS

채권은 신용위험을 집니다 -- 발행자 부도와 등급하락. K-ICS 는 이를 **신용등급 x 유효만기**
격자의 위험계수 (부도 + 등급하락 부담률, 시가의 %) 로 매깁니다 (공공 / 일반기업 / 유동화
익스포저별로 표가 다름). `Bond` 에 `credit_rating` (AAA~D / unrated) 과 `exposure_class`
(corporate / public / securitisation) 를 주면 됩니다. 유효만기는 현금흐름가중 평균만기
(`fcf.effective_maturity`) 라 같은 만기라도 쿠폰이 크면 짧아집니다.

```python
mixed = fcf.AssetPortfolio(holdings=(
    alm.Bond(3_000_000.0, 0.03, 10, 1, credit_rating="AA", exposure_class="corporate"),
    alm.Bond(2_000_000.0, 0.04, 8, 1, credit_rating="BBB", exposure_class="corporate"),
    fcf.Cash(2_000_000.0)))
k = fcf.assess_solvency(mixed, mp, basis, regime=fcf.KICS)
print(f"insurance SCR     = {k.insurance_scr:>14,.0f}")
print(f"credit SCR        = {k.credit_scr:>14,.0f}")
print(f"market module SCR = {k.market_module_scr:>14,.0f}")
print(f"BSCR              = {k.bscr:>14,.0f}")
print(f"operational SCR   = {k.operational_scr:>14,.0f}")
print(f"total SCR         = {k.total_scr:>14,.0f}")
print(f"solvency ratio    = {k.solvency_ratio:>13.1%}")
```

출력:

```text
insurance SCR     =      1,187,554
credit SCR        =        173,441
market module SCR =              0
BSCR              =      1,242,317
operational SCR   =         20,884
total SCR         =      1,263,201
solvency ratio    =        135.1%
```

낮은 등급 (BBB) 과 긴 만기일수록 위험계수가 큽니다. BSCR 은 보험 + 시장 + **신용** 을
table 3 상관 (전부 0.25) 으로 묶습니다 (여기선 K-ICS 라 곡선 미공급 -> 시장모듈 0).
신용위험은 K-ICS 격자만 반영했고, Solvency II 의 스프레드 / 거래상대방 위험은 별도
체계라 아직 0 (후속) 입니다.

## 외환위험 SCR -- K-ICS

외화자산은 환율위험을 집니다. K-ICS 는 통화별 순익스포저에 원화 기준 환율충격 (표22, 통화
별로 다름 -- USD 25% / EUR 35% / JPY 40% ...) 을 주고, **순자산이 감소하는 통화만 상관 0.5**
로 묶어 (원화상승 / 원화하락 시나리오 중 나쁜 쪽) 외환위험액을 냅니다. 자산에 `currency`
(ISO 코드) 를 달면 됩니다. 외환은 시장모듈의 네 번째 하위위험으로, 금리 / 주식 / 부동산과
표19 상관으로 묶입니다 (주식-외환은 **음(-)의 0.25** -- 환율 급등 시 외국인 매도로 주가가
빠지는 국내 특성).

```python
fxport = fcf.AssetPortfolio(holdings=(
    alm.Bond(3_000_000.0, 0.03, 10, 1, credit_rating="A", currency="USD"),
    alm.Bond(2_000_000.0, 0.03, 8, 1, credit_rating="AA", currency="EUR"),
    fcf.Cash(2_500_000.0)))
k = fcf.assess_solvency(fxport, mp, basis, regime=fcf.KICS)
print(f"FX SCR            = {k.fx_scr:>14,.0f}")
print(f"credit SCR        = {k.credit_scr:>14,.0f}")
print(f"market module SCR = {k.market_module_scr:>14,.0f}")
print(f"insurance SCR     = {k.insurance_scr:>14,.0f}")
print(f"BSCR              = {k.bscr:>14,.0f}")
print(f"total SCR         = {k.total_scr:>14,.0f}")
print(f"solvency ratio    = {k.solvency_ratio:>13.1%}")
```

출력:

```text
FX SCR            =      1,255,986
credit SCR        =        128,000
market module SCR =      1,255,986
insurance SCR     =      1,187,554
BSCR              =      1,976,444
total SCR         =      1,997,328
solvency ratio    =        103.5%
```

USD 채권 (시가 x 25%) 과 EUR 채권 (x 35%) 의 손실을 상관 0.5 로 묶어
sqrt(750k^2 + 700k^2 + 2 x 0.5 x 750k x 700k) = 1,255,986. 외화 비중이 크면 외환위험액이
요구자본을 지배합니다 (비율 103.5%). 외환위험은 K-ICS 만 반영했고, Solvency II 의 통화위험
(별도 충격) 은 후속입니다.

## 자산집중위험 SCR -- K-ICS

분산이 부족한 (한 발행자 / 한 부동산에 쏠린) 책은 추가 위험을 집니다. K-ICS 는 **총자산
대비 한도** 를 넘는 익스포저에만 위험계수를 매깁니다 -- 거래상대방은 신용등급별 한도 (1~2
등급 4% / 3~4등급 3% / 5~7등급 1.5%, 표23), 부동산은 개별 6% / 전체 25% (표24). 자산에
`issuer` (거래상대방) 를 달면 같은 발행자끼리 묶입니다.

```python
conc = fcf.AssetPortfolio(holdings=(
    alm.Bond(4_000_000.0, 0.03, 7, 1, credit_rating="A", issuer="BankA"),
    alm.Bond(2_000_000.0, 0.04, 5, 1, credit_rating="A", issuer="BankA"),  # same issuer
    fcf.Property(3_000_000.0),
    fcf.Cash(3_000_000.0)))
k = fcf.assess_solvency(conc, mp, basis, regime=fcf.KICS)
print(f"concentration SCR = {k.concentration_scr:>14,.0f}")
print(f"market module SCR = {k.market_module_scr:>14,.0f}")
print(f"insurance SCR     = {k.insurance_scr:>14,.0f}")
print(f"BSCR              = {k.bscr:>14,.0f}")
print(f"total SCR         = {k.total_scr:>14,.0f}")
print(f"solvency ratio    = {k.solvency_ratio:>13.1%}")
```

출력:

```text
concentration SCR =      1,502,719
market module SCR =      1,679,483
insurance SCR     =      1,187,554
BSCR              =      2,337,118
total SCR         =      2,358,002
solvency ratio    =        282.4%
```

발행자 BankA 에 6,000,000 (총자산의 ~50%) 이 쏠려 한도초과분에 25% 가 붙고, 부동산
3,000,000 도 개별한도 (6%) 를 넘어 집중위험액이 큽니다. 자산집중은 다른 시장 하위위험과
**상관 0** (각 자산의 고유위험) 이라 시장모듈에 제곱합으로 더해집니다. K-ICS 만 반영했고
Solvency II 의 집중위험 (별도 체계) 은 후속입니다.

## 법인세효과 -- 총요구자본

충격 시 손실이 나면 과세소득이 줄어 법인세가 절감되고, 이 절감분만큼 손실이 흡수됩니다.
K-ICS 는 이를 **총요구자본 = 기본요구자본 - 법인세조정액** 으로 반영합니다
(법인세조정액 = min(기본요구자본 x 평균세율, 실현가능성 한도), 해설서 제7장). 평균세율은
회사별 (직전 3년 세전이익 기준) 이라 인자로 받습니다 (기본값 0 = 미반영, 보수적).

```python
port = fcf.AssetPortfolio(holdings=(
    alm.Bond(3_000_000.0, 0.03, 10, 1, credit_rating="AA"), fcf.Cash(4_000_000.0)))
a = fcf.assess_solvency(port, mp, basis, regime=fcf.KICS, tax_rate=0.22)
print(f"basic required capital = {a.basic_required_capital:>14,.0f}")
print(f"  tax adjustment       = {a.tax_adjustment:>14,.0f}")
print(f"total required capital = {a.total_scr:>14,.0f}")
print(f"solvency ratio         = {a.solvency_ratio:>13.1%}")
```

출력:

```text
basic required capital =      1,224,840
  tax adjustment       =        269,465
total required capital =        955,376
solvency ratio         =        164.0%
```

평균세율 22% 면 기본요구자본의 22% (269,465) 가 법인세조정액으로 차감돼 총요구자본이
955,376 으로 줄고, 비율은 (세효과 미반영) 127.9% 에서 164.0% 로 올라갑니다. 실현가능성
한도 (직전 5년 세전이익 x 50% + 순이연법인세 등) 가 있으면 `tax_recoverability_limit` 로
넘기면 차감액이 그만큼 캡 됩니다.

## 함정 / 검증

- **법인세효과는 옵트인** -- `tax_rate` 기본값 0 이면 총요구자본 = 기본요구자본 (보수적).
  평균세율을 주면 기본요구자본 x 세율 (한도 내) 만큼 차감해 비율이 올라갑니다. 회사별
  세무자료 (세전이익 / 이연법인세) 는 엔진 밖이라 세율 / 한도를 caller 가 공급합니다.
- **자산집중은 K-ICS, SII 는 후속** -- 자산집중위험액 = sqrt(거래상대방집중^2 + 부동산집중^2),
  각 한도초과분 (총자산 x 등급별 / 부동산 한도) 에 위험계수. 발행자 미태깅 + 부동산 없으면 0.
  시장모듈에 상관 0 으로 들어갑니다. Solvency II 집중위험은 별도라 0 (후속).
- **외환위험은 K-ICS, SII 는 후속** -- 외환위험액 = max(원화상승, 원화하락) 손실 (통화별
  표22 충격, 감소통화만 상관 0.5) + 가격변동 (파생, v1 은 0). 시장모듈에 네 번째로 들어가며
  주식-외환 상관은 음(-) 입니다. Solvency II 통화위험은 별도 체계라 0 (후속).
- **신용위험은 K-ICS 격자, SII 는 후속** -- 채권 신용 SCR = 시가 x 위험계수 (신용등급 x
  유효만기, 공공 / 일반기업 / 유동화 표). Solvency II 의 스프레드 / 거래상대방 위험은 별도
  체계라 미반영 (0). 현금은 무위험, 주식 / 부동산은 시장위험으로 잡힙니다.
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

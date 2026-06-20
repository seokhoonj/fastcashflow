# 8.9 공시 재현 -- IFRS17 부채 + K-ICS 지급여력

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 한국 보험사 공개공시 (DART 사업/분기보고서) 의 세 핵심 표를 fastcashflow 로 재현:
  **IFRS17 보험계약부채** (최선추정 + 위험조정 + 보험계약마진), **K-ICS 지급여력**
  (가용자본 / 요구자본 / 비율), **금리 민감도**
- 이 챕터들에서 만든 도구 (`measure` / `estimate_at` / `assess_solvency`) 가 실제
  공시 골격과 어떻게 맞물리는지
:::

실제 생보사 공시는 세 가지를 냅니다 -- IFRS17 보험계약부채를 **최선추정 (BEL) + 위험조정
(RA) + 보험계약마진 (CSM)** 으로 분해하고, K-ICS **지급여력비율 = 가용자본 / 요구자본**
(요구자본은 보험·시장·신용·운영 위험액의 분산집계) 을 내며, 주요 가정의 **민감도** 를
공시합니다. fastcashflow 의 측정·자본·ALM 이 이 골격을 그대로 재현합니다 (여기선
합성데이터 -- 실제 공시값과 1:1 이 아니라 **구조와 크기 범위** 를 재현).

## IFRS17 보험계약부채 -- BEL + RA + CSM

보유계약 (in-force) 한 시점의 보험계약부채 분해. 수익성 계약은 신계약 시점엔 부채가
0 (day-1 이익 없음) 이고, 보유로 가면 적립금 (BEL) 이 쌓이고 CSM 이 상각됩니다 --
`estimate_at` 로 보유 5년차 잔액을 봅니다.

```python
import numpy as np
from dataclasses import replace
import fastcashflow as fcf
from fastcashflow import pricing
from fastcashflow.engine import measure

endow0 = fcf.ModelPoints.single(40, 0.0, 240, benefits={"DEATH": 100_000_000.0},
    maturity_benefit=100_000_000.0, calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
eb = fcf.Basis(mortality_annual=0.01, lapse_annual=0.0, discount_annual=0.03,
    ra_confidence=0.75, mortality_cv=0.10, coverages=(fcf.CoverageRate("DEATH", 0.01),))
endow = replace(endow0, premium=np.full(1, pricing.solve_premium(endow0, eb, margin=0.10)[0]))

e = measure(endow, eb, full=True).estimate_at(60)          # in force, month 60
bel, ra, csm = float(np.sum(e.bel)), float(np.sum(e.ra)), float(np.sum(e.csm))
print(f"  best estimate (BEL)    = {bel:>15,.0f}")
print(f"  risk adjustment (RA)   = {ra:>15,.0f}")
print(f"  CSM                    = {csm:>15,.0f}")
print(f"  insurance liability    = {bel + ra + csm:>15,.0f}")
```

출력:

```text
  best estimate (BEL)    =      11,310,206
  risk adjustment (RA)   =         729,083
  CSM                    =       5,658,276
  insurance liability    =      17,697,565
```

공시의 `보험계약부채 = 최선추정액 + 위험조정 + 보험계약마진` 과 같은 분해입니다 (BEL=
현행추정부채=최선추정, RA=위험마진=위험조정, CSM=보험계약마진). 실제 대형사는 이 합이
수십~수백조; 유배당 / 무배당 / 변액으로 더 나뉩니다 (변액은 `vfa.measure`). **현실 비율
감각** -- 한 대형 생보사 FY2025 공시는 BEL 206.9조, RA 4.58조 (**RA/BEL ~2.2%**), CSM 13.5조
(**CSM/BEL ~6.5%**) 였습니다. RA 는 BEL 의 수% 수준이 자연스럽습니다.

## K-ICS 지급여력 -- 가용자본 / 요구자본 / 비율

보장성 계약을 채권·현금·주식으로 백업한 포지션. `assess_solvency` 가 가용자본, 요구자본
(보험위험 + 시장위험), 비율을 냅니다.

```python
from fastcashflow import alm

mp = fcf.ModelPoints.single(40, 60_000.0, 240, benefits={"DEATH": 100_000_000.0},
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
basis = fcf.Basis(mortality_annual=0.012, lapse_annual=0.03, discount_annual=0.03,
    ra_confidence=0.75, mortality_cv=0.10, coverages=(fcf.CoverageRate("DEATH", 0.012),))
dv01 = alm.liability_dv01(mp, basis)
per = alm.bond_duration(alm.Bond(100.0, 0.03, 15, 1), 0.03).dv01
port = fcf.AssetPortfolio(holdings=(alm.Bond(dv01 / per * 100.0, 0.03, 15, 1),
                                    fcf.Cash(6_000_000.0), fcf.Equity(2_000_000.0)))

a = fcf.assess_solvency(port, mp, basis, regime=fcf.SOLVENCY2)
print(f"  available capital      = {a.available_capital:>15,.0f}")
print(f"  required capital (SCR) = {a.total_scr:>15,.0f}")
print(f"    insurance risk       = {a.insurance_scr:>15,.0f}")
print(f"    market risk          = {a.market_module_scr:>15,.0f}")
print(f"    credit risk          = {a.credit_scr:>15,.0f}")
print(f"    operational risk     = {a.operational_scr:>15,.0f}")
print(f"  solvency ratio         = {a.solvency_ratio:>14.1%}")
```

출력:

```text
  available capital      =       5,169,221
  required capital (SCR) =       2,410,964
    insurance risk       =       1,987,539
    market risk          =         713,427
    credit risk          =         307,115
    operational risk     =          28,246
  solvency ratio         =         214.4%
```

`지급여력비율 = 가용자본 / 요구자본` 의 214.4% 는 실제 대형 생보사의 공시 범위
(대략 185~210%) 근처입니다 -- 합성데이터인데도 **구조와 크기** 가 현실 공시와 맞물립니다.
요구자본은 보험·시장·신용·운영 위험의 집계입니다 (여기선 Solvency II regime 이라 신용은
Art 176 스프레드).

## 실제 공시 요구자본 재현 -- 모듈 위험액 -> 기본요구자본

위 합성예제는 구조를 보여주지만, **공시된 모듈 위험액을 그대로 넣으면 실제 지급여력기준금액이
원 단위로 재현**됩니다. `fcf.aggregate_required_capital` 이 표3 상관집계 (생명장기·시장·신용
0.25, 운영은 밖에서 가산) 를 합니다. 아래는 **국내 대형 생보사 FY2025 경영공시 (DART)** 의
요구자본 모듈 (단위: 백만원) 입니다.

```python
ins, mkt, cr, op = 11_628_115, 34_552_189, 4_166_014, 1_083_844   # 공시 모듈 위험액
tax, other = 11_153_682, 2_737_202                                # 법인세조정 / 기타요구자본
avail = 65_740_200                                                # 지급여력금액 (가용자본)

basic = fcf.aggregate_required_capital(ins, mkt, cr, regime=fcf.KICS, operational=op)
div = (ins + mkt + cr) - fcf.aggregate_required_capital(ins, mkt, cr, regime=fcf.KICS)
total = basic - tax + other                                       # 기본 - 법인세 + 기타
print(f"basic required capital = {basic:>14,.0f}")
print(f"  diversification      = {div:>14,.0f}")
print(f"total required capital = {total:>14,.0f}")
print(f"solvency ratio         = {avail / total:>13.1%}")
```

출력:

```text
basic required capital =     41,624,007
  diversification      =      9,806,155
total required capital =     33,207,527
solvency ratio         =        198.0%
```

공시값 (기본요구자본 41,624,006 / 분산효과 9,806,156 / 지급여력기준금액 33,207,526 / 비율
198.0%) 과 **반올림 (+/-1 백만원) 까지 일치**합니다. 표3 0.25 상관·운영 외부가산·법인세조정·
기타요구자본 구조가 그대로 재현됩니다. 손보를 겸영하면 `general_insurance=` 로 일반손해
모듈을 넷째로 더합니다 (생명장기-일반손해 상관 0). 모듈 위험액 자체는 보유계약을 측정해
나오지만 (앞 챕터들), 이 집계 단계는 **공시와 동일한 산식**입니다.

## 금리 민감도

공시의 금리 민감도 표 -- 할인율 +/-50bp 에 보험계약부채 (최선추정) 가 어떻게 움직이는지.

```python
base = float(measure(mp, basis, full=False).bel.sum())
for bp in (-50, 50):
    s = float(measure(mp, replace(basis, discount_annual=0.03 + bp / 10000.0)).bel.sum())
    print(f"  {bp:>+4d}bp -> BEL {s:>13,.0f}  ({s - base:>+11,.0f})")
```

출력:

```text
   -50bp -> BEL     5,353,457  (   +199,686)
   +50bp -> BEL     4,965,963  (   -187,808)
```

금리가 내리면 부채가 늘고 (할인 약화), 오르면 줄어듭니다. 이 민감도가 [8.7 ALM](alm-duration)
의 DV01 과 같은 뿌리이고, [8.8](solvency-balance-sheet) 의 순금리 SCR 로도 들어갑니다.

## 함정 / 검증

- **합성데이터, 구조 재현** -- 실제 공시값과 1:1 이 아니라 분해·모듈구성·비율 범위를
  재현합니다. 회사 전체 공시는 수많은 계약·자산의 집계 (per-portfolio 합) 입니다.
- **시점 일관성** -- 완전한 공시는 보고시점 한 날에 자산·부채를 함께 평가합니다. 위 세
  표는 각 항목의 골격을 보이는 예시 (IFRS17 부채는 보유시점, K-ICS 는 별도 백업 포지션).
- **모듈 구성** -- 보험·시장·신용·운영 위험액이 들어 있습니다. 이 예시는 Solvency II
  regime 이라 신용은 Art 176 스프레드 (위 307,115) 로 잡힙니다. 외환·자산집중도 양 regime
  구현돼 있고 (해당 익스포저가 있을 때 발동), top-level 모듈간 상관 (SII Directive Annex IV)
  만 미추출이라 SII 는 단순합 (보수적) -- 그만큼 비율이 낮게 나옵니다.

## 인접 레시피

- [9.4 결산팩 -- 공시 명세서 조립](close-pack) -- SoFP / 보험금융손익 / 보험서비스손익을
  세그먼트별로 조립하는 결산 공시.
- [8.5 임베디드밸류](embedded-value) · [8.6 요구자본](required-capital) ·
  [8.8 지급여력비율](solvency-balance-sheet) -- 위 표의 각 구성요소.

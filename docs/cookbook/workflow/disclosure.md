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

공시의 `보험계약부채 = 최선추정액 + 위험조정 + 보험계약마진` 과 같은 분해입니다 (실제
대형사는 이 합이 수십~수백조; 유배당 / 무배당 / 변액으로 더 나뉩니다 -- 변액은
`vfa.measure`).

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
print(f"    operational risk     = {a.operational_scr:>15,.0f}")
print(f"  solvency ratio         = {a.solvency_ratio:>14.1%}")
```

출력:

```text
  available capital      =       5,191,995
  required capital (SCR) =       2,680,035
    insurance risk       =       1,938,361
    market risk          =         713,427
    operational risk     =          28,246
  solvency ratio         =         193.7%
```

`지급여력비율 = 가용자본 / 요구자본` 의 193.7% 는 실제 대형 생보사의 공시 범위
(대략 185~210%) 에 듭니다 -- 합성데이터인데도 **구조와 크기** 가 현실 공시와 맞물립니다.
요구자본은 보험위험과 시장위험 (금리·주식·부동산) 의 분산집계입니다.

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
- **미반영 모듈** -- K-ICS 요구자본의 신용·운영·외환·자산집중 위험액은 아직 빠져 있어
  (공시엔 있음) 비율은 그만큼 보수적이지 않습니다 ([8.8](solvency-balance-sheet) 참고).

## 인접 레시피

- [9.4 결산팩 -- 공시 명세서 조립](close-pack) -- SoFP / 보험금융손익 / 보험서비스손익을
  세그먼트별로 조립하는 결산 공시.
- [8.5 임베디드밸류](embedded-value) · [8.6 요구자본](required-capital) ·
  [8.8 지급여력비율](solvency-balance-sheet) -- 위 표의 각 구성요소.

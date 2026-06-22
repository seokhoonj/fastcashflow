# 8.7 ALM -- 듀레이션 / DV01 / 자산-부채 gap

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 부채의 **금리 민감도** -- DV01 과 effective duration -- 을 재측정으로 산출
  (`fcf.liability_duration`, `fcf.liability_dv01`)
- 공개 EV/IFRS17 공시처럼 **금리 +/-100bp** 부채 변화를 보는 법
- **Key-rate DV01** (연도 버킷별 민감도) 로 금리 노출의 만기 구조를 분해
  (`fcf.key_rate_dv01s`)
- **채권** 듀레이션 (Macaulay / Modified / DV01) 과 **자산-부채 DV01 gap** 으로
  면역화 (immunisation) 를 보는 법 (`fcf.alm.Bond`, `fcf.bond_duration`, `fcf.alm_gap`)
:::

ALM 의 실무적 시작은 거창한 동적 모델이 아니라 **결정론적 금리 민감도** -- 듀레이션과
DV01 (**D**ollar **V**alue of an **01** = 1bp 당 가치변화) 입니다. 유럽 생보 ALM 도
자산듀레이션·부채듀레이션·금리민감도 관리 체계이고, 공개 공시도 금리 +/-100bp 민감도를
냅니다. fastcashflow 는 재료를 이미 다 갖고 있어 -- 부채 현금흐름, 할인곡선, 충격곡선
재측정 -- ESG 없이 결정론적으로 바로 나옵니다.

DV01 이 자산과 부채를 잇는 **공통 단위** 입니다 (같은 곡선·같은 관례). 부채 DV01 에 자산
DV01 을 맞추면 평행 금리이동에 면역.

## 부채 듀레이션 / DV01 과 +/-100bp 민감도

```python
import numpy as np
from dataclasses import replace
import fastcashflow as fcf
from fastcashflow import alm
from fastcashflow.engine import measure

mp = fcf.ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 100_000_000.0},
                            calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
basis = fcf.Basis(mortality_annual=0.012, lapse_annual=0.0, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.0,
                  coverages=(fcf.CoverageRate("DEATH", 0.012),))

d = alm.liability_duration(mp, basis)
print(f"BEL              = {d.pv:>14,.0f}")
print(f"effective duration = {d.modified:>8.2f} yr")
print(f"DV01             = {d.dv01:>14,.0f}")
up = float(measure(mp, replace(basis, discount_annual=0.03 + 0.01)).bel.sum())
dn = float(measure(mp, replace(basis, discount_annual=0.03 - 0.01)).bel.sum())
print(f"BEL @ +100bp     = {up:>14,.0f}  ({up - d.pv:+,.0f})")
print(f"BEL @ -100bp     = {dn:>14,.0f}  ({dn - d.pv:+,.0f})")
```

출력:

```text
BEL              =      4,958,586
effective duration =     4.56 yr
DV01             =          2,260
BEL @ +100bp     =      4,740,483  (-218,104)
BEL @ -100bp     =      5,193,061  (+234,475)
```

DV01 2,260 은 금리 1bp 상승당 BEL 이 약 2,260 줄어든다는 뜻 -- 양(+)준비금 부채라 금리가
오르면 부채가 줄고 (자산엔 손해, 부채엔 이득). +100bp 면 약 -218,104 (DV01 x 100 에
convexity 약간) 입니다. effective duration 4.56년 = BEL 의 금리 민감 평균만기.

## Key-rate DV01 -- 노출의 만기 구조

평행 DV01 을 **연도 버킷별** 로 분해합니다 (각 연도의 곡선만 충격). 버킷 합이 평행 DV01.

```python
krd = alm.key_rate_dv01s(mp, basis)
for yr, k in enumerate(krd, 1):
    print(f"  year {yr:>2}: {k:>12,.0f}")
print(f"  sum    : {krd.sum():>12,.0f}  (~ parallel DV01)")
```

출력:

```text
  year  1:          455
  year  2:          398
  year  3:          344
  year  4:          292
  year  5:          242
  year  6:          194
  year  7:          148
  year  8:          104
  year  9:           62
  year 10:           21
  sum    :        2,260  (~ parallel DV01)
```

민감도가 앞쪽 만기에 실립니다 (사망보험금이 인포스가 큰 초기에 더 무게). 합 2,260 이
평행 DV01 과 일치 -- key-rate 가 그 분해입니다.

## 채권과 자산-부채 gap -- 면역화

부채 DV01 에 맞춘 채권 묶음은 평행 금리이동에 **gap 0** (면역) 입니다.

```python
bond = alm.Bond(face=100.0, coupon_rate=0.03, maturity_years=10, frequency=1)
bd = alm.bond_duration(bond, 0.03)
print(f"bond: value {bd.pv:.2f}  Macaulay {bd.macaulay:.2f}yr  "
      f"Modified {bd.modified:.2f}yr  DV01 {bd.dv01:.4f}")

face = d.dv01 / bd.dv01 * 100.0           # face sized so bond DV01 == liability DV01
matched = alm.Bond(face=face, coupon_rate=0.03, maturity_years=10, frequency=1)
g = alm.alm_gap(alm.bond_duration(matched, 0.03).dv01, d.dv01)
print(f"bond face to match: {face:>14,.0f}")
print(f"|dv01 gap|       = {abs(g['dv01_gap']):>14,.2f}  (~0 = immunised)")
```

출력:

```text
bond: value 100.00  Macaulay 8.79yr  Modified 8.53yr  DV01 0.0853
bond face to match:      2,649,930
|dv01 gap|       =           0.00  (~0 = immunised)
```

채권은 단일부호 현금흐름이라 **교과서 Macaulay/Modified** 가 깨끗합니다 (par 3% 10년 ->
Macaulay 8.79년). 부채 DV01 2,260 을 채권 DV01 (face 100 당 0.0853) 로 나눠 face 약
265만을 보유하면 두 DV01 이 상쇄 -- 평행 금리이동에 순가치가 면역됩니다. (만기별 mismatch
는 key-rate 로 더 정밀하게.)

## 함정 / 검증

- **부채는 DV01 이 헤드라인** -- 보험료(-)·청구(+) 혼합부호라 수익성 계약은 BEL 이 0/음수에
  가깝고, 그러면 Macaulay/Modified 비율이 ill-conditioned (`modified` 가 `nan` 으로 guard).
  DV01 (달러 민감도) 은 BEL 부호·크기 무관 항상 정의됩니다.
- **effective vs analytic** -- 부채 DV01 은 곡선 +/-1bp **재측정** (engine 타이밍·optionality
  반영). 채권 DV01 은 닫힌형 `Modified x value x 1bp`; 둘은 재가격 교차검증으로 일치.
- **면역은 평행 이동 한정** -- gap 0 은 평행 금리이동만 막습니다. 비평행 (twist) 은 key-rate
  DV01 로 버킷별 매칭해야 합니다.
- **v1 범위** -- 부채 DV01/duration/KRD, 채권 듀레이션, DV01 gap. 자산 포트폴리오 투영
  (롤·재투자) = 동적 ALM, real-world ESG, convexity, 신용스프레드 DV01 (CS01) 은 후속입니다.

## 인접 레시피

- [8.6 요구자본 (Solvency II / K-ICS)](required-capital) -- 금리위험 SCR 도 같은 곡선
  충격 재측정. ALM 의 금리 민감도와 규제 금리자본은 같은 뿌리.
- [8.1 시나리오 / 민감도](sensitivity) -- 충격 후 재측정 관용구.
- [경제적 가정 -- 시나리오 생성](../basics/scenario-generation) -- 곡선과 (후속) 동적 ALM 의 토대.

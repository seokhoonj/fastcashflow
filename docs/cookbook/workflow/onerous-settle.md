# 9.3 손실부담 계약과 경험조정의 결산

[9.1](settlement) 은 보유계약 결산의 뼈대 (`gmm.settle` -> `reconcile` ->
분기 체이닝) 를 다뤘습니다. 이 챕터는 그 위에 얹히는 **고급 메커니즘 네
가지** 를 한 권으로 묶습니다 — 모두 `gmm.settle` 의 movement 에 라인으로
이미 들어있고, 입력 한 컬럼이나 산출기초 한 항목으로 켜집니다.

| 메커니즘 | IFRS 17 | movement 라인 | 켜는 입력 |
|---|---|---|---|
| 손실요소 체계적 배분 | Sec. 50(a)-52 | `loss_component_finance` / `_amortised` | `prior_loss_component` (손실부담 계약) |
| 발생손해부채 (LIC) | Sec. 40(b) / 42(c) / 37 | `lic_opening` / `claims_incurred` / `lic_finance` / `claims_paid` / `lic_closing` | `settlement_pattern` (산출기초) |
| 보험료 경험조정 | Sec. B96(a) / B97(c) | `csm_premium_experience` / `premium_experience_revenue` | `actual_premium` (결산 상태) |
| 투자요소 경험조정 | Sec. B96(c) | `csm_investment_experience` | `actual_investment_component` (결산 상태) |

예제는 `gmm.settle` 로 보이지만 **`vfa.settle` 도 같은 라인을 똑같이 냅니다**
(맨 끝 절). 자기 입력이 없으면 라인은 0 으로, 결과는 기존 결산과 byte 단위로
같습니다.

아래 예제들이 공유하는 산출기초 / 보유계약 헬퍼를 먼저 정의합니다.

```python
import fastcashflow as fcf
import numpy as np
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)

CM    = {"DEATH": CalculationMethod.DEATH}
death = lambda sex, issue_age, dur: np.full(dur.shape, 0.012)   # 평탄 사망률

def basis(*, settlement=None):
    kw = {} if settlement is None else {"settlement_pattern": settlement}
    return Basis(
        mortality_annual=death,                                # 탈퇴 (in-force decrement)
        lapse_annual=lambda s, ia, d: np.full(d.shape, 0.05),  # 해지 (평탄 5%)
        discount_annual=0.03,                                  # 할인율
        ra_confidence=0.75,                                    # 위험조정 신뢰수준
        mortality_cv=0.10,                                     # 사망 변동계수
        coverages=(CoverageRate("DEATH", death),),             # 사망 보장
        **kw)

def book(b, *, prior_csm=0.0, lc_open=0.0,
         actual_premium=None, actual_ic=None, em_open=12, period=12, term=36):
    # 한 보유계약을 가입 후 em_open + period 개월 시점에 앉힘 (on-track).
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([45]), premium=np.array([120.0]),
                    term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        b, full=True).cashflows.inforce[0]
    em_close     = em_open + period
    prior_count  = 1000.0 * surv[em_open]   # 기초 잔존 (1,000 건 계약)
    count_close  = 1000.0 * surv[em_close]  # 기말 잔존
    ids = np.array(["H1"])
    mp = ModelPoints(
        issue_age=np.array([45]), premium=np.array([120.0]),
        term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
        count=np.array([count_close]), elapsed_months=np.array([em_close]),
        mp_id=ids, product=np.array(["HEALTH"]), calculation_methods=CM)
    state = InforceState(
        mp_id=ids,
        elapsed_months=np.array([em_close]),
        count=np.array([count_close]),
        prior_csm=np.array([prior_csm]),
        lock_in_rate=0.03,
        prior_count=np.array([prior_count]),
        prior_loss_component=(np.array([lc_open]) if lc_open else None),
        actual_premium=(None if actual_premium is None else np.array([actual_premium])),
        actual_investment_component=(None if actual_ic is None else np.array([actual_ic])),
    )
    return mp, state
```

## 손실요소 체계적 배분 — Sec. 50(a)-52

손실부담 계약 (직전 CSM = 0, 손실요소를 운반) 은 보장이 제공되면서 **손실요소를
당기로 환입** 합니다 (Sec. 50(a)/51). 엔진은 매기 비율 `r = 손실요소 /
(claims+expenses+RA pool)` 로 그 기의 claims/RA release 를 손실요소와 잔여
LRC 에 나눕니다. 손실요소로 간 몫 (`loss_component_amortised`) 은 손실의
환입이라 **보험수익에서 제외** 되고 (Sec. 49 / B123(b)), 손실요소는 보장기간
말에 0 으로 수렴합니다 (Sec. 52). 투자요소 (해지환급금 / 만기 / 연금) 는
claims 가 아니므로 pool 에서 빠집니다 (Sec. 51(a) / 85).

```python
# 손실요소 40,000 을 운반하는 손실부담 보유계약
mp, state = book(basis(), prior_csm=0.0, lc_open=40_000.0)
mv = fcf.gmm.settle(mp, state, basis(), period_months=12)

print("=== loss component (Sec. 50(a)-52) ===")
print(f"loss_component_opening   = {float(mv.loss_component_opening[0]):>12,.2f}")
print(f"loss_component_finance   = {float(mv.loss_component_finance[0]):>12,.2f}")    # 51(c) 이자부리
print(f"loss_component_amortised = {float(mv.loss_component_amortised[0]):>12,.2f}")  # 50(a) 환입
print(f"loss_component_closing   = {float(mv.loss_component_closing[0]):>12,.2f}")
```

출력:

```
=== loss component (Sec. 50(a)-52) ===
loss_component_opening   =    40,000.00
loss_component_finance   =       905.86
loss_component_amortised =    21,262.30
loss_component_closing   =    19,643.56
```

항등식은 `closing == opening + finance - amortised - reversed + recognised`
입니다 (`reversed` / `recognised` 는 Sec. 48/50(b) 미래서비스 채널, 여기선 0).
1 년 만에 손실요소가 40,000 -> 19,644 로 줄며, 그 환입 (21,262) 이 당기 P&L 에
손실 환입으로 잡힙니다. 직전엔 손실요소가 remeasurement 사이 정적이었습니다.

## 발생손해부채 (LIC) — `settlement_pattern`

청구가 발생 즉시 지급되지 않고 **정산 패턴** 으로 풀려나가면, 미지급 청구가
발생손해부채 (LIC, Liability for Incurred Claims) 로 쌓입니다 (Sec. 40(b)).
산출기초에 `settlement_pattern` (발생월·익월·… 지급 비율, 합 = 1) 을 주면
`gmm.settle` 이 LIC 블록을 냅니다. LIC 는 **이행현금흐름** 으로 측정합니다 —
미지급 runoff 의 할인 현재가치 + 위험조정 (Sec. 40(b)/42(c)/37). `claims_incurred`
와 `claims_paid` 는 명목 현금 그대로 (`claims_paid` 는 무할인 잔차) 두고, 할인·
위험조정은 잔액만 움직이며, `lic_finance` 가 그 차이 — 보험금융손익 (Sec. 42(c)
할인 unwind) + 할인·위험조정 측정효과 — 를 받는 잔차라 블록이 닫힙니다
(`lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid`).

```python
# 발생 청구를 60% 당월 / 30% 익월 / 10% 익익월로 지급하는 정산 패턴
sp = basis(settlement=np.array([0.6, 0.3, 0.1]))
mp, state = book(sp)
mv = fcf.gmm.settle(mp, state, sp, period_months=12)

print("=== liability for incurred claims (Sec. 40(b)/42/37) ===")
print(f"lic_opening     = {float(mv.lic_opening[0]):>13,.2f}")      # 할인 PV + RA
print(f"claims_incurred = {float(mv.claims_incurred[0]):>13,.2f}")  # 42(a) 명목
print(f"lic_finance     = {float(mv.lic_finance[0]):>13,.2f}")      # 42(c) + 측정효과
print(f"claims_paid     = {float(mv.claims_paid[0]):>13,.2f}")      # 패턴 runoff 명목
print(f"lic_closing     = {float(mv.lic_closing[0]):>13,.2f}")
```

출력:

```
=== liability for incurred claims (Sec. 40(b)/42/37) ===
lic_opening     =    506,684.39
claims_incurred = 11,003,258.44
lic_finance     =     -1,951.44
claims_paid     = 11,032,417.42
lic_closing     =    475,573.97
```

지급 시차가 있으니 기초·기말 모두 미지급 청구 (475,574 / 506,684) 가 남아
있습니다. 이 잔액은 **할인 현재가치 + 위험조정** 이라 무할인 명목 잔액보다 큽니다
(여기서는 RA 가 할인효과를 압도해 순증). `lic_finance` (-1,951) 는 명목 in/out
과 할인·위험조정 잔액의 차이를 받는 보험금융손익 라인입니다. LIC RA 는 할인
runoff 의 신뢰수준 마진을 위험군별로 쪼갠 것입니다 (cost-of-capital LIC runoff 는
향후 정교화). `settlement_pattern` 이 없으면 청구는 발생 즉시 지급되어 LIC 는
양쪽 0, `lic_finance` 0, `claims_paid == claims_incurred` 입니다.

## 보험료 경험조정 — Sec. B96(a) / B97(c)

이번 분기 실제 수취 보험료가 기대와 다르면, 그 차이를 미래서비스 (CSM,
Sec. B96(a)) 와 당기·과거서비스 (보험수익, Sec. B97(c)) 로 나눕니다. 분할은
회사 차원 판단이라 `premium_experience_future_fraction` (기본 0.0 = 전부 수익) 로
노출합니다. 실제 보험료는 결산 상태의 `actual_premium` 으로 줍니다.

```python
# 기대 (~1,313,113) 보다 36,887 더 받음 -> 미래/당기 절반씩 (frac=0.5)
mp, state = book(basis(), prior_csm=8_000.0, actual_premium=1_350_000.0)
mv = fcf.gmm.settle(mp, state, basis(), period_months=12,
                    premium_experience_future_fraction=0.5)  # 절반은 CSM, 절반은 수익

print("=== premium experience (Sec. B96(a)/B97(c)) ===")
print(f"csm_premium_experience     = {float(mv.csm_premium_experience[0]):>12,.2f}")      # B96(a) -> CSM
print(f"premium_experience_revenue = {float(mv.premium_experience_revenue[0]):>12,.2f}")  # B97(c) -> P&L
```

출력:

```
=== premium experience (Sec. B96(a)/B97(c)) ===
csm_premium_experience     =    18,443.48
premium_experience_revenue =    18,443.48
```

유리한 경험 (+36,887) 이 절반씩 (18,443) CSM 과 수익으로 갈렸습니다.
`csm_premium_experience` 는 BEL/RA 대응이 없는 새 미래서비스 변화라 세-항
교차항등식 밖에 있고, `premium_experience_revenue` 는 잔액 재귀에 들어가지
않는 P&L 메모입니다. `actual_premium` 이 없으면 두 라인 다 0 입니다. (해지가
미래 보험료를 줄이는 효과는 이미 건수 채널이 잡으므로, frac 의 기본값 0.0 이
이중계상을 막습니다.)

## 투자요소 경험조정 — Sec. B96(c)

투자요소 (해지환급금 / 만기 / 연금 — 보험사고와 무관하게 환급하는 금액,
Sec. 부록 A) 의 실제 지급이 기대와 다르면, 그 차이 **전부** 가 CSM 을 조정
합니다 (Sec. B96(c) 는 전부 미래서비스라 분할 없음). 투자요소는 보험수익을
건드리지 않습니다. 실제 지급액은 `actual_investment_component` 로 줍니다.

```python
# 해지환급금 (투자요소) 이 있는 계약. 기대 (~1,166,843) 보다 33,157 더 지급
b_surr = Basis(
    mortality_annual=death,                                # 탈퇴 (in-force decrement)
    lapse_annual=lambda s, ia, d: np.full(d.shape, 0.05),  # 해지 (평탄 5%)
    discount_annual=0.03,                                  # 할인율
    ra_confidence=0.75,                                    # 위험조정 신뢰수준
    mortality_cv=0.10,                                     # 사망 변동계수
    coverages=(CoverageRate("DEATH", death),),             # 사망 보장
    surrender_value_curve=np.full(36, 25_000.0),           # 정액 해지환급금
    surrender_value_basis="amount_per_policy",             # 환급금 기준
)
mp, state = book(b_surr, prior_csm=8_000.0, actual_ic=1_200_000.0)
mv = fcf.gmm.settle(mp, state, b_surr, period_months=12)

print("=== investment-component experience (Sec. B96(c)) ===")
print(f"csm_investment_experience  = {float(mv.csm_investment_experience[0]):>12,.2f}")
```

출력:

```
=== investment-component experience (Sec. B96(c)) ===
csm_investment_experience  =   -33,157.15
```

기대보다 33,157 더 지급 (더 많은 해지) 했으니 불리 -> CSM 이 그만큼 내려갑니다
(`csm_investment_experience = 기대 - 실제`). 덜 지급 (해지 적음, 유지) 했다면
양수로 CSM 이 올라갑니다. 기간내 지급은 기대 (k_exp) 척도로 돌므로 건수 채널과
이중계상하지 않습니다. `actual_investment_component` 가 없으면 0 입니다.

## VFA 대응 — `vfa.settle`

변액 (VFA) 도 같은 네 라인을 똑같이 냅니다. 한 가지 통합이 있습니다:
**VFA 의 투자요소는 계좌가치** 라, Sec. 51(a) 의 claims+expenses pool (투자요소
는 Sec. 85 로 제외) 이 VFA 에선 **보증초과액** (GMDB/GMAB 가 계좌가치를 넘는
보험 부분) + 사업비 로 자동으로 투자요소를 뺀 모양이 되고, Sec. B96(c) 의
투자요소는 출금 시 돌려주는 계좌가치 (`benefit_cf - guarantee_excess_cf`) 입
니다. 호출은 동일합니다:

```text
mv = fcf.vfa.settle(vfa_mp, vfa_state, vfa_basis, period_months=12,
                    premium_experience_future_fraction=0.0)
# mv.loss_component_finance / _amortised, mv.lic_*, mv.csm_premium_experience,
# mv.csm_investment_experience -- gmm.settle 과 같은 라인
```

## 함정 / 검증

- **손실요소 pool 은 claims+expenses 만** — 해지환급금 / 만기 / 연금 (투자
  요소) 은 빠집니다 (Sec. 51(a) / 85). 순수 보장 계약은 투자요소가 없어 배분
  숫자가 기존과 같습니다.
- **`settlement_pattern` 은 합이 정확히 1.** LIC 는 이행현금흐름 (할인 PV +
  위험조정, Sec. 40(b)/42(c)/37) 으로 측정하고, 명목 in/out 과 잔액의 차이는
  `lic_finance` 가 받습니다. LIC 할인은 평탄 월금리 (`discount_monthly`) 를 씁니다.
- **`premium_experience_future_fraction` 기본 0.0** — 해지구동 미래 보험료가
  이미 건수 채널에 잡히므로, 0 이 이중계상을 막습니다. 진짜 미래 coverage 의
  선납만 0 초과로.
- **부재 = byte-identical** — `prior_loss_component` 없으면 손실요소 라인 0,
  `settlement_pattern` 없으면 LIC 0, `actual_premium` / `actual_investment_
  component` 없으면 경험 라인 0. 기존 결산과 정확히 같습니다.

## 인접 레시피

- [9.1 결산 / 보유계약 평가](settlement) — settle 의 뼈대.
- [9.2 변동분해](movement) — 신계약 측정의 보고기간별 귀속.
- [9.4 결산팩 — 공시 명세서 조립](close-pack) — 정산표를 공시 명세서·엑셀 결산팩으로.

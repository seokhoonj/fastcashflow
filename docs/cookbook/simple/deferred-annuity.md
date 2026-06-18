# 2.6 거치연금 / 보증기간부 / 정기연금

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 거치 후 지급하는 **연금보험** 상품을, 계좌형(UL) 전환이 아니라 일반
  `annuity_payment` 의 **지급 일정** 으로 설정하는 법
- 세 가지 지급형태 필드: **지급개시 지연** (`annuity_start_months`),
  **정기연금** (`annuity_term_months`), **보증기간부** (`annuity_guarantee_months`)
- **보증기간부** 의 보증지급이 생존 무관 확정 지급이라, BEL 에는 들어가되
  **longevity RA 에서는 빠지는** 이유와 그 효과
- 번들 샘플 `samples("annuity")` 로 거치 보증기간부·거치 정기연금을 한 번에 보기
:::

연금의 **지급 단계** 는 두 길이 있습니다. 하나는 계좌형(UL)에서 적립금을
연금으로 **전환** 하는 길 (`annuitization_months`, [7.x 계좌형](../account/universal-life)),
다른 하나는 비계좌 전통형 연금에서 `annuity_payment` 를 **일정에 맞춰 지급** 하는
길입니다. 이 챕터는 후자 — 거치기간 동안 보험료로 BEL 을 적립하고, 정해진 시점부터
생존연금을 지급하는 한국 **연금보험** 의 기본 구조입니다.

## 상품 소개 — 세 가지 지급형태

`annuity_payment` 은 기본적으로 가입 시점부터 생존자에게 매월 지급하는 종신연금
입니다. 한국 연금보험은 여기에 세 가지 **지급 일정** 을 얹습니다:

- **거치 (지급개시 지연)** — 거치기간 동안은 지급하지 않고 보험료만 적립하다,
  거치 종료 시점부터 연금 개시. 거치연금의 핵심.
- **정기연금 (term-certain)** — 종신이 아니라 정해진 **횟수만** 지급하고 종료
  (예: 20년 확정연금).
- **보증기간부 (certain-and-life)** — 종신연금이되 첫 일정 기간(보증기간)은
  **생존 여부와 무관하게** 지급을 보증. 보증기간 중 사망해도 남은 보증 지급은
  수익자에게 가므로, 그 부분은 확정 지급입니다. 한국에서 가장 흔한 지급옵션.

## 모델링 매핑 — 세 개의 ModelPoints 필드

세 형태는 `annuity_payment` 에 붙는 세 정수 필드로 설정합니다 (모두 0 = 기본
종신·가입시점부터). 이 필드를 쓰는 계약은 full 투영 커널로 라우팅됩니다.

:::{list-table}
:header-rows: 1
:widths: 34 66

* - 필드
  - 의미
* - `annuity_start_months`
  - 지급개시까지의 개월 (거치). 0 = 가입시점부터.
* - `annuity_term_months`
  - 지급개시 후 지급 횟수 (정기연금). 0 = 무제한(종신).
* - `annuity_guarantee_months`
  - 보증기간 개월 (certain-and-life). 첫 이만큼은 생존 무관 확정 지급. 0 = 순수 종신.
:::

## 최소 작동 예제 — 번들 샘플

`samples("annuity")` 는 비계좌 거치연금 두 계약입니다 — 계약 0 은 10년 거치 후
20년 보증의 종신연금, 계약 1 은 5년 거치 후 20년 정기연금.

```python
import fastcashflow as fcf
import numpy as np

mp    = fcf.samples.model_points("annuity")   # 2 standalone deferred-annuity contracts
basis = fcf.samples.basis("annuity")
m     = fcf.gmm.measure(mp, basis)

print(f"{'contract':<26}{'BEL':>14}{'RA':>12}{'CSM':>12}")
labels = ["deferred guaranteed life", "deferred term-certain"]
for i, lab in enumerate(labels):
    print(f"{lab:<26}{m.bel[i]:>14,.0f}{m.ra[i]:>12,.0f}{m.csm[i]:>12,.0f}")

# the payout schedule -- when the income starts and (contract 1) stops
for i in (0, 1):
    acf = m.cashflows.annuity_cf[i]
    start = int(np.argmax(acf > 0))
    last  = int(np.max(np.nonzero(acf)))
    print(f"contract {i}: income months {start}..{last}")
```

출력:

```text
contract                             BEL          RA         CSM
deferred guaranteed life     -12,623,149     725,688  11,897,461
deferred term-certain         -3,842,447   1,158,047   2,684,400
contract 0: income months 120..599
contract 1: income months 60..299
```

계약 0 은 월 120(10년) 까지 지급이 0 이다 거치 종료 후 종신 지급, 계약 1 은
월 60(5년) 부터 240회(20년) 만 지급하고 월 300 에서 종료합니다. 음수 BEL 과
양수 CSM 은 보험료가 연금 급부를 충당하고 남는 수익성 계약이라는 뜻입니다.

## 보증기간부의 RA — 확정 지급은 longevity 위험이 없다

보증기간부의 핵심은 **보증지급은 확정** 이라는 점입니다. 보증기간 중에는 생존
여부와 무관하게 지급하므로, 그 부분은 **longevity 위험(연금수급자가 오래 사는
위험)을 지지 않습니다.** 따라서 보증지급은 BEL 에는 들어가되 longevity RA 에서는
빠져야 합니다.

같은 종신연금을 보증 없이 / 10년 보증으로 측정해 비교하면:

```python
import fastcashflow as fcf
import numpy as np

# one contract, two ways: a pure life annuity vs the same with a 10-year
# guarantee. The guaranteed payments are certain, so they carry no longevity
# risk -- the RA falls, the BEL does not.
def mp(guarantee):
    return fcf.ModelPoints.single(
        issue_age=60, premium=0.0, term_months=360, annuity_payment=1_000_000.0,
        annuity_guarantee_months=guarantee,
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})

basis = fcf.Basis(mortality_annual=0.02, lapse_annual=0.0, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.0, longevity_cv=0.15,
                  coverages=(fcf.CoverageRate("DEATH", 0.02),))

life = fcf.gmm.measure(mp(0), basis)
guar = fcf.gmm.measure(mp(120), basis)   # first 10 years guaranteed
print(f"{'':<22}{'BEL':>14}{'RA':>12}")
print(f"{'pure life annuity':<22}{life.bel[0]:>14,.0f}{life.ra[0]:>12,.0f}")
print(f"{'10y guaranteed':<22}{guar.bel[0]:>14,.0f}{guar.ra[0]:>12,.0f}")
print(f"RA falls by {life.ra[0] - guar.ra[0]:,.0f} (the certain payments bear no longevity risk)")
```

출력:

```text
                                 BEL          RA
pure life annuity        187,343,890  18,954,230
10y guaranteed           196,630,074   9,369,853
RA falls by 9,584,377 (the certain payments bear no longevity risk)
```

보증을 붙이면 **BEL 은 오르고**(보증지급이 확정이라 미래 현금흐름이 늘어남)
**RA 는 내려갑니다**(확정분이 longevity 위험기반에서 빠짐). 엔진이 두 효과를
자동으로 분리합니다 — BEL 의 `annuity_cf` 에는 보증지급이 들어가되, longevity
RA 의 `pv_survival` 에서는 보증지급의 PV 를 차감합니다.

## 함정 / 검증

- **full 경로 전용 (v1)** — 세 지급형태는 full 투영 커널에만 있습니다.
  `measure(full=False)` 로 측정하면 자동으로 full 로 라우팅되고, GPU / 할인곡선
  override 와는 결합할 수 없습니다 (명시적 에러).
- **UL 전환과 동시 사용 불가 (v1)** — `annuitization_months > 0` 인 계좌형 연금
  전환과 이 세 필드를 함께 설정하면 `NotImplementedError`. UL 전환형 보증연금은
  후속 단계입니다.
- **보증기간 <= 지급기간** — `annuity_guarantee_months` 는 `annuity_term_months`
  (설정 시) 를 넘을 수 없습니다. 지급개시월은 계약경계 안에 지급월이 최소 1회
  남도록 `contract_boundary_months` 보다 작아야 합니다.

## 인접 레시피

- [2.5 체증형 보험금 / 간병비 / 연금](escalating-benefits) — 연금액이 매년
  체증하는 `annuity_factor_annual` (이 세 형태와 함께 쓸 수 있음).
- [9.2 변동분해](../workflow/movement) — `estimate_at` 로 거치연금의 미래
  시점별 현재추정 (거치 중 / 지급 중) 을 보기.

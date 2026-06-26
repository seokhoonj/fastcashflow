# 8.3 수익성 분석 / profit-testing

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 측정 결과(BEL / RA / CSM)에서 **신계약가치 (NBV)** 와 **수익성 마진** 을 뽑는 법
  (`pricing.nbv`, `pricing.profit_margin`)
- **IFRS 17 이익 시그니처** (연도별 보험서비스결과의 emergence) 와, 그 현재가치가
  NBV 로 환원되는 관계 (`pricing.signature`)
- **전통형 (통계기초) profit-test** — commutation (Dx / Nx / Mx) 없이 투영으로
  **순보험료식 (NLP) 준비금** 을 만들고 (`pricing.statutory_reserve`), 준비금이
  **자기조달** 임을 확인한 뒤 보험료 로딩 / 금리 스프레드가 이익으로 떠오르는 것을
  보는 법 (`pricing.statutory_profit_signature`)
- 주주 현금흐름의 **IRR** 과 **회수기간** (`pricing.irr`, `pricing.break_even_year`)
:::

fastcashflow 는 부채 **측정** 엔진입니다. 이 챕터는 같은 측정 결과를 **pricing /
profit-testing** 관점 — "이 계약이 회사에 얼마의 이익을, 언제 안기는가" — 으로
다시 읽는 얇은 레이어 `fcf.pricing` 를 다룹니다. 새로 투영하는 것은 없습니다:
지표는 이미 산출된 BEL / RA / CSM, `report` 의 손익 emergence, 할인곡선을
조립해서 나옵니다.

두 가지 관점을 모두 봅니다:

- **IFRS 17 관점** — 신계약가치는 곧 음의 BEL (`CSM + RA`), 이익 시그니처는
  CSM / RA 의 기간별 인식.
- **전통형 (통계) 관점** — 순보험료식 준비금을 기준으로 한 전통적 이익원천 분석
  (보험료차 / 이자차). 한국 실무의 profit-test 가 이쪽입니다.

## 모델링 매핑 — 측정 결과의 어느 항이 어느 지표인가

:::{list-table}
:header-rows: 1
:widths: 30 70

* - 지표
  - 측정 결과에서의 정의
* - `nbv(m)`
  - 신계약가치 = `CSM + RA - loss_component` = `-BEL`. 발행시점에 계약이 풀어낼
    이익의 현재가치 (세전 · 필요자본비용 차감 전).
* - `profit_margin(m)`
  - `NBV / PV(보험료)` (PVNBP 마진). full 측정 필요 (보험료 현금흐름).
* - `signature(m, period_months)`
  - IFRS 17 이익 시그니처 — 기간별 보험서비스결과 (`report` 재사용).
* - `statutory_reserve(mp, stat)`
  - 통계기초 NLP 준비금 궤적 + 순보험료. 투영 PV 가 곧 전향적 준비금.
* - `statutory_profit_signature(mp, pricing, stat)`
  - 통계 이익 시그니처 — 준비금을 들고 실제 경험으로 이익을 emergence.
* - `irr(stream)` / `break_even_year(stream)`
  - 주주 현금흐름 (신계약비 strain + 이익)의 내부수익률 / 회수기간.
:::

## 최소 작동 예제 1 — NBV 와 마진 (IFRS 17)

`solve_premium` (기존 보험료 해) 로 **CSM 마진 10%** 가 되게 보험료를 풀고, 같은
계약의 NBV 와 `profit_margin` 을 봅니다.

```python
import numpy as np
from dataclasses import replace
import fastcashflow as fcf
from fastcashflow import pricing

mp0 = fcf.ModelPoints.single(
    issue_age=40, premium=0.0, term_months=120,
    benefits={"DEATH": 100_000_000.0},
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
basis = fcf.Basis(mortality_annual=0.01, lapse_annual=0.05, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.10,
                  coverages=(fcf.CoverageRate("DEATH", 0.01),))

gross = pricing.solve_premium(mp0, basis, margin=0.10)[0]   # 10% CSM margin
mp = replace(mp0, premium=np.full(1, gross))
m  = fcf.gmm.measure(mp, basis)

print(f"priced premium = {gross:,.2f}")
print(f"NBV            = {pricing.nbv(m)[0]:,.0f}")
print(f"  = CSM {m.csm[0]:,.0f} + RA {m.ra[0]:,.0f} - loss {m.loss_component[0]:,.0f}")
print(f"profit margin  = {pricing.profit_margin(m)[0]:.4f}")
```

출력:

```text
priced premium = 99,171.57
NBV            = 1,230,855
  = CSM 784,642 + RA 446,213 - loss 0
profit margin  = 0.1569
```

`solve_premium` 의 마진 목표는 **CSM 기준** (`CSM / PV(보험료) = 0.10`) 인데,
`profit_margin` 은 **NBV 기준** (`(CSM + RA) / PV(보험료)`) 이라 15.69% 로 더
넓습니다. 차이 5.69%p 가 곧 **RA** — 위험이 풀리면서 함께 이익으로 인식될
부분입니다. NBV 는 CSM 과 RA 를 모두 발행시점 이익으로 보는 더 넓은 신계약가치
지표입니다.

## 최소 작동 예제 2 — IFRS 17 이익 시그니처

같은 계약의 **연도별 이익 emergence** 입니다. 시그니처를 발행시점으로 할인하면
NBV 로 환원됩니다.

```python
sig = pricing.signature(m, period_months=12)
print(f"{'year':>4}{'profit':>14}")
for yr, p in zip(sig.month_end // 12, sig.profit):
    print(f"{yr:>4}{p:>14,.0f}")
print(f"PV @3% = {sig.present_value(0.03):,.0f}   NBV total = {pricing.nbv(m).sum():,.0f}")
```

출력:

```text
year        profit
   1       169,263
   2       162,111
   3       155,293
   4       148,792
   5       142,592
   6       136,679
   7       131,036
   8       125,652
   9       120,512
  10       115,605
PV @3% = 1,231,100   NBV total = 1,230,855
```

이익은 CSM / RA 가 해마다 풀리며 인식되어 **매년 감소** 합니다 (인포스가 줄어드니까).
시그니처의 현재가치 1,231,100 이 NBV 1,230,855 와 일치 (0.02% 이내) — 시그니처는
같은 신계약가치를 연도별로 펼친 표현일 뿐입니다.

## 최소 작동 예제 3 — 전통형 (통계) profit-test

전통형은 **순보험료식 (NLP) 준비금** 을 기준으로 이익을 봅니다. 양로보험
(만기 환급) 으로 통계기초 준비금을 만들고, 자기조달 → 로딩 / 이자차 emergence 를
확인합니다.

```python
endow = fcf.ModelPoints.single(
    issue_age=40, premium=0.0, term_months=120,
    benefits={"DEATH": 100_000_000.0}, maturity_benefit=100_000_000.0,
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
stat = fcf.Basis(mortality_annual=0.01, lapse_annual=0.0, discount_annual=0.025,
                 ra_confidence=0.75, mortality_cv=0.0,
                 coverages=(fcf.CoverageRate("DEATH", 0.01),))

V, net = pricing.statutory_reserve(endow, stat)
print(f"net premium = {net[0]:,.0f}")
print(f"V[0] = {V[0, 0]:,.0f}   V[60] = {V[0, 60]:,.0f}   V[120] = {V[0, 120]:,.0f}")
```

출력:

```text
net premium = 779,560
V[0] = 0   V[60] = 43,429,832   V[120] = 90,438,208
```

준비금은 발행시점 0 에서 만기 환급액 쪽으로 쌓입니다 — **commutation 함수 없이**
투영의 후방 PV 가 곧 전향적 준비금입니다 (`Dx / Nx / Mx` 는 컴퓨터 이전의 같은
PV 단축법일 뿐).

순보험료를 그대로 내는 (통계기초 = pricing 기초) 계약은 준비금이 **자기조달** 이라
이익이 0 입니다. 실제로는 보험료에 **로딩** 을 얹으므로 그 차이가 이익으로 떠오릅니다:

```python
from dataclasses import replace

# self-financing: net premium on the statutory basis -> profit ~ 0
selfp = pricing.statutory_profit_signature(replace(endow, premium=net), stat, stat)
# 10% loading: the loading emerges as profit
psig = pricing.statutory_profit_signature(replace(endow, premium=net * 1.10), stat, stat)

print(f"self-financing |max profit| = {np.max(np.abs(selfp.profit)):.2e}")
print(f"loading profit total        = {psig.total:,.0f}")
print(f"{'year':>4}{'profit':>12}")
for yr, p in zip(psig.month_end[:5] // 12, psig.profit[:5]):
    print(f"{yr:>4}{p:>12,.0f}")
```

출력:

```text
self-financing |max profit| = 4.47e-08
loading profit total        = 8,922,063
year      profit
   1     933,095
   2     923,764
   3     914,527
   4     905,381
   5     896,328
```

순보험료식 자기평가 이익이 부동소수점 0 (4e-08) — 준비금과 emergence 점화식이
정확히 일치한다는 뜻이고, 이것이 이 전통형 엔진의 핵심 검산 앵커입니다. 10% 로딩을
얹으면 그 로딩이 매년 이익으로 인식됩니다 (총 8,922,063).

## IRR 과 회수기간

이익 시그니처 앞에 **신계약비 strain** (발행시점 음의 현금흐름) 을 붙이면 주주
현금흐름이 부호를 바꿔 IRR / 회수기간이 의미를 가집니다.

```python
acq = 4_000_000.0                              # day-0 new-business strain
stream = np.concatenate([[-acq], psig.profit])  # year 0 strain, then yearly profit
print(f"IRR             = {pricing.irr(stream):.4f}")
print(f"break-even year = {pricing.break_even_year(stream)}")
```

출력:

```text
IRR             = 0.1845
break-even year = 6
```

연 18.45% 의 내부수익률입니다. `break_even_year` 는 현금흐름 배열에서 누적이
양전환하는 **1-기반 위치** 를 돌려줍니다 — 0 번 항목이 발행시점 strain 이므로
6 은 **5 년차에 회수** 됨을 뜻합니다.

## 변형

- **마진 / IRR / NBV 목표로 보험료 풀기** — `solve_premium(..., margin=)` /
  `csm=` 로 목표를 정해 보험료를 역산하고, 이 챕터의 지표로 확인하는 왕복.
- **금리 스프레드** — `statutory_profit_signature(..., earned_rate=0.04)` 로
  평가이율(2.5%) 위로 운용수익(4%)을 가정하면 이자차가 추가 이익으로 emergence.
- **월 단위 시그니처** — `signature(m, period_months=1)` / 통계 시그니처도 동일
  인자로 월별 emergence.
- **포트폴리오** — 모든 지표가 모델포인트 축으로 벡터화 (`nbv` 는 `(n_mp,)`),
  시그니처는 포트폴리오 전체 합산.

## 함정 / 검증

- **`profit_margin` 은 full 측정 필요** — 보험료 현금흐름을 쓰므로
  `measure(full=False)` 면 명시적 에러. `nbv` 는 headline 만 쓰므로 fast path 에서도
  됩니다.
- **CSM 마진 != NBV 마진** — `solve_premium` 의 `margin` 은 CSM 기준,
  `profit_margin` 은 NBV(CSM+RA) 기준. RA 만큼 후자가 넓습니다 (예제 1).
- **IRR 은 부호변화 필요** — IFRS 17 시그니처는 전부 양수라 IRR 이 없습니다
  (`ValueError`). 신계약비 strain 을 붙여 부호를 바꿔야 의미가 생깁니다.
- **통계 profit-test v1 가정** — pricing 기초와 통계기초가 **decrement (사망 / 해지)
  를 공유** 하고 평가이율만 다르다고 가정합니다. 서로 다른 decrement 로 재설정한
  준비금은 후속 단계입니다. 라우터(dict basis) 가 아니라 단일 `Basis` 를 넘기세요.
- **자기조달 앵커** — 통계기초 = pricing 기초 + 순보험료면 이익이 0 (부동소수점)
  이어야 합니다. 0 이 아니면 준비금 / emergence 점화식의 부호 · 타이밍이 어긋난 것.

## 인접 레시피

- [8.2 검증 패턴](validation) — 손계산으로 한 계약의 측정 경로를 추적. 이 챕터의
  지표도 같은 손계산 앵커 위에 섭니다.
- [2.6 거치연금 / 보증기간부 / 정기연금](../simple/deferred-annuity) — 연금
  상품의 측정 (전통형 준비금이 자연스럽게 쌓이는 자리).
- [9.2 변동분해](movement) — 이익 시그니처를 보고기간별 BEL / CSM 움직임으로
  더 잘게 귀속.

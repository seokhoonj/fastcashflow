# 2.5 체증형 보험금 / 간병비 / 연금

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 보험금이 시간에 따라 **올라가는** 체증형 상품을 두 가지 축으로 설정하는 법:
  **계단식 (step)** 과 **매년 % (escalation)**
- `coverages` 파일의 네 컬럼 (`step_month` / `step_factor` /
  `escalation_annual` / `escalation_cap`) 로 담보 금액을 시간축에서 키우는 법
- 체증형 **연금** 은 담보 컬럼이 아니라 `Basis` 의 `annuity_factor_annual`
  로 설정하는 법
- 체증이 **감액 (reduction) 의 반대 방향** 일 뿐 같은 메커니즘이라는 점
:::

체증은 **상태가 아니라 담보별 룰** 입니다 (한 담보의 시간축 위 금액 배수
곡선) — 2.3 의 면책 / 감액과 같은 자리에 붙는, **방향만 반대인** 룰입니다.
감액 (reduction) 이 초기 지급을 **누른다면**, 체증 (escalation) 은 후기
지급을 **키웁니다**. 둘은 한 담보에 공존할 수 있습니다 (초기 감액 + 후기
체증).

## 상품 소개 — 세 가지 체증 구조

한국 시장의 체증형 상품은 보험금이 오르는 **모양** 에 따라 갈립니다:

- **체증형 종신 / CI** — 사망·진단 보험금이 가입 일정 기간 후부터 **매년
  일정 %** 씩 오릅니다 (예: 5년 후부터 매년 10%, 최대 2배까지). 인플레이션
  헤지 성격. 보통 **상한 (cap)** 을 둡니다.
- **체증형 간병비 / 장기요양** — 간병 정액이 **특정 경과 시점에 계단식**
  으로 점프합니다 (예: 가입 20년 후 월 간병비 2배). 매년 % 가 아니라
  **경과월 기준 step** 이 핵심.
- **체증형 연금** — 연금액이 전년 대비 **매년 일정 %** 체증합니다 (예:
  매년 5%). 물가연동 / 체증연금 성격.

CI (=Critical Illness=중대질병) 진단 보험금처럼 한 번 지급하고 끝나는
담보든, 간병비처럼 매월 반복 지급하는 담보든, **금액 배수** 를 시간축에
설정하는 방식은 같습니다.

## 모델링 매핑 — coverages 파일의 네 컬럼

체증은 `coverages` 파일의 담보 행에 네 컬럼으로 붙습니다. 두 축
(step / escalation) 은 독립이라 한 담보에 함께 쓸 수도, 따로 쓸 수도
있습니다. 모두 **가입 시점 (t=0) 기준** 입니다.

:::{list-table}
:header-rows: 1
:widths: 24 16 60

* - 컬럼
  - 단위
  - 적용
* - `step_month`
  - 정수 (개월)
  - `t >= step_month` 부터 금액에 `step_factor` 를 곱함 (계단식 점프).
    `0` (기본) 이면 계단 없음
* - `step_factor`
  - 실수
  - 계단 이후 배수 (예: `2.0` = 2배). 기본 `1.0` (변화 없음)
* - `escalation_annual`
  - 실수 (연 %)
  - 매년 복리로 금액을 키우는 비율 (예: `0.10` = 매년 10%).
    경과연수 `d` 에서 배수 `(1 + escalation_annual) ** d`. 기본 `0.0`
* - `escalation_cap`
  - 실수 (배수)
  - 매년 % 체증의 **상한 배수** (예: `2.0` = 최대 2배에서 멈춤).
    `0.0` (기본) 이면 상한 없음
:::

월 `t` (경과연수 `d = t // 12`) 의 금액 배수:

| 축 | 배수 |
|---|---|
| 계단 (step) | `t >= step_month` 이면 `step_factor`, 아니면 `1.0` |
| 매년 % (escalation) | `min((1 + escalation_annual) ** d, escalation_cap)` (cap 이 `0` 이면 상한 없음) |

두 축을 함께 쓰면 배수는 둘의 **곱** 입니다. 2.3 의 감액 배수와도 곱해져,
한 담보의 최종 지급 금액은 `amount x 감액배수 x 체증배수` 가 됩니다.

## 한 계약 — 손계산과 엔진 (체증형 간병비, 계단식)

가장 또렷한 toy 로 시작합니다. 간병비 담보 (월 정액) 하나에 **12개월 후
2배** 계단을 겁니다 (실무의 "20년 후 2배" 를 짧게 압축한 형태).

:::{admonition} 예제 설정
:class: note

- 보험기간 24개월, 사망 / 해지 없음 (배수에 집중), 할인 0
- 간병 월정액 1,000 (MORBIDITY -- 매월 반복 지급)
- 12개월 후 2배 (`step_month=12`, `step_factor=2.0`)
:::

```python
import numpy as np
import polars as pl
from pathlib import Path
import fastcashflow as fcf

# 계리적 가정 -- 감쇠 없는 toy (배수 효과에 집중)
care_rate = 1.0   # 간병 발생률 1.0 (정액 매월 지급)

# 산출기초 (Basis)
basis = fcf.Basis(
    mortality_annual = 0.0,   # 보유계약 사망률 0
    lapse_annual     = 0.0,   # 해지율 0
    discount_annual  = 0.0,   # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,  # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,   # 사망률 변동계수 0
    morbidity_cv     = 0.0,   # 발생률 변동계수 0
    coverages        = (
        fcf.CoverageRate("CARE", care_rate),  # 간병 담보 (청구 rate = care_rate)
    ),
)

# 입력 파일 -- coverages 에 계단식 체증 두 컬럼
Path("samples").mkdir(exist_ok=True)
pl.DataFrame({
    "mp_id":       ["P001"],   # 계약 식별자
    "issue_age":   [50],       # 가입연령 50세
    "term_months": [24],       # 보험기간 24개월
    "premium":     [0],        # 월납 보험료 0
}).write_csv("samples/policies.csv")

pl.DataFrame({
    "mp_id":       ["P001"],    # 어느 계약의 담보인지
    "coverage":    ["CARE"],    # 담보 코드
    "amount":      [1000],      # 간병 월정액 1,000
    "step_month":  [12],        # 12개월 후 계단
    "step_factor": [2.0],       # 계단 이후 2배
}).write_csv("samples/coverages.csv")

mp = fcf.read_model_points(
    "samples/policies.csv",                                # 계약 spec 파일
    coverages="samples/coverages.csv",                              # 담보 + 체증 룰
    calculation_methods={"CARE": fcf.CalculationMethod.MORBIDITY},
)

m = fcf.gmm.measure(mp, basis)
print(f"morbidity_cf[11], [12] = {m.cashflows.morbidity_cf[0, 11]}, "
      f"{m.cashflows.morbidity_cf[0, 12]}")        # 계단 직전 / 직후
print(f"BEL                    = {m.bel[0]:.2f}")  # 최선추정부채
```

출력:

```
morbidity_cf[11], [12] = 1000.0, 2000.0
BEL                    = 36000.00
```

손계산. 감쇠가 없으니 in-force 는 1 로 유지되고, 금액만 계단을 따릅니다:

| 구간 | 월수 | 월 지급 (배수 x 1,000) | 소계 |
|---|---|---|---|
| `t < 12` (계단 전) | 12 | 1,000 |  12,000 |
| `t >= 12` (계단 후, 2배) | 12 | 2,000 |  24,000 |

- BEL = 12,000 + 24,000 = **36,000** (할인 0, 보험료 0)
- 계단이 없으면 매월 1,000, BEL = 24,000. 12개월 후 2배가 BEL을
  **24,000 -> 36,000** 으로 올립니다 (보험사 지급이 늘어나니까).

:::{admonition} 보험금 체증과 보험료 — 정액(평준)이 더 높아질 뿐
:class: note

보험금이 오르면 부채 (BEL) 가 커지니, 그 큰 보장을 메우려면 **보험료를 더 높게**
책정해야 합니다. 다만 보험료는 여전히 **정액 (평준)** — 시간축에서 일정하되
수준만 높습니다 (체증형 종신·정기의 표준 구조). 이 더 높은 정액 보험료를 체증
보험금에 맞춰 **푸는** 도구가 `fcf.solve_premium` 입니다. 이 챕터는 주어진 체증
구조의 **측정** 에 집중합니다.

**보험료 자체가 시간축에서 오르는** 체증형 보험료는 이와 **별개의 상품 구조**
입니다 (예: 갱신형 재산정) — 보험금 체증의 자동 결과가 아니라 독립적인 설계이고,
`Basis` 의 `premium_factor_annual` (경과연수 → 보험료 배수) 로 표현합니다.
:::

## 결과 읽기

`morbidity_cf` 가 체증의 계단을 그대로 보여줍니다 — 12개월까지 1,000,
그 이후 2,000. 이 곡선의 현재가치 합이 BEL이고, 체증이 이를수록 / 클수록
BEL이 커집니다. 같은 곡선을 [검증 패턴](../workflow/validation) 의
`gmm.trace` 로 담보별 배수가 어느 월에 바뀌는지 한 줄씩 확인할 수 있습니다.

## 변형 1 — 매년 % 체증 + 상한 (체증형 종신 / CI)

종신·CI 보험금은 보통 **매년 % + 상한** 입니다. 같은 간병 toy 를 길게
늘이고, 매년 15% 체증에 **최대 2배** 상한을 겁니다.

```python
pl.DataFrame({
    "mp_id":             ["P001"],
    "coverage":          ["CARE"],
    "amount":            [1000],
    "escalation_annual": [0.15],   # 매년 15% 복리 체증
    "escalation_cap":    [2.0],    # 최대 2배에서 멈춤
}).write_csv("samples/coverages.csv")

# 상한이 무는 것을 보려면 보험기간을 늘립니다 (12년)
pl.DataFrame({
    "mp_id":       ["P001"],
    "issue_age":   [50],
    "term_months": [144],          # 12년
    "premium":     [0],
}).write_csv("samples/policies.csv")

mp = fcf.read_model_points(
    "samples/policies.csv",
    coverages="samples/coverages.csv",
    calculation_methods={"CARE": fcf.CalculationMethod.MORBIDITY},
)
cf = fcf.gmm.measure(mp, basis).cashflows.morbidity_cf[0]
print(f"year 0, 1, 2 = {cf[0]:.1f}, {cf[12]:.1f}, {cf[24]:.1f}")  # x1, x1.15, x1.15^2
print(f"year 5, 6    = {cf[60]:.1f}, {cf[72]:.1f}")               # 상한에 닿는 자리
```

출력:

```
year 0, 1, 2 = 1000.0, 1150.0, 1322.5
year 5, 6    = 2000.0, 2000.0
```

손계산. 배수는 `1.15 ** d` 이되 2.0 에서 멈춥니다. `1.15^4 = 1.749`,
`1.15^5 = 2.011 > 2.0` 이므로 **5년차부터 2,000 에 고정** 됩니다. 매년 %
체증은 이렇게 복리로 오르다 상한에 닿으면 평평해집니다.

## 변형 2 — 체증형 연금 (연금액 매년 % 증가)

연금은 담보 컬럼이 아니라 **`Basis` 의 `annuity_factor_annual`** 로
체증합니다 — 연금액 (`annuity_payment`) 에 곱해지는 경과연수 함수입니다.
매년 5% 체증을 설정합니다.

```python
# 체증 배수 함수 -- 5-인자 rate callable (sex, issue_age, duration,
# issue_class, elapsed). 경과연수 d 에서 1.05 ** d
annuity_factor = lambda s, a, d, ic, el: 1.05 ** d

mp_ann = fcf.ModelPoints(
    issue_age                = np.array([60]),                          # 가입연령 60세
    premium                  = np.array([0.0]),                         # 보험료 0
    term_months              = np.array([36]),                          # 3년 (연 3회 지급)
    annuity_payment          = np.array([100.0]),                       # 연금 연액 100
    annuity_frequency_months = np.array([12]),                          # 매년 지급
    benefits                 = {"ANN": np.array([0.0])},                # 사망보험금 없음
    calculation_methods      = {"ANN": fcf.CalculationMethod.ANNUITY},
)

basis_ann = fcf.Basis(
    mortality_annual      = 0.0,                              # 감쇠 0
    lapse_annual          = 0.0,                              # 해지 0
    discount_annual       = 0.0,                              # 할인 0
    ra_confidence         = 0.75,                             # 위험조정 신뢰수준 75%
    mortality_cv          = 0.0,                              # 변동계수 0
    coverages             = (fcf.CoverageRate("ANN", 0.0),),
    annuity_factor_annual = annuity_factor,                   # 체증형 연금 배수
)

m = fcf.gmm.measure(mp_ann, basis_ann, full=True)
acf = m.cashflows.annuity_cf[0]
print(f"annuity year 0, 1, 2 = {acf[0]}, {acf[12]}, {acf[24]:.4f}")  # 100, 105, 110.25
print(f"BEL                  = {m.bel[0]:.2f}")                       # 세 지급의 합
```

출력:

```
annuity year 0, 1, 2 = 100.0, 105.0, 110.2500
BEL                  = 315.25
```

손계산. 연금 배수는 `1.05 ** d` 라 100 -> 105 -> 110.25 로 매년 5% 복리
체증합니다. 연금은 **유출** 이라 (보험사가 지급) 체증이 BEL을 키웁니다:
BEL = 100 + 105 + 110.25 = **315.25** (할인 0).

:::{admonition} 담보 체증 vs 연금 체증 -- 왜 설정 위치가 다른가
:class: note

진단 / 간병 같은 **담보** 의 체증은 `coverages` 파일의 담보별 컬럼
(`escalation_annual` 등) 으로 — 담보마다 다른 체증을 줄 수 있어야 하니까.
**연금** 의 체증은 계약의 `annuity_payment` 한 금액에 지정하므로 `Basis` 의
`annuity_factor_annual` 한 곳에 둡니다. 둘 다 "경과연수 -> 배수" 곡선으로
같은 모양이지만, 설정 위치는 체증의 **단위** (담보별 vs 계약 연금액) 를
따릅니다.
:::

## 함정

### 함정 1 — `escalation_cap` 을 배수가 아닌 % 로 줌

`escalation_cap` 은 **배수** (예: `2.0` = 2배) 이지 % 가 아닙니다.
`escalation_cap=0.10` 은 "최대 0.1배" — 즉 금액을 10분의 1로 줄이는
상한이라 의도와 정반대입니다. 상한을 안 두려면 `0.0` (기본) 으로.

### 함정 2 — 체증을 `annual_to_monthly` 로 월 환산하는 줄 착각

체증 배수는 **금액 배수** 이지 발생률이 아닙니다. `1.0` 을 넘을 수 있고
(2배, 3배), 발생률처럼 `annual_to_monthly` 로 월 환산하지 않습니다.
`escalation_annual` 은 **연 복리 배수** 로 그대로 `(1 + r) ** d` 에
들어갑니다.

### 함정 3 — 계단식 간병비를 fast path 로 측정

체증 / 계단 (`escalation_annual` / `step_month`) 은 현재 **full path
(`measure(full=True)`)** 에서만 적용됩니다. `full=False` (융합 fast
kernel) 로 체증형 담보를 측정하면 엔진이 `NotImplementedError` 로
거부합니다 — 조용히 체증을 빠뜨려 BEL을 틀리게 내느니 명시적으로 막는
편이 안전하기 때문. 대량 포트폴리오의 체증형 담보는 `full=True` 로.

## 인접 레시피

- [2.3 다종 진단 + 면책 / 감액](diagnosis-rules) — 같은 담보별 룰 축의
  **반대 방향** (감액). 체증과 한 담보에 공존 가능.
- [2.4 갱신형](renewable) — 계약의 경계 / 갱신 구조. **보험료 자체가 시간축에서
  오르는** 체증형 보험료 (`Basis` 의 `premium_factor_annual`) 는 보험금 체증과
  **별개의 상품 구조** — 갱신형 재산정이 그 예입니다.
- [4.3 간병 / 치매 (장기요양)](../semi-markov/long-term-care) — 상태 추적이
  필요한 간병 (지급 한도 / 상태 이탈). 정액 체증과 직교하며 함께 쓸 수
  있음.
- [검증 패턴](../workflow/validation) — `gmm.trace` 로 담보별 체증 배수가
  어느 월에 바뀌는지 한 줄씩 확인.

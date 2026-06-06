# 4.3 간병 / 치매 (LTC, Semi-Markov)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **간병 / 치매 (LTC)** 보장을 Semi-Markov 로 — 간병 상태에 진입하면 **진단금
  일시금** 한 번 + **월정액** 을 매월 받되, 지급은 **보증한도** 까지만
- `State.benefit_max_months` — 월정액의 **sojourn 한도** (예: 36 회 보증 후
  지급 중단, 계약은 유지)
- `State.mortality_rate` — **간병 상태의 상승 사망률** (간병 진입자는 사망률이
  높음); in-force 가 그만큼 빨리 소멸
- 한 상태가 **일시금 (`disability_benefit`) + 월정액 (`disability_income`)** 을
  함께 다루는 구조 — [4.1](reincidence) 의 일시금, [4.2](disability-income) 의
  월정액을 한 챕터에서 결합
```

## 상품 소개 — 간병 / 치매

**간병 / 장기요양 (LTC) · 치매보험** 은 피보험자가 장기요양 등급을 받거나 치매로
진단되면 보장이 시작됩니다. 한국 상품의 전형적 지급 구조는 두 갈래입니다:

- **진단 일시금** — 간병 / 치매 진단 시 한 번 지급 (예: 중증 2,000 만).
- **월정액** — 간병 상태가 지속되는 동안 매월 지급 (예: 월 100 만), 단
  **보증한도** (예: 36 회 보증 후 종신, 120 회 확정) 까지.

월정액은 무한정 나가지 않습니다 — **보증 개월수만큼만** 지급하고, 그 뒤에는
계약은 살아 있되 (사망 보장 등) 월정액만 멈춥니다. 이 "상태에 머문 개월수
(sojourn) 에 따른 지급 한도" 가 Semi-Markov 가 아니면 표현되지 않는 부분이고,
`State.benefit_max_months` 가 그 자리입니다.

또한 간병 상태 진입자는 **사망률이 일반보다 훨씬 높습니다** — in-force 가 빨리
소멸하므로 월정액 부채도 그만큼 작아집니다. 이 상태별 사망률을
`State.mortality_rate` 로 줍니다.

## 모델링 매핑 — active / care (2-state)

```{list-table}
:header-rows: 1
:widths: 42 58

* - 선언
  - 역할
* - `State("active", premium=True, ...)`
  - 건강 상태. 보험료 납입. 간병 발생 시 care 로.
* - `Transition("waiver_incidence", to="care", lump_sum=True)`
  - 간병 발생 — active -> care. `lump_sum=True` 가 진입 시
    `disability_benefit` (진단금) 를 한 번 지급.
* - `State("care", benefit=True, benefit_max_months=36, mortality_rate="dth_care", duration_max=60)`
  - 간병 상태. `benefit=True` 가 매월 `disability_income` 지급,
    `benefit_max_months=36` 이 **36 회까지만** 지급. `mortality_rate="dth_care"`
    가 이 상태의 **상승 사망률** 을 라우팅. `duration_max` 가 sojourn 코호트 추적.
* - `Basis.state_mortality_annual={"dth_care": fn}`
  - `dth_care` 라는 이름의 사망률 함수. 이름이 없으면 전역 `mortality_annual`
    로 fallback.
```

```{admonition} duration_max > benefit_max_months (strict)
:class: warning

`benefit_max_months > 0` 이면 `duration_max` 가 그보다 **커야** 합니다 (가드
코호트 1 개 이상). 같으면 마지막 흡수 코호트에 한도 넘은 계약이 고여 영원히
지급되는 off-by-one 이 생기므로, 생성자가 명시적으로 거부합니다.
```

두 상태를 그림으로 (care 에 머무는 동안 월정액 지급, 한도까지; 상승 사망률):

```{mermaid}
stateDiagram-v2
    [*] --> active
    active --> care: waiver_incidence (진단금 lump)
    active --> [*]: mortality / lapse
    care --> [*]: mortality (상승률)
```

## 율 — 간병 발생률 (long-form 표)

간병 발생률은 고령에서 급증합니다 (장기요양 등급 인정률). 견본 위험률표와 같은
연령별 long-form 표로 깔고 룩업합니다:

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel

# 계리적 가정 -- 간병 발생률 연령표 (long-form 룩업; 실무는 Excel)
ages  = np.array([    50,     60,     70,     80,     90])
ltc_m = np.array([0.0015, 0.0050, 0.0160, 0.0450, 0.1000])   # 남
ltc_f = np.array([0.0018, 0.0060, 0.0190, 0.0520, 0.1100])   # 여

def ltc_incidence(s, a, d):                          # 연령표 룩업 (VLOOKUP 식)
    a = np.asarray(a, dtype=float)
    return np.where(np.asarray(s) == 1,
                    np.interp(a, ages, ltc_f), np.interp(a, ages, ltc_m))

print("발생률 50/60/70/80 (남) :",
      [round(float(ltc_incidence(np.array([0]), np.array([a]), 0)[0]), 4)
       for a in (50, 60, 70, 80)])
```

```text
발생률 50/60/70/80 (남) : [0.0015, 0.005, 0.016, 0.045]
```

## 최소 작동 예제 — 진단금 + 보증한도 월정액 + 상승 사망률

60 세 가입, 90 세까지. 간병 진단 시 진단금 2,000 만 (일시금) + 월정액 100 만
(36 회 보증), 간병 상태 사망률은 연 20%:

```python
# 사망 / 해지 / 간병 상태 사망률
active_mort = lambda s, a, d: np.full(np.shape(a), 0.01)   # active 사망 (연 1% toy)
care_mort   = lambda s, a, d: np.full(np.shape(a), 0.20)   # 간병 상태 상승 사망률 (연 20%)
lapse_fn    = lambda s, a, d: np.full(np.shape(d), 0.03)   # 해지 연 3%

# 상태 모델 -- active -> care; care 는 진단금(lump) + 월정액(36 회) + 상승 사망률
model = StateModel(states=(
    State("active", premium=True, transitions=(
        Transition("mortality"),
        Transition("lapse"),
        Transition("waiver_incidence", to="care", lump_sum=True))),  # 진단금 lump
    State("care", benefit=True, duration_max=60, benefit_max_months=36,
          mortality_rate="dth_care", transitions=(
          Transition("mortality"),)),                                 # 상승 사망률
), seating=(0, 1))

# 산출기초
basis = fcf.Basis(
    mortality_annual        = active_mort,                 # active 사망 decrement
    lapse_annual            = lapse_fn,                     # 해지율
    waiver_incidence_annual = ltc_incidence,               # 간병 발생률 (위 표)
    state_mortality_annual  = {"dth_care": care_mort},     # 간병 상태 사망률
    discount_annual         = 0.03,                        # 할인율 3%
    ra_confidence           = 0.75,                        # 위험조정 신뢰수준
    mortality_cv            = 0.10,                        # 사망률 변동계수
    disability_cv           = 0.20,                        # 간병 발생 변동계수
    state_model             = model,
    coverages               = (fcf.CoverageRate("DEATH", active_mort),))

# 모델 포인트 -- 진단금 2,000 만, 월정액 100 만
mp = fcf.ModelPoints(
    issue_age         = np.array([60], dtype=np.int64),    # 60 세 가입
    benefits          = {0: np.array([0.0])},              # 사망보험금 0 (간병에 집중)
    premium           = np.array([90_000.0]),              # 월 보험료 9 만
    term_months       = np.array([360], dtype=np.int64),   # 90 세까지 (30 년)
    disability_benefit = np.array([20_000_000.0]),         # 진단금 2,000 만 (lump)
    disability_income  = np.array([1_000_000.0]),          # 월정액 100 만
    state             = np.array([0], dtype=np.int64),     # active 가입 (신계약)
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH})

m = fcf.gmm.measure(mp, basis)
print(f"BEL  : {m.bel[0]:>14,.0f}")
print(f"RA   : {m.ra[0]:>14,.0f}")
print(f"CSM  : {m.csm[0]:>14,.0f}")
print(f"Loss : {m.loss_component[0]:>14,.0f}")
```

```text
BEL  :    -10,221,701
RA   :        362,964
CSM  :      9,858,737
Loss :              0
```

진단금 일시금, 보증한도 월정액, 간병 상태의 상승 사망률이 한 모델에서 함께
작동합니다. 이익이 나는 계약 (`CSM > 0`) 입니다.

## 손계산 검증 — 보증한도가 정확히 끊는가

`benefit_max_months` 가 의도대로 끊는지 작은 toy 로 확인합니다. **간병 상태에
자리 지정** 하고 (`state=1`), 감쇠 없이 (사망 / 해지 0, 할인 0) 굴리면, 월정액은
보증 개월수만큼만 나와야 합니다:

```python
zero = lambda s, a, d: np.full(np.shape(a), 0.0)

toy_model = StateModel(states=(
    State("active", premium=True, transitions=(
        Transition("mortality"), Transition("lapse"))),
    State("care", benefit=True, duration_max=8, benefit_max_months=3,   # 3 회 보증
          transitions=(Transition("mortality"),)),
), seating=(0, 1))
toy_basis = fcf.Basis(
    mortality_annual=zero, lapse_annual=zero, discount_annual=0.0,
    ra_confidence=0.75, mortality_cv=0.10, state_model=toy_model,
    coverages=(fcf.CoverageRate("DEATH", zero),))
toy_mp = fcf.ModelPoints(
    issue_age=np.array([70], dtype=np.int64), benefits={0: np.array([0.0])},
    premium=np.array([0.0]), term_months=np.array([12], dtype=np.int64),
    disability_income=np.array([1_000_000.0]),
    state=np.array([1], dtype=np.int64),                  # 간병 상태에 자리 지정
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
tm = fcf.gmm.measure(toy_mp, toy_basis)
print("월정액 cf :", [f"{x:,.0f}" for x in tm.cashflows.disability_cf[0][:6]])
print(f"BEL       : {tm.bel[0]:,.0f}   (= 3 x 1,000,000, 할인 0)")
```

```text
월정액 cf : ['1,000,000', '1,000,000', '1,000,000', '0', '0', '0']
BEL       : 3,000,000   (= 3 x 1,000,000, 할인 0)
```

보증 3 회 (`benefit_max_months=3`) 이므로 sojourn `tau = 0, 1, 2` 세 달만
월정액이 나오고, `tau = 3` 부터 0 입니다. 계약은 여전히 in-force 지만 (감쇠가
없으니 사라지지 않음) 지급만 멈춥니다 — **종신보증형 LTC** 의 "보증 후 지급중단,
계약 유지" 가 이것입니다.

## 변형

### 확정형 (120 회 확정 후 보장 종료)

본 예제는 **보증형** (지급만 멈추고 계약 유지) 입니다. 일부 상품은 **확정형**
— 정해진 횟수를 다 지급하면 보장 자체가 끝납니다. 확정형은 점유가 한도 후
상태를 떠나는 별도 전이가 필요하므로 (occupancy 보존식 변경) 현재 `v1` 의
지급-마스크만으로는 표현되지 않습니다 — 보증형으로 모델링하거나 별도 작업이
필요합니다.

### 90 일 대기 / 2 년 감액

간병 진단 후 일정 기간 무지급 (대기) 하거나 일부만 지급 (감액) 하는 설계는,
지급 **금액** 을 sojourn (`sd`) 으로 가르는 별도 메커닉입니다.
`benefit_max_months` 는 지급 **개월수** 의 한도만 끊으므로, 금액 변조 (대기 0 /
감액 50%) 는 현재 별도 처리 (입력 단계의 금액 조정 등) 가 필요합니다.

### 회복 (간병 상태 이탈)

장기요양 등급이 호전돼 간병 상태를 벗어나는 경우는 [4.2 DI](disability-income)
의 회복 re-entry (`disability_recovery`, `duration_dependent=True`) 와 같은
방식으로 care -> active 전이를 추가합니다. LTC 는 회복이 드물어 본 예제는
생략했습니다.

## 함정 / 검증

### 함정 1 — `duration_max <= benefit_max_months`

가드 코호트가 없으면 (`duration_max == cap`) 흡수 코호트에 한도 넘은 계약이
고여 무한 지급됩니다. 생성자가 `duration_max > benefit_max_months` 를 강제하니,
보증 36 회면 `duration_max` 를 60 (또는 그 이상) 으로 넉넉히 둡니다.

### 함정 2 — 진단금과 월정액 혼동

- **`disability_benefit`** — `lump_sum` 전이가 진입 시 한 번 지급 (진단금).
- **`disability_income`** — benefit 상태 점유에 매월 지급 (월정액).

둘은 별개 필드입니다. 진단금만 주고 월정액을 비우면 매월 0 이 나옵니다.

### 함정 3 — 간병 상태 사망률을 안 주면 전역으로 fallback

`mortality_rate="dth_care"` 라고 선언해도 `state_mortality_annual` 에
`"dth_care"` 가 없으면 **전역 `mortality_annual`** 로 돌아갑니다 (기본값 보존).
상승 사망률을 의도했다면 dict 에 그 이름의 함수를 반드시 넣으세요.

## 인접 레시피

- [4.1 재진단암 보험](reincidence) — 같은 일시금 (`disability_benefit`,
  `lump_sum`) 메커닉. 본 챕터의 진단금이 같은 자리.
- [4.2 장해소득보상 (DI)](disability-income) — 같은 월정액
  (`disability_income`, `benefit=True`) 메커닉. 본 챕터는 거기에 **보증한도** 와
  **상태 사망률** 을 더한 것.
- [검증 패턴](../workflow/validation) — `gmm.trace` 로 상태별 · 코호트별 계산을
  풀어 보기.

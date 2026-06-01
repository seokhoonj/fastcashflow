# 4.2 장해소득보상 (DI, semi-Markov)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **장해소득보상 (DI)** 을 semi-Markov 로 — disabled 상태에 머무는 동안 *매월*
  정액 소득을 지급하고, 회복하면 active 로 **되돌아가는 (re-entry)** 구조
- **회복률 (disabled -> active) 이 장해 경과 (sojourn) 에 가파르게 의존** —
  급성기엔 회복이 잦고 만성화되면 거의 멈춤. semi-Markov 의 가장 교과서적인 동기
- `State(benefit=True)` 가 점유에 `disability_income` 을 *매월* 곱하는 자리
  (lump 이 아니라 정기 소득)
- 이미 장해 중인 청구건의 준비금 = **disabled life reserve (DLR)** 를
  disabled 자리 지정으로 평가
- [4.1 재진단암](reincidence) 과의 차이 — 재진단은 *전진* (post_first ->
  post_second, lump), DI 는 *회복 re-entry* (disabled -> active, 매월 소득)
```

[4.1 재진단암](reincidence) 은 한 방향으로만 흐르는 (healthy -> post_first ->
post_second) semi-Markov 였습니다. 이 챕터는 **되돌아오는** 전이를 더합니다 —
장해에서 **회복** 해 active 로 복귀하는 흐름이고, 그 회복률이 장해 경과에
의존하는 것이 DI 의 핵심입니다.

## 상품 소개 — 장해소득과 회복

**장해소득보상 (Disability Income)** 은 피보험자가 장해 상태가 되면 그 기간
동안 **매월 정액** 을 지급하고, 회복하면 지급을 멈추는 보장입니다. 진단 시
한 번 주는 일시금 (4.1 의 진단금) 과 달리, **장해가 지속되는 한 매월** 나가는
소득입니다.

DI 가 semi-Markov 의 대표 동기인 이유는 **회복률 (장해 -> 정상) 이 장해 경과에
가파르게 의존** 하기 때문입니다:

- **급성기 (장해 직후 몇 달)** — 회복이 잦습니다. 일시적 장해가 빠르게 풉니다.
- **만성기 (장해가 오래됨)** — 회복이 거의 멈춥니다. 오래 장해 상태인 사람은
  계속 장해일 확률이 높습니다.

"지금 장해냐 아니냐" 만이면 Markov 로 충분하지만, **"장해가 된 지 몇 개월이냐"**
가 회복률을 가르므로 disabled 상태의 경과 (코호트) 를 추적해야 합니다 —
semi-Markov 입니다.

```{admonition} DLR — disabled life reserve
:class: note

DI 준비금은 두 조각입니다. **active life reserve (ALR)** 는 아직 건강한
가입자가 *미래에* 장해가 될 위험의 준비금이고, **disabled life reserve (DLR)**
는 *이미 장해 중인* 청구건이 회복 / 사망할 때까지 줄 미래 소득의 준비금입니다.
회복률의 경과 의존이 가장 직접적으로 작동하는 곳이 DLR 이라, 이 챕터의 기본
예제는 **disabled 에 자리 지정한 DLR** 입니다 (ALR 은 변형에서).
```

## 모델링 매핑 — active / disabled 2-state (회복 re-entry)

이 모델도 번들 (`STATE_MODELS`) 에 없어 직접 조립합니다.

```{list-table}
:header-rows: 1
:widths: 36 64

* - 자리
  - 무엇
* - `State("active", premium=True, ...)`
  - 정상. 보험료 납입, 사망 / 장해 발생 / 해지에 노출
* - `State("disabled", benefit=True, duration_max=D, ...)`
  - 장해. `benefit=True` 가 **매월 `disability_income` 지급**, `duration_max > 0`
    이 경과 코호트 추적 (semi-Markov)
* - `Transition("waiver_incidence", to="disabled")`
  - 장해 발생 — active -> disabled
* - `Transition("disability_recovery", to="active", duration_dependent=True)`
  - 회복 — disabled -> active, **경과 의존** (re-entry)
* - `Basis.waiver_incidence_annual`
  - 장해 발생률. 시그니처 `(sex, issue_age, duration)`
* - `Basis.disability_recovery_annual`
  - 회복률. 시그니처 `(sex, issue_age, duration, state_duration)` — **네 번째
    인자가 장해 경과개월**
* - `ModelPoints.disability_income`
  - disabled 점유에 매월 곱하는 정액 소득 (lump 인 `disability_benefit` 와 다름)
```

```{admonition} 장해 발생률이 waiver_incidence 슬롯을 쓰는 이유
:class: note

active -> disabled 전이는 `waiver_incidence_annual` 을 씁니다. [3.1 납입면제](../markov/waiver)
의 "납입면제 발생" 과 DI 의 "장해 발생" 은 **같은 사건 (장해 발생)** 이기
때문입니다 — 한쪽은 그 결과로 보험료를 면제하고, 다른 쪽은 소득을 지급할 뿐
트리거는 동일합니다. 그래서 같은 발생률 슬롯을 공유합니다.
```

회복률은 경과 (`sd`) 가 작을 때 높고 커질수록 떨어지는 표를 룩업합니다 — DI
경험표의 전형적 모양입니다:

```python
# 급성기 (sd<2) 월 30% 회복, 만성기 (sd>=2) 월 5%
recovery_fn = lambda s, a, d, sd: np.where(sd < 2, 1 - (1 - 0.30) ** 12,
                                                    1 - (1 - 0.05) ** 12)
```

## 최소 작동 예제 — DLR (disabled 자리 지정)

이미 장해 중인 청구건 하나의 준비금을 봅니다. 계약을 disabled 에 자리 지정하고
(`state = 1`), 회복 / 사망까지 줄 미래 소득을 평가합니다.

```{admonition} 예제 설정
:class: note

- 가입연령 45세, 잔여 6개월, **disabled 로 시작** (DLR)
- 월 사망률 1%, 월 장해소득 1,000,000, 사망보험금 0 (소득에 집중)
- 회복률 급성기 (sd<2) 월 30% -> 만성기 (sd>=2) 월 5%
- 신규 장해 없음 (DLR 이라 active 유입 불필요), 월 할인율 0
```

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel

# 계리적 가정 -- 평탄 rate (실무는 경험률표 룩업)
death_fn     = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)  # 사망률 월 1%
lapse_fn     = lambda s, a, d: np.full(d.shape, 0.0)                   # 해지 없음
incidence_fn = lambda s, a, d: np.full(a.shape, 0.0)                   # 신규 장해 없음 (DLR)
# 회복률 -- 네 번째 인자 sd = 장해 경과개월. 급성기 30% -> 만성기 5%
recovery_fn  = lambda s, a, d, sd: np.where(sd < 2, 1 - (1 - 0.30) ** 12,
                                                     1 - (1 - 0.05) ** 12)

# 상태 모델 -- active <-> disabled (회복 re-entry)
model = StateModel(states=(
    State("active", premium=True, transitions=(
        Transition("mortality"),                              # in-force 감쇠
        Transition("waiver_incidence", to="disabled"),        # 장해 발생
        Transition("lapse"),
    )),
    State("disabled", benefit=True, duration_max=24, transitions=(  # 매월 소득 + 경과 추적
        Transition("mortality"),
        Transition("disability_recovery", to="active",
                   duration_dependent=True),                  # 회복 (경과 의존)
    )),
), seating=(0, 1, 1))

basis = fcf.Basis(
    mortality_annual           = death_fn,        # 보유계약 감쇠용 사망률 (월 1%)
    lapse_annual               = lapse_fn,        # 해지율 (없음)
    waiver_incidence_annual    = incidence_fn,    # 장해 발생률 (DLR 이라 0)
    disability_recovery_annual = recovery_fn,     # 회복률 (급성 30% -> 만성 5%)
    discount_annual            = 0.0,             # 연 할인율 0 (검증 단순화)
    ra_confidence              = 0.75,            # 위험조정 신뢰수준 75%
    mortality_cv               = 0.10,            # 사망률 변동계수 10%
    disability_cv              = 0.20,            # 장해율 변동계수 20%
    state_model                = model,           # 직접 조립한 semi-Markov 모델
    coverages                  = (
        fcf.CoverageRate("DEATH", death_fn),      # 사망 보장 1종
    ),
)

mp = fcf.ModelPoints(
    issue_age         = np.array([45], dtype=np.int64),     # 가입연령 45세
    benefits          = {0: np.array([0.0])},               # 사망보험금 0
    level_premium     = np.array([0.0]),                    # 보험료 0
    term_months       = np.array([6], dtype=np.int64),      # 잔여 6개월
    disability_income = np.array([1_000_000.0]),            # 월 장해소득 1,000,000
    state             = np.array([1], dtype=np.int64),      # disabled 코호트 0 에 자리 지정
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},
)

m = fcf.gmm.measure(mp, basis)
print(f"inforce       = {m.cashflows.inforce[0]}")        # 보유계약 (active + disabled)
print(f"disability_cf = {m.cashflows.disability_cf[0]}")  # 장해소득 (disabled 점유 × 월액)
print(f"BEL           = {m.bel[0]:.2f}")               # 최선추정부채 (= DLR)
print(f"RA            = {m.ra[0]:.2f}")                # 위험조정
print(f"CSM           = {m.csm[0]:.2f}")               # 계약서비스마진
```

출력:

```
inforce       = [1.         0.99       0.9801     0.970299   0.96059601 0.95099005]
disability_cf = [1000000.          693000.          480249.          451674.1845
  424799.57052225  399523.99607618]
BEL           = 3449246.75
RA            = 465296.32
CSM           = 0.00
```

```{note}
4.1 과 마찬가지로 `disability_income` / `state` 는 `ModelPoints.single()` 의
범위 밖이라 전체 생성자 `fcf.ModelPoints(...)` 로 넘깁니다.
```

## 결과 읽기 — 회복률의 경과 의존이 만드는 꺾임

DI 모델의 한 줄 요약: **`disability_cf` 의 감소 속도가 꺾이는 자리가 회복률이
떨어지는 (만성화) 경과다.**

| t | disability_cf | 장해 경과 sd | 회복률 | 직전 대비 |
|---|---|---|---|---|
| 0 | 1,000,000.00 | 0 | 30% | — |
| 1 |   693,000.00 | 1 | 30% | ×0.693 |
| 2 |   480,249.00 | 2 | 5% | ×0.693 |
| 3 |   451,674.18 | 3 | 5% | **×0.9405** |
| 4 |   424,799.57 | 4 | 5% | ×0.9405 |
| 5 |   399,523.996 | 5 | 5% | ×0.9405 |

- **`disability_cf[t] = disabled 점유 × 1,000,000`** — 매월 disabled 점유에 비례.
- **급성기 (sd 0~1)** 회복률 30% 라, disabled 점유가 매월 `×0.99 (사망) ×0.70
  (회복) = ×0.693` 로 **빠르게** 빠집니다.
- **만성기 (sd 2 부터)** 회복률 5% 로 떨어져, `×0.99 ×0.95 = ×0.9405` 로
  감소가 **급격히 느려집니다** — 표의 t=2 -> t=3 에서 0.693 이 0.9405 로 꺾이는
  자리입니다. 오래 장해일수록 회복이 어려워지는 DI 의 전형이고, 이 꺾임이
  semi-Markov 가 아니면 표현되지 않습니다.

```{admonition} inforce 와 disability_cf 가 다르게 줄어드는 이유
:class: note

`inforce` (= active + disabled 합) 는 `0.99^t` 로 사망으로만 줄어듭니다 —
회복한 사람은 active 로 옮겨갈 뿐 보유계약에 남고, 이 예제는 신규 장해 ·
해지가 없기 때문입니다. 반면 `disability_cf` 가 떠받치는 disabled 점유는
사망 *과* 회복으로 줄어 더 빠릅니다. 둘의 차이가 회복해서 active 로 돌아간
누적분입니다.
```

## 변형 — ALR · 회복표 · 최소보장기간

### active 에서 시작 — ALR (신계약)

신규 DI 계약을 가입 시점부터 보려면 `state` 를 active (`STATE_ACTIVE`) 로 두고
장해 발생률을 켭니다 (`waiver_incidence_annual > 0`). 그러면 active 점유가
매월 disabled 로 흘러 들어가 소득을 받기 시작하고, 일부는 회복해 돌아옵니다:

```python
from dataclasses import replace
from fastcashflow import STATE_ACTIVE

asmp_alr = replace(basis,
    waiver_incidence_annual=lambda s, a, d: np.full(a.shape, 1 - (1 - 0.02) ** 12))
mp_alr = fcf.ModelPoints(
    issue_age=np.array([45], dtype=np.int64), benefits={0: np.array([0.0])},
    level_premium=np.array([0.0]), term_months=np.array([6], dtype=np.int64),
    disability_income=np.array([1_000_000.0]), state=np.array([STATE_ACTIVE], dtype=np.int64),
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
```

### 회복률 표의 모양

본문은 `sd<2` 2-구간이지만, 실무 DI 경험표는 보통 더 잘게 나눕니다 — 장해
직후 수개월은 높은 회복, 1~2년차에 급락, 이후 거의 평탄:

```python
recovery_fn = lambda s, a, d, sd: np.where(sd < 3,  1 - (1 - 0.20) ** 12,
                                  np.where(sd < 12, 1 - (1 - 0.05) ** 12,
                                                    1 - (1 - 0.01) ** 12))
```

`duration_max` 는 회복을 경과별로 추적할 개월 수입니다. 1~2년차의 회복 급락을
담으려면 `duration_max=24` (2년) 이상이 무난하고, 마지막 코호트가 그 이상의
장기 장해를 흡수합니다.

### 최소보장기간 / 면책

장해 직후 일정 기간은 무조건 지급 (회복 무시) 하거나, 반대로 일정 기간이
지나야 지급을 시작하는 (elimination period) 설계는 회복률 / 소득 지급을 `sd`
로 분기해 표현합니다 — 4.1 의 재진단 면책과 같은 경과 축 기법입니다.

## 함정 / 검증

### 손계산 검증 — disabled 1개월

disabled 에 자리 지정하고 (`state=1`) 한 달만 굴리면, 그 달의 장해소득
하나만 남습니다 (할인 0, 사망보험금 0). BEL = `disability_income` =
**1,000,000**. 위 예제의 `disability_cf[0]` 과 같은 값입니다.

```python
mp1 = fcf.ModelPoints(
    issue_age=np.array([45], dtype=np.int64), benefits={0: np.array([0.0])},
    level_premium=np.array([0.0]), term_months=np.array([1], dtype=np.int64),
    disability_income=np.array([1_000_000.0]), state=np.array([1], dtype=np.int64),
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
print(f"seated 1mo BEL = {fcf.gmm.measure(mp1, basis, full=False).bel[0]:.2f}")   # -> 1000000.00
```

### 함정 1 — `disability_income` 과 `disability_benefit` 혼동

- **`disability_income`** — benefit 상태 점유에 *매월* 곱하는 정기 소득. DI 가
  쓰는 자리.
- **`disability_benefit`** — `lump_sum` 전이가 한 번 지급하는 일시금. 4.1
  재진단금이 쓰는 자리.

DI 에서 `disability_benefit` 에 금액을 넣고 `disability_income` 을 비우면
매월 소득이 0 이 됩니다 (transition lump 이 없으니 아무것도 안 나감).

### 함정 2 — `duration_max = 0` 이면 회복의 경과 의존 불가

disabled 의 `duration_max` 를 0 으로 두면 경과를 추적하지 않아 회복률의
경과 의존 (급성 -> 만성 꺾임) 을 표현할 수 없습니다 — DI 의 핵심이 사라집니다.

### 함정 3 — 회복을 in-force 감쇠로 착각

회복 (disabled -> active) 은 보유계약을 떠나보내지 않습니다 — active 로
되돌아갈 뿐입니다. in-force 를 줄이는 것은 사망 / 해지뿐입니다. 회복을
decrement 로 잘못 두면 보유계약이 과소평가됩니다.

## 인접 레시피

- [4.1 재진단암 보험](reincidence) — 같은 semi-Markov 인프라, 단 *전진*
  (재진단 lump) 이고 본 챕터는 *회복 re-entry* (매월 소득).
- [3.1 보험료 납입면제](../markov/waiver) — 장해 발생률 (`waiver_incidence`)
  의 출발점. DI 의 active -> disabled 가 같은 슬롯을 공유.
- [검증 패턴](../workflow/validation) — `gmm.trace` 로 상태별 · 코호트별
  점유와 소득 지급을 한 줄씩 확인.
```

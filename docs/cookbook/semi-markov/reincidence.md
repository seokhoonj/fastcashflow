# 4.1 재진단암 (Semi-Markov)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **Semi-Markov** (상태 안에서의 **경과 시간** 에 의존하는 모델) 의 첫 사례 —
  앞의 Markov 챕터는 "어느 상태냐" 만 봤지만, 여기서는 "그 상태에 들어온 지
  몇 개월이냐" 가 보험금을 가른다
- 등록된 모델이 없어 `State` / `Transition` / `StateModel` 로 **상태 모델을
  직접 조립** — 쿡북에서 처음
- `State.sojourn_tracking_months` (코호트 추적) 와 `Transition.sojourn_dependent`
  (경과 의존 전이) 의 연결
- **재진단 면책기간** 을 `ci_reincidence_annual` 의 네 번째 인자 (state
  duration) 로 표현하는 자리
- 1차 / 2차 진단금이 `pays_lump_sum` 전이로 들어가는 자리와, `disability_benefit`
  한 금액을 공유하는 제약
```

[3. Markov 상태](../markov/waiver) 는 "어느 상태에 있느냐" 로 보험료와
보장이 갈렸습니다. 이 챕터는 한 걸음 더 갑니다 — **그 상태에 들어온 지 얼마나
지났느냐** 가 보장을 가르는 **Semi-Markov** 입니다. 한국 시장의 재진단암
보장이 대표 사례입니다.

## 상품 소개 — 재진단암과 면책기간

**재진단암 보장** 은 암을 한 번 진단받은 뒤, 일정 기간이 지나 **다시** 암을
진단받으면 진단금을 또 지급하는 구조입니다. 핵심은 1차 진단 직후에는 재진단
보장이 **발효되지 않는** 점입니다 — 1차 진단 후 보통 1~2년의 **재진단
면책기간** 을 둡니다. 같은 암의 잔존 / 전이를 새 사건으로 잘못 지급하는 것을
막기 위해서입니다.

이 구조가 [1.4 담보별 산출방법](../basics/coverage-mechanics) 의 단순
진단금 (DIAGNOSIS) 으로 표현되지 않는 이유는, **2차 보장이 두 가지에 동시에
의존** 하기 때문입니다:

- **상태** — 1차 진단을 이미 거쳤는가 (안 거쳤으면 재진단이라는 말이 성립 안 함)
- **경과** — 1차 진단 후 면책기간을 넘겼는가

"어느 상태냐" 만이면 Markov 로 충분하지만, "그 상태에 들어온 지 몇 개월이냐"
가 더해지면 **상태별 경과(코호트)를 추적** 해야 합니다. 그것이 Semi-Markov
입니다.

```{admonition} 코호트 (cohort) 란
:class: note

같은 시점에 같은 상태로 들어온 무리를 하나의 **코호트** 로 봅니다. post_first
(1차 진단 후) 상태를 "들어온 지 0개월 / 1개월 / 2개월 ..." 코호트로 쪼개
추적하면, 각 코호트가 면책기간을 넘겼는지 따로 알 수 있습니다. `sojourn_tracking_months`
가 추적할 코호트 수 (개월) 입니다.
```

## 모델링 매핑 — Semi-Markov 3-state

이 상품은 번들 모델 (`STATE_MODELS`) 에 없습니다. `State` / `Transition` /
`StateModel` 로 직접 조립합니다 — 세 상태와 그 사이 전이를 그대로 적습니다.

```{list-table}
:header-rows: 1
:widths: 34 66

* - 자리
  - 무엇
* - `State("healthy", pays_premium=True, ...)`
  - 정상 (1차 진단 전). 보험료 납입, 사망 / 1차 진단 / 해지에 노출
* - `State("post_first", sojourn_tracking_months=12, ...)`
  - 1차 진단 후. `sojourn_tracking_months > 0` 이 **코호트 추적을 켠다** (Semi-Markov)
* - `State("post_second", ...)`
  - 2차 진단 후. 추가 추적 없이 사망까지
* - `Transition("ci_incidence", to="post_first", pays_lump_sum=True)`
  - 1차 진단 — healthy → post_first, 진단금 지급
* - `Transition("ci_reincidence", to="post_second", pays_lump_sum=True, sojourn_dependent=True)`
  - 2차 진단 — post_first → post_second, **경과 의존** (면책기간이 여기)
* - `Basis.ci_incidence_annual`
  - 1차 진단율. 시그니처 `(sex, issue_age, duration)` — 기존 rate 와 동일
* - `Basis.ci_reincidence_annual`
  - 2차 진단율. 시그니처 `(sex, issue_age, duration, state_duration)` — **네
    번째 인자 `state_duration`** 가 post_first 진입 후 경과개월
* - `ModelPoints.disability_benefit`
  - `pays_lump_sum` 전이가 지급하는 금액. **모든 pays_lump_sum 전이가 공유** (아래 함정)
```

세 상태와 전이를 그림으로:

```{mermaid}
flowchart LR
    START(("신계약")) --> H["healthy<br/>건강"]
    H -->|"ci_incidence<br/>(1차 진단금)"| P1["post_first<br/>1차 후"]
    P1 -->|"ci_reincidence<br/>(2차 진단금, 경과 의존)"| P2["post_second<br/>2차 후"]
    H -->|"mortality · lapse"| EXIT(("종료"))
    P1 -->|"mortality"| EXIT
    P2 -->|"mortality"| EXIT
    classDef stock fill:#eaf1f8,stroke:#547fa6,color:#17344e
    classDef step fill:#f7f2e8,stroke:#b38a45,color:#493617
    class H,P1,P2 stock
    class START,EXIT step
```

면책기간은 별도 필드가 아니라 **`ci_reincidence_annual` 안에서 자연스럽게**
표현됩니다 — 네 번째 인자 `state_duration` 가 면책개월 미만이면 0 을
돌려줍니다:

```python
# 면책 2개월: post_first 진입 후 sd < 2 면 재진단율 0, 이후 월 20%
reincid_fn = lambda s, a, d, sd: np.where(sd < 2, 0.0, 1 - (1 - 0.20) ** 12)
```

## 최소 작동 예제

가입연령 40세, 보험기간 4개월의 한 계약입니다. 손계산이 따라가도록 rate 를
평탄하게 (실무는 경험률표 룩업) 두고, 진단율 · 진단금을 일부러 크게 잡아
면책기간의 효과가 또렷이 보이게 했습니다.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 4개월, healthy 로 시작
- 월 사망률 1%, 사망보험금 100,000, 보험료 0 (보장 움직임에 집중)
- 1차 진단 월 5%, 재진단 월 20% (면책 2개월), 진단금 1,000,000 (1차 = 2차)
- 월 할인율 0
```

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel

# rate 함수 -- 모든 rate 는 평탄 상수 (실무는 경험률표 룩업)
death_rate     = 1 - (1 - 0.01) ** 12  # 사망률 월 1%
lapse_rate     = 0.0                   # 해지 없음
incidence_rate = 1 - (1 - 0.05) ** 12  # 1차 진단 월 5%

# 재진단 -- 네 번째 인자 sd = post_first 진입 후 경과개월. 면책 2개월 후 월 20%
reincid_fn   = lambda s, a, d, sd: np.where(sd < 2, 0.0, 1 - (1 - 0.20) ** 12)

# 상태 모델 -- healthy → post_first → post_second (직접 조립)
model = StateModel(states=(
    State("healthy", pays_premium=True, transitions=(
        Transition("mortality"),  # in-force 감쇠
        Transition("ci_incidence", to="post_first", pays_lump_sum=True),  # 1차 진단금
        Transition("lapse"),
    )),
    State("post_first", sojourn_tracking_months=12, transitions=(                 # 경과 추적 (코호트)
        Transition("mortality"),
        Transition("ci_reincidence", to="post_second",
                   pays_lump_sum=True, sojourn_dependent=True),  # 2차 진단금 (면책 의존)
    )),
    State("post_second", transitions=(
        Transition("mortality"),
    )),
), seating=(0, 1, 2))

# 산출기초
basis = fcf.Basis(
    mortality_annual      = death_rate,      # 보유계약 사망률 (월 1%)
    lapse_annual          = lapse_rate,      # 해지율 (없음)
    ci_incidence_annual   = incidence_rate,  # 1차 진단율 (월 5%)
    ci_reincidence_annual = reincid_fn,      # 2차 진단율 (면책 2개월 후 월 20%)
    discount_annual       = 0.0,             # 연 할인율 0 (검증 단순화)
    ra_confidence         = 0.75,            # 위험조정 신뢰수준 75%
    mortality_cv          = 0.10,            # 사망률 변동계수 10%
    state_model           = model,           # 직접 조립한 Semi-Markov 모델
    coverages             = (
        fcf.CoverageRate("DEATH", death_rate),  # 사망 보장 1종
    ),
)

# 모델 포인트
mp = fcf.ModelPoints(
    issue_age          = np.array([40], dtype=np.int64),  # 가입연령 40세
    benefits           = {"DEATH": np.array([100_000.0])},  # 사망보험금 100,000
    premium            = np.array([0.0]),                 # 보험료 0
    term_months        = np.array([4], dtype=np.int64),   # 보험기간 4개월
    disability_benefit = np.array([1_000_000.0]),         # 진단금 1,000,000 (1차 = 2차)
    calculation_methods= {"DEATH": fcf.CalculationMethod.DEATH},
)

m = fcf.gmm.measure(mp, basis)
print(f"inforce       = {m.cashflows.inforce[0]}")  # 보유계약 (사망으로만 감쇠)
print(f"claim_cf      = {m.cashflows.claim_cf[0]}")  # 사망보험금
print(f"disability_cf = {m.cashflows.disability_cf[0]}")  # 진단금 (1차 + 2차)
print(f"BEL           = {m.bel[0]:.2f}")  # 최선추정부채
print(f"RA            = {m.ra[0]:.2f}")   # 위험조정
print(f"CSM           = {m.csm[0]:.2f}")  # 계약서비스마진
```

출력:

```
inforce       = [1.       0.99     0.9801   0.970299]
claim_cf      = [1000.     990.     980.1    970.299]
disability_cf = [49500.         46554.75       43784.742375   50785.51030369]
BEL           = 194565.40
RA            = 265.78
CSM           = 0.00
```

```{note}
이 예제는 전체 생성자 `fcf.ModelPoints(...)` 를 씁니다 — `single()` 도
`disability_benefit` / `state` 를 받지만, Semi-Markov 예제들과 같은 **명시적
배열 스타일**을 유지하려고 전체 생성자로 넘깁니다.
```

## 결과 읽기 — 면책기간이 만드는 점프

재진단암 모델의 한 줄 요약: **`disability_cf` 의 점프가 면책기간이 끝나는
자리다.**

- **`inforce = 0.99^t`** — 사망 (월 1%) 으로만 줄어듭니다. 1차 / 2차 진단은
  상태 사이를 옮길 뿐 보유계약을 떠나보내지 않으므로 (healthy → post_first
  → post_second 모두 보유계약), in-force 총합은 진단과 무관합니다.
- **`claim_cf`** — 사망보험금. `inforce × 1% × 100,000`.
- **`disability_cf`** — 진단금 lump 들의 합. 여기에 면책기간이 드러납니다:

| t | disability_cf | 내역 |
|---|---|---|
| 0 | 49,500.00 | 1차 진단금만 (post_first 가 아직 빔) |
| 1 | 46,554.75 | 1차 진단금만 (재진단은 면책 안) |
| 2 | 43,784.74 | 1차 진단금만 (재진단은 면책 안) |
| 3 | 50,785.51 | 1차 진단금 (약 41,180) **+ 첫 2차 진단금 (약 9,606)** |

t=0~2 는 1차 진단금이 매월 사망만큼 (`× 0.9405`) 줄며 이어집니다. **t=3 에서
값이 점프** 하는 것이 재진단의 신호입니다:

- 1차 진단 코호트는 t=0 의 transition 으로 post_first 에 생깁니다 (코호트 0).
- 면책 2개월 = post_first 진입 후 경과 `sd = 0, 1` 두 달은 재진단율 0.
- `sd = 2` (post_first 3개월째) 부터 재진단이 발효 → **t=3 에서 첫 2차 진단금**.

즉 t=3 의 50,785 중 약 41,180 은 1차 진단금의 연속이고, 약 9,606 이 면책이
풀리며 처음 나타난 2차 진단금입니다. 면책을 더 길게 잡으면 이 점프가 그만큼
뒤로 밀립니다.

## 변형 — 면책 · 추적기간 · 진단금 분리

### 현실적 율 — 공개 암발생률 + 재발 hazard

면책은 `ci_reincidence_annual` 의 `sd` 분기 하나로 정해집니다 (1년이면 `sd < 12`,
2년이면 `sd < 24`). 실무에서는 평탄 상수 대신 **1차 진단율을 공개 암발생률
(국가암등록) 의 연령표** 로 깔고 (Excel 룩업), **재진단율을 그 위에 재발 hazard
배수** 로 얹습니다 — 재발 위험은 1차 진단 후 1~3년에 높고 (면책 직후), 5년이
지나면 신규 원발암 수준으로 가라앉는 임상적 패턴입니다. 율은 분석식이 아니라
**연령별 long-form 표 + 룩업** (견본 `incidence_rate_tables` 와 같은
`(sex, age) -> rate` 구조를 inline 으로) 입니다:

```python
import numpy as np

# 계리적 가정 -- 1차 진단율 = KOSIS 국가암등록 연령표 (long-form; 실무는 Excel 룩업)
ages = np.array([   30,     40,     50,     60,     70])
ca_m = np.array([0.0010, 0.0023, 0.0052, 0.0117, 0.0264])  # 남
ca_f = np.array([0.0028, 0.0040, 0.0061, 0.0108, 0.0220])  # 여 (젊은연령 갑상선↑)

def ca_incidence(s, a, d):                          # 연령표 룩업 (VLOOKUP 식 보간)
    a = np.asarray(a, dtype=float)
    return np.where(np.asarray(s) == 1,
                    np.interp(a, ages, ca_f), np.interp(a, ages, ca_m))

def ca_reincidence(s, a, d, sd):                    # 재진단율 = 1차 x 재발배수 (면책 12개월)
    factor = np.select([sd < 12, sd < 36, sd < 60], [0.0, 1.8, 1.2], default=0.9)
    return ca_incidence(s, a, d) * factor

print("1st dx 30/40/50/60 :", [round(float(ca_incidence(np.array([0]), np.array([a]), 0)[0]), 5)
                           for a in (30, 40, 50, 60)])
print("reincidence mult sd6/24/48/72:", [float(np.select([np.array([x]) < 12, np.array([x]) < 36,
       np.array([x]) < 60], [0.0, 1.8, 1.2], default=0.9)[0]) for x in (6, 24, 48, 72)])
```

```text
1st dx 30/40/50/60 : [0.001, 0.0023, 0.0052, 0.0117]
reincidence mult sd6/24/48/72: [0.0, 1.8, 1.2, 0.9]
```

`ca_incidence` 를 `ci_incidence_annual`, `ca_reincidence` 를 `ci_reincidence_annual`
에 넣으면 1차 진단은 공개 발생률을, 재진단은 면책 + 경과 의존 재발 hazard 를
따릅니다.

```{admonition} 출처 / 근거
:class: note

- **1차 진단율 연령 구조** — 국가암등록통계 (KOSIS, 공개) 의 암 발생률 곡선
  (연령 상승, 여성은 젊은 연령서 갑상선암으로 높음).
- **재발 hazard 패턴** — 재발 위험이 진단 후 1~3년에 높고 5년 후 신규 원발암
  수준으로 가라앉는 임상적 재발 곡선; `sd` 경과 배수로 표현.
- 면책 (1~2년) 은 `sd` 임계 하나로 — `sd < 12` (1년) / `sd < 24` (2년).
```

### 암진단 후 사망률 (조건부 사망 가정)

암진단을 받은 사람은 그 뒤 (전원인) 사망률이 일반보다 훨씬 높습니다 — **암진단
후 사망률**. 이건 **담보(지급 항목)가 아니라 가정** 입니다: 진단 후 상태에
머무는 사람의 사망 decrement 로, **재진단금** 의 경쟁위험 (진단자가 빨리 죽으면
재진단까지 생존자가 줄어 재진단금 부채 감소) 이자, 진단 후 월정액 같은 다른
보장을 평가하는 사망률 가정입니다 (암을 死因으로 하는 **암사망 보험금** 과도
별개). 진단 상태 (`post_first` / `post_second`) 가 **자기 사망률** 을 갖게 하려면
`State.mortality_rate_name` 로 다른 이름을 라우팅하고, `Basis.state_mortality_annual`
에 그 함수를 줍니다 — in-force 가 진단 후 더 빨리 소멸합니다:

```python
pm_healthy = 0.005  # 건강 사망 연 0.5%
pm_post    = 0.02   # 암진단 후 연 2% (건강의 4배)
pm_lapse   = 0.05

pm_model = StateModel(states=(
    State("healthy", pays_premium=True, transitions=(
        Transition("mortality"), Transition("lapse"),
        Transition("ci_incidence", to="post_first", pays_lump_sum=True))),
    State("post_first", sojourn_tracking_months=120, mortality_rate_name="dth_aft_can", transitions=(
        Transition("mortality"), Transition("lapse"),
        Transition("ci_reincidence", to="post_second", pays_lump_sum=True, sojourn_dependent=True))),
    State("post_second", mortality_rate_name="dth_aft_can", transitions=(
        Transition("mortality"), Transition("lapse"))),
), seating=(0, 1, 2))
pm_basis = fcf.Basis(
    mortality_annual=pm_healthy,           # 건강상태 사망률
    lapse_annual=pm_lapse,                 # 해지
    ci_incidence_annual=ca_incidence,      # 1차 암 진단율
    ci_reincidence_annual=ca_reincidence,  # 재진단율 (sojourn 의존)
    state_mortality_annual={"dth_aft_can": pm_post},  # 암진단 후 사망률 (가정)
    discount_annual=0.03,                  # 할인율
    ra_confidence=0.75,                    # 위험조정 신뢰수준
    mortality_cv=0.10,                     # 사망 변동계수
    morbidity_cv=0.15,                     # 발생 변동계수
    state_model=pm_model,                  # 상태기계
    coverages=(fcf.CoverageRate("CANCER1", ca_incidence),),  # 1차 암 진단 담보
)

# post_first 에 자리 지정 -> 사망건수가 암진단 후 사망률을 따른다
pm_mp = fcf.ModelPoints(
    issue_age=np.array([50], dtype=np.int64), benefits={"CANCER1": np.array([0.0])},
    premium=np.array([20_000.0]), term_months=np.array([13], dtype=np.int64),
    disability_benefit=np.array([20_000_000.0]), state=np.array([1], dtype=np.int64),
    calculation_methods={"CANCER1": fcf.CalculationMethod.DIAGNOSIS})
pm_m = fcf.gmm.measure(pm_mp, pm_basis)
print("post-dx seated deaths[0] :", round(float(pm_m.cashflows.deaths[0][0]), 5))
print("healthy / post-dx monthly mort :", round(1 - (1 - 0.005) ** (1 / 12), 5),
      "/", round(1 - (1 - 0.02) ** (1 / 12), 5))
```

```text
post-dx seated deaths[0] : 0.00168
healthy / post-dx monthly mort : 0.00042 / 0.00168
```

`post_first` 에 자리 지정한 계약의 사망건수 (`deaths[0]`) 가 **암진단 후 사망률
0.00168** 을 따릅니다 (건강 0.00042 가 아니라). `mortality_rate_name` 를 안 주면
전역 `mortality_annual` 로 fallback 하므로, 암진단 후 상승 사망을 의도했다면
`state_mortality_annual` 에 그 함수를 반드시 넣습니다.

### 추적기간 `sojourn_tracking_months`

`sojourn_tracking_months` 는 post_first 의 경과를 몇 개월까지 코호트로 추적할지입니다.
한국 재진단암은 보통 1차 진단 후 5년 (60개월) 을 추적하므로 `sojourn_tracking_months=60`
이 무난합니다. 마지막 코호트는 그 이상의 경과를 모두 흡수합니다 (long-tail).
`sojourn_tracking_months` 가 크면 코호트 수만큼 계산이 늘지만 시간에 선형으로 증가합니다.

```{admonition} sojourn_tracking_months = 0 이면 Markov 로 돌아간다
:class: warning

`post_first` 의 `sojourn_tracking_months` 를 0 으로 두면 코호트 추적이 꺼져 경과를 알 수
없습니다. 그러면 `sojourn_dependent=True` 전이의 면책기간을 표현할 수 없습니다 —
재진단 보장의 핵심이 사라집니다. Semi-Markov 의 본질이 이 `sojourn_tracking_months > 0`
입니다.
```

### 1차 ≠ 2차 진단금 — DIAGNOSIS 담보로 분리

본문 예제는 1차 / 2차 진단금이 **같은 금액** 입니다 — 모든 `pays_lump_sum` 전이가
`disability_benefit` 한 값을 공유하기 때문입니다 (아래 함정). 1차를 다른
금액으로 주려면, 1차 진단금을 **DIAGNOSIS 담보** 로 분리하고 (고유 금액),
2차만 transition lump 로 남깁니다:

```python
coverages = (
    fcf.CoverageRate("DEATH",   death_rate),      # 사망
    fcf.CoverageRate("CANCER1", incidence_rate),  # 1차 진단금 (DIAGNOSIS, 고유 금액)
)
# ci_incidence 전이에서는 pays_lump_sum 을 빼고 (Transition("ci_incidence", to="post_first")),
# benefits 에 CANCER1 의 진단금을, calculation_methods 에 DIAGNOSIS 를 등록
```

이때 1차 진단의 발생률을 **두 자리** 에 쓰게 됩니다 — DIAGNOSIS 담보의 rate
와 `ci_incidence` 전이율. 둘은 같은 사건 (첫 암진단) 이므로 **같은 함수로
맞춰** 두어야 모델이 일관됩니다 (담보는 진단금 지급, 전이는 재진단 자격을
위한 상태 진행을 맡는 분업).

### 위험률의 vintage / composite

한 재진단암 특약이라도 가입연도 (vintage) 에 따라 다른 경험률이 적용되거나
(예: `갑상선암 발생률 (2019)` vs `(2021)`), 유사암처럼 여러 발생률을 합성한
(composite) 위험률을 쓸 수 있습니다. fastcashflow 는 `coverage → 단일
rate` 만 받으므로, vintage 선택과 composite 합성은 **ETL 단계에서 미리** 해서
이미 합쳐진 한 rate 를 넘깁니다.

## 함정 / 검증

### 손계산 검증 — post_first 에 자리 지정해 한 달

면책 밖의 재진단을 또렷이 검증하려면, 계약을 **처음부터 post_first 코호트 0**
에 자리 지정하고 (`state = 1`) 재진단을 면책 없이 (`sd` 무관 상수) 한 달만
굴립니다. 손계산:

- 사망보험금 = `1.0 × 0.01 × 100,000 = 1,000`
- 재진단금 = 사망 경합 후 `0.99 × 0.20 × 1,000,000 = 198,000`
- BEL = 1,000 + 198,000 = **199,000**

```python
from dataclasses import replace

mp_seat = fcf.ModelPoints(
    issue_age          = np.array([40], dtype=np.int64),
    benefits           = {"DEATH": np.array([100_000.0])},
    premium            = np.array([0.0]),
    term_months        = np.array([1], dtype=np.int64),
    disability_benefit = np.array([1_000_000.0]),
    state              = np.array([1], dtype=np.int64),           # post_first 코호트 0 에 자리 지정
    calculation_methods= {"DEATH": fcf.CalculationMethod.DEATH},
)
# 재진단을 면책 없이 상수로 (검증용)
asmp_no_excl = replace(basis,
    ci_reincidence_annual=lambda s, a, d, sd: np.full_like(sd, 1 - (1 - 0.20) ** 12,
                                                           dtype=float))
print(f"seated BEL = {fcf.gmm.measure(mp_seat, asmp_no_excl, full=False).bel[0]:.2f}")   # -> 199000.00
```

### 함정 1 — `disability_benefit` 한 금액을 모든 lump 이 공유

`pays_lump_sum=True` 전이는 전부 `ModelPoints.disability_benefit` 한 값을 지급합니다.
1차 / 2차 진단금을 다르게 주려면 위 **변형** 처럼 1차를 DIAGNOSIS 담보로
분리해야 합니다. 같은 금액이면 본문 예제처럼 둘 다 transition lump 로 두는
것이 가장 단순합니다.

### 함정 2 — `sojourn_dependent` 인데 rate 가 3-인자

`ci_reincidence_annual` 은 네 번째 인자 `state_duration` 을 받아야 합니다.
`(s, a, d)` 3-인자로 쓰면 `sojourn_dependent=True` 전이가 경과를 넘겨줄 자리가
없습니다. 면책 / 경과 의존을 쓰려면 `(s, a, d, sd)` 4-인자로 정의하세요.

### 함정 3 — 진단을 in-force 감쇠로 착각

진단 (`ci_incidence` / `ci_reincidence`) 은 상태 사이 의 이동이라 보유계약을
떠나보내지 않습니다. in-force 를 줄이는 것은 사망 / 해지뿐입니다. 진단을
decrement 로 잘못 모델링하면 in-force 가 과소평가되고 이후 보장이 틀립니다.

## 인접 레시피

- [1.4 담보별 산출방법](../basics/coverage-mechanics) — 단순 진단금
  (DIAGNOSIS, depleting pool). 재진단이 이걸로 안 되는 이유의 출발점.
- [2.3 다종 진단 + 면책 / 감액](../simple/diagnosis-rules) — 담보 룰의 면책
  (가입경과 축) 과 본 챕터의 면책 (state 경과 축) 의 대비.
- [3.1 보험료 납입면제](../markov/waiver) — 상태 전이 입문 (Markov).
- [4.2 장해소득보상 (DI)](disability-income) — 같은 Semi-Markov 인프라로 회복률
  (disabled → active) 을 경과 의존으로.
- [검증 패턴](../workflow/validation) — `gmm.trace` 로 연도별 rate 와 cash
  flow, BEL/CSM 궤적을 한 줄씩 확인 (상태별 · 코호트별 점유는 커널 안에만
  있어 trace 에 직접 나오지 않습니다).

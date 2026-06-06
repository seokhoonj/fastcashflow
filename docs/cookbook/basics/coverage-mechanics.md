# 1.4 담보별 산출방법

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 사망·진단·입원 세 보장의 보험금이 **모두 발생(incidence)으로 계산**되고,
  식은 `inforce × 발생률 × benefit`로 셋 다 똑같다는 것
- "한 번 vs 반복"을 가르는 건 식이 아니라 1.3에서 본 그 축 — **그 발생
  사건이 너를 어느 풀에서 빼내는가**: 계약 전체(사망) / 자기 미진단 풀
  (진단) / 없음(입원)
- 사망의 탈퇴는 *별도의* 사망탈퇴율이, 진단의 탈퇴는 *같은* 진단 발생률이
  맡는 비대칭
- `is_diagnosis` flag의 진짜 의미 — "한 번만 지급되는가"가 아니라
  "이 보장이 자기만의 미진단 풀을 갖고 있는가"
- 손계산 / 엔진 출력 / `gmm.trace`의 `undiagnosed` 트리가 어떻게
  서로를 검증하는가
```

각 `CalculationMethod` 값이 `calculation_methods.csv` 에 적는 라벨 이라면 —
`DEATH` 라고 적는 것 vs `DIAGNOSIS` 라고 적는 것의 차이는 어디서 정해지나?
그 선택은 [CalculationMethod 결정 가이드](calculation-methods) 의 자리이고,
본 챕터는 **엔진이 그 산출방법을 받아서 무엇을 다르게 하는가** 입니다.

이 챕터의 모든 예제는 동일한 toy 설정 — 월 사망/발생/진단율 1%, 보험금
12,000, 3 개월 — 위에서 산출방법만 바꿔 결과를 비교합니다. 한 페이지에 세 모드의
차이가 명료하게 드러나도록 의도된 셋업입니다.

## 발생은 같고, 빠지는 풀이 다르다

세 보장 모두 보험금은 **발생(incidence)**으로 계산합니다 — 매월
`inforce × 발생률 × benefit`. 식은 셋이 똑같습니다. "한 번이냐 반복이냐"를
가르는 건 식이 아니라, 1.3에서 본 그 축 — **그 발생 사건이 너를 어느 풀에서
빼내는가**:

```{list-table}
:header-rows: 1
:widths: 20 16 44 20

* - 보장
  - 발생 사건
  - 빠지는 풀
  - 결과
* - 사망 (DEATH)
  - 사망
  - 계약 전체 `inforce` (별도 사망탈퇴율로 축소)
  - 한 번
* - 진단 (DIAGNOSIS)
  - 진단
  - 그 담보의 `undiagnosed` 풀 (진단 발생률로 축소)
  - 한 번
* - 입원 (MORBIDITY)
  - 입원
  - 없음 (축소되는 풀 없음)
  - 반복
```

```{mermaid}
flowchart TB
    DEATH["사망 (DEATH)<br/>계약 전체 풀에서 탈퇴 · 한 번"]
    DIAG["진단 (DIAGNOSIS)<br/>자기 미진단 풀에서 탈퇴 · 한 번"]
    MORB["입원 (MORBIDITY)<br/>어느 풀도 안 빠짐 · 반복"]
    classDef stock fill:#eaf1f8,stroke:#547fa6,color:#17344e
    classDef outflow fill:#f9eeee,stroke:#b96d6d,color:#552626
    classDef step fill:#f7f2e8,stroke:#b38a45,color:#493617
    class DEATH step
    class DIAG stock
    class MORB outflow
```

엔진 **구현**으로는 청구 알고리즘이 둘뿐입니다 — 자기 미진단 풀을 가진
진단(B)과, 그렇지 않은 나머지(A):

```{list-table}
:header-rows: 1
:widths: 22 18 60

* - 알고리즘
  - 보장
  - 청구 식
* - (A) `inforce × rate`
  - 사망, 입원
  - 매월 `inforce[t] × rate[t] × benefit` 누적. 끝.
* - (B) `inforce × undiagnosed × rate`
  - 진단
  - 매월 `inforce[t] × undiagnosed[t] × rate[t] × benefit` 누적.
    `undiagnosed` 는 보장마다 자기 미진단 풀, 매월 `(1 - 진단발생률)` 로 감쇠.
```

`is_diagnosis` flag 가 정확히 이 (A)/(B) 분기 결정자입니다. False → 식 (A),
True → 식 (B). 한 가지 주의 — 사망의 "한 번"은 알고리즘 (A) *안*이 아니라
**별도의 사망탈퇴율이 계약 풀을 줄이는 데서** 나옵니다. 그래서 사망과 입원이
같은 (A)인데도 결과가 정반대입니다.

아래 세 절이 같은 toy 데이터로 그 차이를 **사망 → 진단 → 입원** 순으로
보여줍니다.

## 사망 — 계약 전체 풀에서 탈퇴

월 사망률 1% 의 단일 사망보장. 사망 사건은 사람을 **계약 전체 `inforce`
에서 빼냅니다** — 그 탈퇴를 맡는 건 **별도의 사망탈퇴율**(`mortality_annual`)
이고, 사망보험금 발생률 자체는 어떤 풀도 줄이지 않습니다(1.3). 이 toy 에서는
두 입력에 같은 1% 를 주어, 발생률로 청구가 매월 일어나도 같은 율로 계약 풀이
줄어 누적이 "한 번"으로 수렴하는 걸 봅니다.

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda s, a, d: np.full(d.shape, 0.0)

basis = fcf.Basis(
    mortality_annual = death_fn,                                # 보유계약 사망률 (위 death_fn)
    lapse_annual     = lapse_fn,                                # 해지율 (해지 없음)
    discount_annual  = 0.0,                                     # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,                                    # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,                                     # 사망률 변동계수 0 (RA = 0 강제)
    coverages        = (fcf.CoverageRate("DEATH", death_fn),),  # 사망 보장 1 종 (청구 rate = 같은 death_fn)
)
mp = fcf.ModelPoints.single(
    issue_age           = 40,           # 가입연령 40세
    sex                 = 0,            # 성별 (0=남, 1=여)
    benefits            = {0: 12_000},  # 0번 보장 (= DEATH) 의 보험금 12,000
    premium             = 0,            # 월납 보험료 0 (보험료 cash flow 무시)
    term_months         = 3,            # 보험기간 3개월
    calculation_methods = {
        "DEATH": fcf.CalculationMethod.DEATH,   # 코드 → 산출방법 매핑
    },
)
r = fcf.gmm.measure(mp, basis)

print(f"inforce       : {r.cashflows.inforce[0, :3]}")   # 보유계약 trajectory
print(f"claim_cf      : {r.cashflows.claim_cf[0, :3]}")  # 사망보험금 cash flow
print(f"BEL[0]        : {float(r.bel[0]):.2f}")          # 최선추정부채
print(f"cumulative 3m : {float(r.cashflows.claim_cf[0, :3].sum()):.2f}")
```

출력:

```
inforce       : [1.     0.99   0.9801]
claim_cf      : [120.    118.8   117.612]
BEL[0]        : 356.41
cumulative 3m : 356.41
```

손계산:

| t | `inforce` | rate × benefit | claim |
|---|---|---|---|
| 0 | 1.0000 | 0.01 × 12,000 | 120.00 |
| 1 | 0.9900 | 0.01 × 12,000 | 118.80 |
| 2 | 0.9801 | 0.01 × 12,000 | 117.61 |

**핵심**: 청구가 매월 발생하는데도 누적이 ~356 으로 수렴 — `inforce` 가
줄어드니까 (죽은 사람은 다음 달 `inforce`에 없으니까). "한 번만" 은 청구 식의 특성이 아니라
**같은 율로 `inforce` 가 감쇠한다** 는 calibration 의 결과.

## 진단 — 자기 미진단 풀에서 탈퇴

진단도 사망처럼 "한 번"입니다 — 단, 빠지는 풀이 다릅니다. 진단 사건은 사람을
**계약(`inforce`)에서 빼내지 않습니다**. 그 사람은 계약에 남아 보험료를 내고
다른 보장도 그대로 작동합니다(1.3의 사건 표 — 진단은 탈퇴 '아니요'). 대신
빠지는 건 **그 담보만의 미진단(`undiagnosed`) 풀** — 한 번 진단받은 사람은
그 풀에서 빠져, 다음 달에 또 같은 진단을 받지 않습니다.

그래서 엔진은 보장마다 자기 `undiagnosed` 풀을 두고, **진단 발생률**로
청구하면서 동시에 그 풀을 `(1 - 진단발생률)` 로 축소합니다. 사망에서 *별도의*
사망탈퇴율이 계약 풀을 줄이던 자리를, 진단에서는 *같은* 진단 발생률이 자기
미진단 풀을 줄이는 것 — 탈퇴와 발생이 한 입력에 묶입니다.

```python
import numpy as np
import fastcashflow as fcf

# 진단 발생 함수 -- 월 1% 진단율의 연 환산
cancer_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 감쇠 없음
no_decr = lambda s, a, d: np.full(a.shape, 0.0)

basis = fcf.Basis(
    mortality_annual = no_decr,                                       # 보유계약 사망률 0 (감쇠 안 함)
    lapse_annual     = no_decr,                                       # 해지율 0 (해지 없음)
    discount_annual  = 0.0,                                           # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,                                          # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,                                           # 사망률 변동계수 0 (RA = 0 강제)
    coverages        = (fcf.CoverageRate("CANCER", cancer_fn),),      # 암진단 보장 1 종 (청구 rate = cancer_fn)
)
mp = fcf.ModelPoints.single(
    issue_age           = 40,           # 가입연령 40세
    sex                 = 0,            # 성별 (0=남, 1=여)
    benefits            = {0: 12_000},  # 0번 보장 (= CANCER) 의 진단 일시금 12,000
    premium             = 0,            # 월납 보험료 0 (보험료 cash flow 무시)
    term_months         = 3,            # 보험기간 3개월
    calculation_methods = {
        "CANCER": fcf.CalculationMethod.DIAGNOSIS,   # 코드 → 산출방법 매핑
    },
)
r = fcf.gmm.measure(mp, basis)

print(f"inforce       : {r.cashflows.inforce[0, :3]}")
print(f"morbidity_cf  : {r.cashflows.morbidity_cf[0, :3]}")           # 진단 cash flow
print(f"BEL[0]        : {float(r.bel[0]):.2f}")
print(f"cumulative 3m : {float(r.cashflows.morbidity_cf[0, :3].sum()):.2f}")
```

출력:

```
inforce       : [1. 1. 1.]
morbidity_cf  : [120.    118.8   117.612]
BEL[0]        : 356.41
cumulative 3m : 356.41
```

손계산:

| t | `inforce` | `undiagnosed` | rate × benefit | claim |
|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.01 × 12,000 | 120.00 |
| 1 | 1.0000 | 0.9900 | 0.01 × 12,000 | 118.80 |
| 2 | 1.0000 | 0.9801 | 0.01 × 12,000 | 117.61 |

**핵심**: `inforce` 는 안 줄어드는데 **`undiagnosed` 가 별도 풀로 매월 1%
씩 감쇠**. 청구 시계열이 사망과 정확히 같음 — 누적 ~356 으로 수렴. 빠지는
풀만 다를 뿐(계약 전체 vs 자기 미진단 풀), "한 번"이 되는 원리는 같습니다.

## 입원 — 어느 풀에서도 안 빠짐, 그래서 반복

입원은 사건이 **발생**하지만 사람을 어느 풀에서도 빼내지 않습니다 —
계약(`inforce`)도, 자기 하위 풀도 줄지 않습니다(1.3의 사건 표 — 입원은 탈퇴
'아니요'이고, 진단 같은 자기 미진단 풀도 없음). 빠지는 풀이 없으니 살아있는
한 매월 새로 청구됩니다.

toy 로 확인하려면 입원 발생률만 두고, 보유계약 사망률(`mortality_annual`)
= 0 으로 두어 사망 / 해지에 의한 `inforce` 감쇠까지 없앱니다 — 그러면 **어떤
풀도 어떤 식으로도 감쇠하지 않는** 순수한 반복이 드러납니다.

```python
import numpy as np
import fastcashflow as fcf

# 입원 발생 함수 -- 월 1% 발생률의 연 환산
inpatient_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 감쇠 없음 -- 사망 / 해지 모두 0
no_decr = lambda s, a, d: np.full(a.shape, 0.0)


# 산출기초
basis = fcf.Basis(
    mortality_annual = no_decr,             # 보유계약 사망률 0 (감쇠 안 함)
    lapse_annual     = no_decr,             # 해지율 0 (해지 없음)
    discount_annual  = 0.0,                 # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,                # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,                 # 사망률 변동계수 0 (RA = 0 강제)
    coverages        = (fcf.CoverageRate("INPATIENT", inpatient_fn),),
)
# 모델 포인트 (계약 하나)
mp = fcf.ModelPoints.single(
    issue_age           = 40,           # 가입연령 40세
    sex                 = 0,            # 성별 (0=남, 1=여)
    benefits            = {0: 12_000},  # 0번 보장 (= INPATIENT) 의 입원 1건당 12,000
    premium             = 0,            # 월납 보험료 0 (보험료 cash flow 무시)
    term_months         = 3,            # 보험기간 3개월
    calculation_methods = {
        "INPATIENT": fcf.CalculationMethod.MORBIDITY,   # 코드 → 산출방법 매핑
    },
)
r = fcf.gmm.measure(mp, basis)

print(f"inforce       : {r.cashflows.inforce[0, :3]}")
print(f"morbidity_cf  : {r.cashflows.morbidity_cf[0, :3]}")  # 입원 cash flow
print(f"BEL[0]        : {float(r.bel[0]):.2f}")
print(f"cumulative 3m : {float(r.cashflows.morbidity_cf[0, :3].sum()):.2f}")
```

출력:

```
inforce       : [1. 1. 1.]
morbidity_cf  : [120. 120. 120.]
BEL[0]        : 360.00
cumulative 3m : 360.00
```

손계산:

| t | `inforce` | rate × benefit | claim |
|---|---|---|---|
| 0 | 1.0000 | 0.01 × 12,000 | 120.00 |
| 1 | 1.0000 | 0.01 × 12,000 | 120.00 |
| 2 | 1.0000 | 0.01 × 12,000 | 120.00 |

**핵심**: 같은 식 `inforce × 발생률 × benefit` 인데, **빠지는 풀이 없어
매월 120 이 새로 발생**. "반복" 도 식의 특성이 아니라 **입원 발생률이 어떤
풀도 줄이지 않는다** 의 결과.

세 모드를 가른 건 단 하나 — **발생 사건이 어느 풀에서 사람을 빼내는가**.
사망은 계약 전체 풀(별도 사망탈퇴율로 축소), 진단은 자기 미진단 풀(진단
발생률로 축소), 입원은 아무 풀도 아님. 청구 식은 셋 다 같고, 이 한 가지가
시계열의 모양을 완전히 바꿉니다.

## 세 모드 나란히

| 보장 | `inforce` | 별도 풀 | 청구 시계열 | 누적 |
|---|---|---|---|---|
| 사망 (DEATH) | `1 → 0.99 → 0.9801` (감쇠 ✓) | 없음 | `120, 118.8, 117.61` | 356.41 |
| 진단 (DIAGNOSIS) | `1, 1, 1` (감쇠 ✗) | `undiagnosed: 1 → 0.99 → 0.9801` | `120, 118.8, 117.61` | 356.41 |
| 입원 (MORBIDITY) | `1, 1, 1` (감쇠 ✗) | 없음 | `120, 120, 120` | 360.00 |

사망 과 진단 의 청구 시계열이 글자 그대로 동일 함에도 메커니즘은 다르고 —
사망 의 "한 번" 은 계약 전체 `inforce` 가, 진단 의 "한 번" 은 자기
`undiagnosed` 풀이 각각 표현. 입원 은 어느 풀도 감쇠 안 하니 "반복" 이
자동입니다.

## `is_diagnosis` flag 의 진짜 의미

```python
import fastcashflow as fcf
fcf.gmm.trace(0, mp, basis)   # 위 DIAGNOSIS (CANCER) 예제의 mp / basis
```

trace 의 Coverages 노드에는 매 보장마다 `is_diagnosis` flag 가 표시됩니다:

```
├─ Coverages (rate-driven, n=1)
│   └─ 'CANCER'   method=DIAGNOSIS  risk=1  is_diagnosis=True   rate -> <callable>
```

`is_diagnosis` 의 진짜 의미는 **"한 번만 지급되는가"** 가 아닙니다. 정확한
의미는:

> "이 보장이 자기만의 별도 풀 을 갖고 있는가"

| 보장 종류 | `is_diagnosis` | 풀 자리 |
|---|---|---|
| DEATH | False | 공유 `inforce` (전 contract 가 같은 풀) |
| MORBIDITY | False | 공유 `inforce` (실제로 감쇠는 안 함) |
| DIAGNOSIS | True | per-coverage `undiagnosed` 풀 |

DEATH 와 MORBIDITY 가 둘 다 `is_diagnosis=False` 인 건 kernel 분기로 보면
같은 알고리즘 (A) 을 쓴다는 뜻. "DEATH 와 MORBIDITY 가 의미적으로 한 묶음"
이 아닙니다.

## `gmm.trace` 의 Undiagnosed share 노드

DIAGNOSIS 보장이 있는 mp 에 한해 trace 가 `undiagnosed` 풀의 시계열도
명시적으로 보여줍니다:

```
├─ Undiagnosed share (key months, per coverage)
│   └─ 'CANCER':
│       ├─ t=   0m: undiagnosed=1.000000
│       ├─ t=  12m: undiagnosed=0.886385
│       ├─ t=  60m: undiagnosed=0.547157
│       └─ ...
```

손계산 시 `undiagnosed(t) = (1 - q_monthly)^t` 와 직접 대조해 검증
가능합니다. 위 그림은 `q_monthly = 0.01` 인 보장기간이 긴 계약을 가정한
**예시 출력** 으로, `0.99^12 ≈ 0.886385` · `0.99^60 ≈ 0.547157` 과 맞습니다
(trace 가 보여주는 key month 는 보장기간에 따라 달라집니다 — 3개월 계약이면
t=0 · t=3 만 나옵니다).

DEATH-only / MORBIDITY-only 의 mp 에서는 이 노드 자체가 출력되지 않습니다.
"DIAGNOSIS 보장이 있을 때만 의미 있는 자리" 라는 트리 구조.

## 함정 — `mortality_annual` ↔ DEATH coverage rate 의 미스매치

위 사망 예제는 같은 `death_fn` 을 두 자리에 넣었습니다:

```text
basis = fcf.Basis(
    mortality_annual = death_fn,                              # 자리 1
    coverages        = (fcf.CoverageRate("DEATH", death_fn),) # 자리 2
)
```

엔진 안에서 이 두 자리는 **별개 양**:

| 자리 | 역할 |
|---|---|
| `mortality_annual` | `inforce` 감쇠 (사망으로 인한 보유계약 감소) |
| `coverages[DEATH].rate` | 사망 보장의 청구 rate |

손계산은 보통 두 양이 같다고 가정하고 결과 숫자 (356.41 등) 를 도출. 한
자리만 override 하면 두 양이 silent 어긋나 결과가 손계산과 안 맞습니다.

예: `mortality_annual = 0` 으로 두고 DEATH coverage rate 만 1% 로 두면:

```text
asmp_buggy = fcf.Basis(
    mortality_annual = no_decr,                               # 감쇠 0
    coverages        = (fcf.CoverageRate("DEATH", death_fn),) # 청구만 1%
)
```

→ DEATH 보장이 MORBIDITY 처럼 작동 (in-force 안 줄어드는데 청구는 매월
1% 발생). in-force가 안 줄어 같은 사람들에게 매월 사망보험금이 다시 지급됨.
누적 360 (위 MORBIDITY 결과와 같음) — 손계산의 356 과 어긋남.

이 함정의 방어 패턴은 [정기보험](../simple/term-life) 와 본 챕터의
모든 예제처럼 **`death_fn = lambda ...` 한 변수를 만들고 두 자리에 넘기는**
방식. 한 자리만 바꾸려 해도 두 자리가 한 변수를 공유하니 silent 어긋남이
구조적으로 차단됩니다.

`gmm.trace` 의 Rates 노드는 두 자리 (mortality_annual / DEATH(an)) 의 값을
**동일 행에 한꺼번에 표시** 하므로 어긋났을 때 한눈에 보입니다.

## 인접 레시피

- [CalculationMethod 결정 가이드](calculation-methods) — 담보별 산출방법의
  매 코드를 어느 산출방법으로 등록할지. 본 챕터는 그 산출방법이 엔진 안에서
  어떻게 다르게 동작하는지.
- [정기보험](../simple/term-life) — DEATH 만 사용하는 가장 단순한 사례.
- 사망 + 단순 진단 일시금 (작성 예정) — DEATH 와 DIAGNOSIS 의 결합. 본
  챕터의 두 메커니즘을 한 contract 에 동시 사용.
- [검증 패턴 — gmm.trace](../workflow/validation) — 본 챕터의 trace
  출력을 어떻게 사용하는지의 전반.

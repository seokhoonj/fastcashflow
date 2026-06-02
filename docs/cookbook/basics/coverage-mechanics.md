# 1.4 보장 청구 메커니즘

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **DEATH** / **MORBIDITY** / **DIAGNOSIS** 세 산출방식이 엔진 안에서
  어떻게 *서로 다른 알고리즘으로* 처리되는지
- "DEATH 는 한 번만 / MORBIDITY 는 여러 번 / DIAGNOSIS 는 한 번만"
  이라는 의미가 엔진 코드 어디에 표현되어 있는가
- 같은 `inforce × rate × benefit` 식이 DEATH 와 MORBIDITY 둘 다에
  적용되면서 결과는 정반대인 이유
- `is_diagnosis` flag 의 진짜 의미 — "한 번만 지급되는가" 가 아니라
  "이 보장이 자기만의 풀을 갖고 있는가"
- 손계산 / 엔진 출력 / `gmm.trace` 의 `undiagnosed` 트리가 어떻게
  서로를 검증하는가
```

각 `CalculationMethod` 값이 *`calculation_methods.csv` 에 적는 라벨* 이라면 —
`DEATH` 라고 적는 것 vs `DIAGNOSIS` 라고 적는 것의 차이는 어디서 정해지나?
그 선택은 [CalculationMethod 결정 가이드](calculation-methods) 의 자리이고,
본 챕터는 **엔진이 그 산출방식을 받아서 무엇을 다르게 하는가** 입니다.

이 챕터의 모든 예제는 동일한 toy 설정 — 월 사망/발생/진단율 1%, 보험금
12,000, 3 개월 — 위에서 산출방식만 바꿔 결과를 비교합니다. 한 페이지에 세 모드의
차이가 명료하게 드러나도록 의도된 셋업입니다.

## 청구 메커니즘 한눈에

엔진은 청구 처리에 *두 가지 알고리즘* 만 갖고 있습니다.

```{list-table}
:header-rows: 1
:widths: 22 18 60

* - 알고리즘
  - 산출방식
  - 청구 식
* - (A) `inforce × rate`
  - DEATH, MORBIDITY
  - 매월 `inforce[t] × rate[t] × benefit` 누적. 끝.
* - (B) `inforce × undiagnosed × rate`
  - DIAGNOSIS
  - 매월 `inforce[t] × undiagnosed[t] × rate[t] × benefit` 누적.
    `undiagnosed` 는 보장마다 자기 풀, 매월 `(1 - rate)` 로 감쇠.
```

`is_diagnosis` flag 는 정확히 이 (A)/(B) 분기 결정자입니다. False → 식 (A),
True → 식 (B).

"한 번 / 여러 번" 의 구분은 **이 식 안에 있지 않습니다**. 같은 식 (A) 가
DEATH 와 MORBIDITY 양쪽에 적용되는데도 한 자리는 "한 번만 청구되는 듯"
다른 자리는 "여러 번 청구되는 듯" 보이는 이유는, **`inforce` 자체가
무엇으로 감쇠하는가** 가 사용자 calibration 의 결과로 다르기 때문입니다.

아래 세 절이 같은 toy 데이터로 그 차이를 한 번에 보여줍니다.

## DEATH — 공유 `inforce` 풀이 자체 감쇠

월 사망률 1% 의 단일 사망보장. 엔진의 `mortality_annual` 이 `inforce` 를
같은 1% 로 감쇠시키니, **사망 사건이 곧 `inforce` 의 감소**.

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda s, a, d: np.full(d.shape, 0.0)

basis = fcf.Basis(
    mortality_annual = death_fn,                                # 보유계약 감쇠용 사망률 (위 death_fn)
    lapse_annual     = lapse_fn,                                # 해지율 (해지 없음)
    discount_annual  = 0.0,                                     # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,                                    # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,                                     # 사망률 변동계수 0 (RA = 0 강제)
    coverages        = (fcf.CoverageRate("DEATH", death_fn),),  # 사망 보장 1 종 (청구 rate = 같은 death_fn)
)
mp = fcf.ModelPoints.single(
    issue_age     = 40,           # 가입연령 40세
    sex           = 0,            # 성별 (0=남, 1=여)
    benefits      = {0: 12_000},  # 0번 보장 (= DEATH) 의 보험금 12,000
    level_premium = 0,            # 월납 보험료 0 (보험료 cash flow 무시)
    term_months   = 3,            # 보험기간 3개월
    calculation_methods = {
        "DEATH": fcf.CalculationMethod.DEATH,   # 코드 → 산출방식 매핑
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
줄어드니까 (한 사람당 한 번 죽음). "한 번만" 은 청구 식의 특성이 아니라
**같은 율로 `inforce` 가 감쇠한다** 는 calibration 의 결과.

## MORBIDITY — 풀 없음, 반복 발생

같은 율을 입원 보장 (MORBIDITY) 에 두면. 단, `mortality_annual = 0` 으로
두어 사망 / 해지에 의한 `inforce` 감쇠도 없게 — 입원 자체가 `inforce` 를
줄이지 않으니, **풀 자체가 어떤 식으로도 감쇠 안 함**.

```python
import numpy as np
import fastcashflow as fcf

# 입원 발생 함수 -- 월 1% 발생률의 연 환산
inpatient_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 감쇠 없음 -- 사망 / 해지 모두 0
no_decr = lambda s, a, d: np.full(a.shape, 0.0)


# 산출기초
basis = fcf.Basis(
    mortality_annual = no_decr,             # 보유계약 감쇠율 0 (감쇠 안 함)
    lapse_annual     = no_decr,             # 해지율 0 (해지 없음)
    discount_annual  = 0.0,                 # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,                # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,                 # 사망률 변동계수 0 (RA = 0 강제)
    coverages        = (fcf.CoverageRate("INPATIENT", inpatient_fn),),
)
# 모델 포인트 (계약 하나)
mp = fcf.ModelPoints.single(
    issue_age     = 40,           # 가입연령 40세
    sex           = 0,            # 성별 (0=남, 1=여)
    benefits      = {0: 12_000},  # 0번 보장 (= INPATIENT) 의 입원 1건당 12,000
    level_premium = 0,            # 월납 보험료 0 (보험료 cash flow 무시)
    term_months   = 3,            # 보험기간 3개월
    calculation_methods = {
        "INPATIENT": fcf.CalculationMethod.MORBIDITY,   # 코드 → 산출방식 매핑
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

**핵심**: 같은 식 `inforce × rate × benefit` 인데, **`inforce` 가 안
줄어드니까 매월 120 이 새로 발생**. "여러 번" 도 식의 특성이 아니라
*`inforce` 가 그 rate 로 감쇠하지 않는다* 의 결과.

DEATH 예제와 MORBIDITY 예제의 차이는 단 하나 — `mortality_annual` 이
청구 rate 와 *같은 값을 쓰는가 0 을 쓰는가*. 그 한 줄 차이가 청구 시계열의
형태를 완전히 바꿉니다.

## DIAGNOSIS — per-coverage `undiagnosed` 풀

진단 보장은 `inforce` 가 안 줄어드는데도 "한 번만" 지급해야 합니다 — 한
번 진단 받은 사람이 다음 달에 또 같은 진단을 받지 않으니까. 식 (A) 는 이걸
못 표현하니, 엔진은 *별도 알고리즘 (B)* 를 씁니다 — 보장마다 자기만의
`undiagnosed` 풀.

```python
import numpy as np
import fastcashflow as fcf

# 진단 발생 함수 -- 월 1% 진단율의 연 환산
cancer_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 감쇠 없음
no_decr = lambda s, a, d: np.full(a.shape, 0.0)

basis = fcf.Basis(
    mortality_annual = no_decr,                                       # 보유계약 감쇠율 0 (감쇠 안 함)
    lapse_annual     = no_decr,                                       # 해지율 0 (해지 없음)
    discount_annual  = 0.0,                                           # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,                                          # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,                                           # 사망률 변동계수 0 (RA = 0 강제)
    coverages        = (fcf.CoverageRate("CANCER", cancer_fn),),      # 암 진단 보장 1 종 (청구 rate = cancer_fn)
)
mp = fcf.ModelPoints.single(
    issue_age     = 40,           # 가입연령 40세
    sex           = 0,            # 성별 (0=남, 1=여)
    benefits      = {0: 12_000},  # 0번 보장 (= CANCER) 의 진단 일시금 12,000
    level_premium = 0,            # 월납 보험료 0 (보험료 cash flow 무시)
    term_months   = 3,            # 보험기간 3개월
    calculation_methods = {
        "CANCER": fcf.CalculationMethod.DIAGNOSIS,   # 코드 → 산출방식 매핑
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

**핵심**: `inforce` 는 안 줄어드는데 **`undiagnosed` 가 별도 풀로 매월
1% 씩 감쇠**. 청구 시계열이 DEATH 와 정확히 같음 — 누적 ~356 으로 수렴.
하지만 메커니즘은 다름.

## 세 모드 나란히

| 산출방식 | `inforce` | 별도 풀 | 청구 시계열 | 누적 |
|---|---|---|---|---|
| DEATH | `1 → 0.99 → 0.9801` (감쇠 ✓) | 없음 | `120, 118.8, 117.61` | 356.41 |
| MORBIDITY | `1, 1, 1` (감쇠 ✗) | 없음 | `120, 120, 120` | 360.00 |
| DIAGNOSIS | `1, 1, 1` (감쇠 ✗) | `undiagnosed: 1 → 0.99 → 0.9801` | `120, 118.8, 117.61` | 356.41 |

DEATH 와 DIAGNOSIS 의 청구 시계열이 *글자 그대로 동일* 함에도 메커니즘은
다르고 — DEATH 의 "한 번" 은 `inforce` 가, DIAGNOSIS 의 "한 번" 은
`undiagnosed` 풀이 각각 표현. MORBIDITY 는 어느 풀도 감쇠 안 하니 "여러
번" 이 자동.

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

> "이 보장이 *자기만의 별도 풀* 을 갖고 있는가"

| 보장 종류 | `is_diagnosis` | 풀 자리 |
|---|---|---|
| DEATH | False | 공유 `inforce` (전 contract 가 같은 풀) |
| MORBIDITY | False | 공유 `inforce` (실제로 감쇠는 안 함) |
| DIAGNOSIS | True | per-coverage `undiagnosed` 풀 |

DEATH 와 MORBIDITY 가 둘 다 `is_diagnosis=False` 인 건 *kernel 분기로 보면*
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
가능합니다. 위 예제는 월 `q_monthly = 0.01` 이라 t=12 에서 `0.99^12 ≈
0.886385` — trace 와 정확히 일치.

DEATH-only / MORBIDITY-only 의 mp 에서는 이 노드 자체가 출력되지 않습니다.
"DIAGNOSIS 보장이 있을 때만 의미 있는 자리" 라는 트리 구조.

## 함정 — `mortality_annual` ↔ DEATH coverage rate 의 미스매치

위 DEATH 예제는 *같은 `death_fn` 을 두 자리에 넣었습니다*:

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
| `coverages[DEATH].rate` | 사망 보장의 *청구* rate |

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
1% 발생). 같은 사람이 매월 죽고 또 죽음. 누적 360 (위 MORBIDITY 결과와
같음) — *손계산의 356 과 어긋남*.

이 함정의 방어 패턴은 [정기보험](../simple/term-life) 와 본 챕터의
모든 예제처럼 **`death_fn = lambda ...` 한 변수를 만들고 두 자리에 넘기는**
방식. 한 자리만 바꾸려 해도 두 자리가 한 변수를 공유하니 silent 어긋남이
구조적으로 차단됩니다.

`gmm.trace` 의 Rates 노드는 두 자리 (mortality_annual / DEATH(an)) 의 값을
**동일 행에 한꺼번에 표시** 하므로 어긋났을 때 한눈에 보입니다.

## 인접 레시피

- [CalculationMethod 결정 가이드](calculation-methods) — 담보별 산출방식의
  매 코드를 어느 산출방식으로 등록할지. 본 챕터는 그 산출방식이 *엔진 안에서*
  어떻게 다르게 동작하는지.
- [정기보험](../simple/term-life) — DEATH 만 사용하는 가장 단순한 사례.
- 사망 + 단순 진단 일시금 (작성 예정) — DEATH 와 DIAGNOSIS 의 결합. 본
  챕터의 두 메커니즘을 한 contract 에 동시 사용.
- [검증 패턴 — gmm.trace](../workflow/validation) — 본 챕터의 trace
  출력을 어떻게 사용하는지의 전반.

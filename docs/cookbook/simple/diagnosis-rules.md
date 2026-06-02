# 2.3 다종 진단 + 면책 / 감액

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 진단 담보의 **면책기간** (waiting) 과 **감액기간** (reduction) 을
  `coverages` 파일의 세 컬럼으로 거는 법
- 룰이 *지급* 만 누르고 *미진단 풀 감쇠* 는 그대로라는 점
- 한 계약에 진단 담보를 여러 개 (암 / 뇌혈관) 얹고 각자 다른 룰을 주는 법
```

면책 / 감액은 **상태가 아니라 담보별 룰** 입니다 (한 담보의 시간축 위
지급률 곡선) — 상태 추적이 없는 단순 정액형에도 그대로 적용됩니다. 상태
축 (납입면제 등) 과는 직교하며 한 계약에 공존할 수 있습니다 (3장).

## 상품 소개 — 면책기간 / 감액기간

진단비 담보 (암 / 뇌혈관 / 심혈관 등) 는 보통 가입 직후의 도덕적 해이 ·
역선택을 막으려고 두 가지 시간 제한을 둡니다:

- **면책기간** — 가입 후 일정 기간은 진단받아도 **지급하지 않음**. 암
  진단비는 보통 90일 (가입 후 3개월).
- **감액기간** — 면책 이후 일정 기간은 **일부만 지급** (보통 50%).
  의도적 사고 / 조기 청구를 억제.

즉 한 진단 담보의 지급률은 시간에 따라 `0% → reduction_factor% → 100%` 로
계단식으로 올라갑니다.

## 모델링 매핑 — coverages 파일의 세 컬럼

면책 / 감액은 `coverages` 파일의 담보 행에 세 컬럼으로 붙습니다.
세 컬럼 모두 **가입 시점 (t=0) 기준** 이라 별도 `*_start` 컬럼은 없습니다.

```{list-table}
:header-rows: 1
:widths: 22 18 60

* - 컬럼
  - 단위
  - 적용 구간
* - `waiting`
  - 정수 (개월)
  - `[0, waiting)` 동안 **미지급** (면책기간)
* - `reduction_end`
  - 정수 (개월)
  - `[0, reduction_end)` 동안 `reduction_factor` 비율로 지급 (감액기간)
* - `reduction_factor`
  - 실수 (0..1)
  - 감액기간 중 지급 비율 (보통 0.5)
```

월 `t` 의 지급률:

| 구간 | 지급률 |
|---|---|
| `t < waiting` | 0% (면책) |
| `waiting <= t < reduction_end` | `reduction_factor` (감액) |
| `t >= reduction_end` | 100% (정상) |

함께 두면 `waiting <= reduction_end` 가 일반적입니다 (예: `waiting=3`,
`reduction_end=24`, `reduction_factor=0.5` → 첫 3개월 면책, 4~24개월 50%,
25개월부터 100%).

## 한 계약 — 손계산과 엔진

면책 / 감액의 효과를 또렷이 보려고 작은 toy 를 씁니다. 암 진단 담보 하나에
면책 1개월 + 감액(0.5) 3개월까지를 겁니다.

```{admonition} 예제 설정
:class: note

- 보험기간 4개월, 사망 / 해지 없음 (진단 풀에 집중), 할인 0
- 암 월 진단율 10%, 진단금 100,000
- 면책 1개월 (`waiting=1`), 감액 3개월까지 (`reduction_end=3`),
  감액 비율 0.5 (`reduction_factor=0.5`)
```

```python
import numpy as np
import polars as pl
import fastcashflow as fcf

# 암 진단율 함수 -- 월 10% 의 연 환산 (평탄)
cancer_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.10) ** 12)
no_decr   = lambda s, a, d: np.full(a.shape, 0.0)

# 산출기초
basis = fcf.Basis(
    mortality_annual = no_decr,    # 보유계약 감쇠율 0 (진단 풀에 집중)
    lapse_annual     = no_decr,    # 해지율 0
    discount_annual  = 0.0,        # 연 할인율 0 (검증 단순화)
    ra_confidence    = 0.75,       # 위험조정 신뢰수준 75%
    mortality_cv     = 0.0,        # 사망률 변동계수 0
    morbidity_cv     = 0.0,        # 진단율 변동계수 0
    coverages        = (
        fcf.CoverageRate("CANCER", cancer_fn),  # 암 진단 1종 (청구 rate = cancer_fn)
    ),
)

# 입력 파일 -- coverages 에 면책/감액 세 컬럼
pl.DataFrame({
    "mp_id":         ["P001"],   # 계약 식별자
    "issue_age":     [40],       # 가입연령 40세
    "term_months":   [4],        # 보험기간 4개월
    "level_premium": [0],        # 월납 보험료 0 (진단 풀에 집중)
}).write_csv("policies.csv")

pl.DataFrame({
    "mp_id":            ["P001"],     # 어느 계약의 담보인지
    "coverage_code":    ["CANCER"],   # 담보 코드
    "amount":           [100_000],    # 진단금 100,000
    "waiting":          [1],          # 면책 1개월
    "reduction_end":    [3],          # 감액 3개월까지
    "reduction_factor": [0.5],        # 감액기간 중 50% 지급
}).write_csv("coverages.csv")

mp = fcf.read_model_points(
    "policies.csv",                                  # 계약 spec 파일
    coverages="coverages.csv",                       # 담보 + 면책/감액 룰
    calculation_methods={"CANCER": fcf.CalculationMethod.DIAGNOSIS},
)

m = fcf.gmm.measure(mp, basis)
print(f"morbidity_cf = {m.cashflows.morbidity_cf[0, :4]}")  # 진단 cash flow
print(f"BEL          = {m.bel[0]:.2f}")                     # 최선추정부채
```

출력:

```
morbidity_cf = [   0.   4500.   4050.   7290.]
BEL          = 15840.00
```

손계산. 미진단 풀은 매월 진단율 10% 로 감쇠하고 (면책기간에도 감쇠),
지급률만 면책 / 감액 룰을 따릅니다:

| t | 미진단 풀 | 지급률 | 진단 cash flow (풀 × 10% × 100,000 × 지급률) |
|---|---|---|---|
| 0 | 1.000000 | 0% (면책) |       0.00 |
| 1 | 0.900000 | 50% (감액) |   4,500.00 |
| 2 | 0.810000 | 50% (감액) |   4,050.00 |
| 3 | 0.729000 | 100% |        7,290.00 |

- BEL = 0 + 4,500 + 4,050 + 7,290 = **15,840** (할인 0, 보험료 0)
- 룰이 없으면 cash flow 는 `[10,000, 9,000, 8,100, 7,290]` 이고 BEL =
  34,390. 면책 + 감액이 BEL을 **34,390 → 15,840** 으로 낮춥니다.

```{admonition} 룰은 지급만 누르고 풀은 그대로
:class: note

t=3 의 cash flow 7,290 은 룰이 있든 없든 같습니다 — 미진단 풀이 두 경우
*동일하게* 감쇠하기 때문 (`0.9^3 = 0.729`). 면책 / 감액은 **지급률** 만
바꿀 뿐, 진단으로 풀이 줄어드는 동학은 건드리지 않습니다. 즉 면책기간에
진단받은 사람은 *지급은 못 받아도 풀에서는 빠져나갑니다* — 같은 진단으로
두 번 청구할 수 없으니 실무와 일치.
```

## 결과 읽기

`morbidity_cf` 의 앞부분이 룰의 계단을 그대로 보여줍니다 — 면책기간은 0,
감액기간은 절반, 그 이후 정상. 이 곡선의 현재가치 합이 BEL이고, 면책 /
감액이 길수록 / 깊을수록 BEL이 낮아집니다 (보험사 지급이 줄어드니까).

## 변형 — 다종 진단 (각자 다른 룰)

한 계약에 암 + 뇌혈관 진단을 함께 얹되, 암만 면책 / 감액을 겁니다.
`coverages` 에 행을 하나 더하고 (뇌혈관), 각 행이 자기 룰 컬럼을 가집니다:

```python
cerebral_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.05) ** 12)

basis = fcf.Basis(
    mortality_annual = no_decr,
    lapse_annual     = no_decr,
    discount_annual  = 0.0,
    ra_confidence    = 0.75,
    mortality_cv     = 0.0,
    morbidity_cv     = 0.0,
    coverages        = (
        fcf.CoverageRate("CANCER",   cancer_fn),    # 암 진단
        fcf.CoverageRate("CEREBRAL", cerebral_fn),  # 뇌혈관 진단
    ),
)

pl.DataFrame({
    "mp_id":            ["P001",    "P001"],
    "coverage_code":    ["CANCER",  "CEREBRAL"],
    "amount":           [100_000,   200_000],
    "waiting":          [1,         0],     # 암만 면책 1개월
    "reduction_end":    [3,         0],     # 암만 감액 3개월까지
    "reduction_factor": [0.5,       1.0],   # 뇌혈관은 룰 없음 (1.0)
}).write_csv("coverages.csv")

mp = fcf.read_model_points(
    "policies.csv",
    coverages="coverages.csv",
    calculation_methods={"CANCER":   fcf.CalculationMethod.DIAGNOSIS,
                         "CEREBRAL": fcf.CalculationMethod.DIAGNOSIS},
)
m = fcf.gmm.measure(mp, basis)
print(f"morbidity_cf = {m.cashflows.morbidity_cf[0, :4]}")
```

출력:

```
morbidity_cf = [10000.    14000.    13075.    15863.75]
```

`morbidity_cf` 는 두 담보의 **합** 입니다 — 암 (면책/감액) + 뇌혈관 (룰
없음). 각 진단 담보는 자기만의 미진단 풀 을 가지므로 (암 진단을 받아도
뇌혈관 풀은 안 줄어듦), 룰도 풀도 담보마다 독립입니다.

## 함정

### 함정 1 — `reduction_factor` 만 주고 `reduction_end` 생략

`reduction_factor` 를 0.5 로 줬는데 `reduction_end` 가 없으면 (기본 0),
감액이 `t < 0` 구간 — 즉 영영 발동 안 함 — 이라 reader 가 거부합니다.
감액을 쓰려면 `reduction_end` 를 반드시 함께.

### 함정 2 — 면책기간에 풀이 안 줄어든다고 가정

면책기간에도 미진단 풀은 진단율로 감쇠합니다. "면책 = 그 기간 진단이
없는 것" 이 아니라 "진단은 일어나지만 지급만 안 하는 것". 위 손계산의
t=3 cash flow 가 룰 유무와 무관하게 같은 이유.

### 함정 3 — 룰을 사망 / 입원 담보에 그대로 적용

면책 / 감액 컬럼은 모든 담보 type 에 붙지만, 의미는 type 마다 다릅니다.
진단 (DIAGNOSIS) 은 풀 기반이라 위 설명대로지만, 사망 (DEATH) / 입원
(MORBIDITY) 은 면책기간 동안 그 달의 지급만 0 / 감액됩니다 (풀 개념 없음).
담보 type 에 맞는 룰인지 확인.

## 인접 레시피

- [2.2 사망 + 단순 진단 일시금](death-diagnosis) — 면책 / 감액 없는 진단
  담보. 본 챕터의 출발점.
- [1.4 보장 청구 메커니즘](../basics/coverage-mechanics) — DIAGNOSIS 의
  `undiagnosed` 풀 동학. 면책 / 감액이 풀을 안 건드리는 이유의 근거.
- [3.1 보험료 납입면제 (waiver)](../markov/waiver) — 상태 축. 면책 / 감액
  (담보 룰 축) 과 직교하며 한 계약에 공존 가능.
- [검증 패턴](../workflow/validation) — `gmm.trace` 로 담보별 면책 / 감액
  적용 구간을 한 줄씩 확인.

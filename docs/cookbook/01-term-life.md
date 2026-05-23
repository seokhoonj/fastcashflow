# 1장. 정기보험 평가

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **정기보험** 을 fastcashflow 로 평가하는 가장 짧은 코드
- `Assumptions` / `ModelPoints` 가 무엇을 담는지
- `measure()` 와 `value()` 의 차이, 언제 어느 것을 쓰나
- 결과의 BEL / RA / CSM 값이 의미하는 바와 부호 규약
- 흔한 실수와 그 자리에서 잡는 방법

이 챕터만 봐도 정기보험 평가는 끝까지 갈 수 있도록 만들었습니다.
다른 자료로 점프할 필요 없음.
```

## 1.1 상품 소개 — 한국 시장의 정기보험

**정기보험 (term life insurance)** 은 미리 정한 보험기간 동안 피보험자가
사망하면 사망보험금을 지급하는, 가장 단순한 형태의 생명보험입니다.
기간이 지나 만기 도달 시에는 어떤 환급도 없습니다.

한국 시장의 전형적 모양:

- 가입연령 25-65세
- 보험기간 10년 / 15년 / 20년 / 60세까지 / 80세까지
- 보험료 납입기간 = 보험기간 (전기납), 또는 짧게 (단기납 — 예: 20년
  만기 10년납)
- 보험료는 매월 동일액 (level premium) 납입
- 보험금은 일정 금액 (level death benefit), 또는 가입 후 일정 기간만
  감액 (감액기간 — 본 챕터에선 단순화 위해 생략)
- 사망 이외에 만기시 어떤 지급도 없음 (만기환급금 0)

이 챕터는 **단순 정기보험** 만 다룹니다. 보험료 납입면제 (waiver) /
다종 진단담보 / 만기환급금 결합 같은 변형은 2장 이후에서 다룹니다.

## 1.2 모델링 매핑 — 상품 → fastcashflow API

| 상품 mechanic | fastcashflow API | 비고 |
|---|---|---|
| 피보험자 가입연령 | `ModelPoints(issue_age=...)` | 정수 (만 나이) |
| 피보험자 성별 | `ModelPoints(sex=...)` | 0 = 남, 1 = 여 |
| 보험기간 (개월) | `ModelPoints(term_months=...)` | 60개월 = 5년 |
| 사망보험금 | `ModelPoints(death_benefit=...)` | 원 단위. 1억 = `100_000_000` |
| 월 보험료 | `ModelPoints(level_premium=...)` | 매월 동일액 |
| 사망률 | `Assumptions(mortality_annual=함수)` | (성별, 가입연령, 경과년수) → 연 사망률 |
| 해지율 | `Assumptions(lapse_annual=함수)` | 같은 시그니처. 해지 없으면 0 |
| 할인율 | `Assumptions(discount_annual=...)` | 연 단위 |
| 신사업비 | `Assumptions(expense_acquisition=...)` | 가입 시 1회 |
| 유지비 | `Assumptions(expense_maintenance_annual=...)` | 연 단위, 인플레이션 적용 |
| 위험조정 신뢰수준 | `Assumptions(ra_confidence=...)` | 0.75 (75%) 등 |
| 사망률 변동계수 | `Assumptions(mortality_cv=...)` | RA 계산에 사용 |

**중요한 점**: rate (사망률 / 해지율) 는 **숫자가 아니라 함수**로 줍니다.
회사의 경험률 테이블이 (성별 × 가입연령 × 경과년수) 의 함수이기 때문.
간단한 상수도 함수 형태로 감쌉니다.

## 1.3 최소 작동 예제

다음 코드는 그대로 복사해서 실행하면 됩니다. 40세 남성, 사망보험금 1억
원, 월 보험료 7만원, 보험기간 10년 (120개월) 의 정기보험 1건을 평가:

```python
import numpy as np
import fastcashflow as fcf

# 가정 -- 회사의 best estimate 가정
assumptions = fcf.Assumptions(
    # 사망률: 성별 / 가입연령 / 경과년수 의 함수. 여기선 단순화해서
    # 모두 0.001 (연 0.1%) 로 설정.
    mortality_annual=lambda sex, issue_age, duration: np.full(
        issue_age.shape, 0.001,
    ),
    # 해지율: 연 1% 가정.
    lapse_annual=lambda sex, issue_age, duration: np.full(
        duration.shape, 0.01,
    ),
    # 할인율 (연): 3% — 한국 보험사가 IFRS 17 평가에 흔히 쓰는 수준.
    discount_annual=0.03,
    # 사업비
    expense_acquisition=300_000.0,        # 가입 시 1회
    expense_maintenance_annual=60_000.0,  # 연 6만원
    expense_inflation=0.02,                # 연 2% 인상
    # 위험조정 (RA)
    ra_confidence=0.75,                    # 75% 신뢰수준
    mortality_cv=0.10,                     # 사망위험 변동계수 10%
)

# 모델포인트 (계약 한 건)
model_points = fcf.ModelPoints.single(
    issue_age=40,
    death_benefit=100_000_000,    # 1억원
    level_premium=70_000,         # 월 7만원
    term_months=120,              # 10년
)

# 측정 -- 두 방법
detail = fcf.measure(model_points, assumptions)
fast = fcf.value(model_points, assumptions)

# 결과 출력 (시점 0 = 가입 시점)
print(f"BEL (최선추정부채):     {detail.bel[0, 0]:>15,.0f}")
print(f"RA  (위험조정):           {detail.ra[0, 0]:>15,.0f}")
print(f"CSM (보험계약마진):      {detail.csm[0, 0]:>15,.0f}")
print(f"손실요소:                {detail.loss_component[0]:>15,.0f}")
print()
print(f"fast 경로 BEL: {fast.bel[0]:>15,.0f}   (detail 과 동일)")
```

실행하면 다음과 같은 출력이 나옵니다 (가정 그대로 사용 시):

```
BEL (최선추정부채):       -5,251,566
RA  (위험조정):                55,485
CSM (보험계약마진):         5,196,081
손실요소:                          0

fast 경로 BEL:        -5,251,566   (detail 과 동일)
```

코드 한 번 돌리면 BEL / RA / CSM 의 시점 0 값을 얻습니다.

## 1.4 결과 해석 — 숫자가 무엇을 말하는가

### 부호 규약 (sign convention)

fastcashflow 는 **부채 관점, 유출 양수** (outflow-positive) 규약을 씁니다:

- **유입** (insurer 가 받는 돈, 보험료) — 부채를 **감소** 시킴
- **유출** (insurer 가 내는 돈, 사망보험금 + 사업비) — 부채를 **증가** 시킴
- 따라서 `BEL = PV(claims) + PV(expenses) - PV(premiums)`

### BEL = -5,251,566 의 의미

BEL 이 **음수** 라는 것은 "예상 미래 보험료 유입" 이 "예상 미래 사망보험금
+ 사업비 유출" 보다 크다는 뜻. 즉 보험사 입장에서 **이익이 나는 계약**.

만약 BEL 이 양수라면 "유출 > 유입" 으로 보험사가 손실을 보는 계약이고,
그 차이는 **손실요소** 로 즉시 인식 (IFRS 17 Sec. 47).

### RA = 55,485 — 위험조정

RA 는 미래 사망률 / 비용 / 해지 의 **불확실성** 에 대해 보험사가 받는
보상. 75% 신뢰수준이면 "BEL + RA" 가 75% 백분위 부채 추정치에 해당.

```{admonition} 확인 포인트
:class: note

RA 가 작으면 BEL 의 불확실성이 작다는 뜻. mortality_cv 를 0.10 → 0.50
으로 바꾸면 RA 가 5배 커집니다. 한 번 실험해보세요 — 위 예제의
`mortality_cv=0.10` 만 바꾸고 실행.
```

### CSM = 5,196,081 — 보험계약마진

CSM 은 IFRS 17 의 핵심 개념. 계약 가입 시점에 "이익이 날 거다" 라고
인식한 부분을 **미래에 걸쳐 분산해서** 손익으로 전환하기 위한 buffer.

- 가입 시점 (t=0): `CSM₀ = max(0, -FCF)` where `FCF = BEL + RA`
- 매 기간 이자 부리 + 보장단위 비례 상각
- 이익이 나는 계약 (FCF < 0) → CSM > 0, 손실요소 = 0
- 손실이 나는 계약 (FCF > 0) → CSM = 0, 손실요소 = FCF

위 예제는 이익 계약이므로 `CSM = 5,196,081`, `loss_component = 0`.

### measure() 와 value() 의 차이

| | `measure()` | `value()` |
|---|---|---|
| 출력 | 시간 trajectory 전체 — BEL/RA/CSM 의 매월 값 + 현금흐름 6갈래 | 시점 0 의 4개 숫자만 |
| 용도 | 상세 검증 / 변동분석 / 시각화 / 보고용 | 대량 portfolio 평가, 민감도, 1M+ 계약 |
| 메모리 | 1M MP × 120개월 ≈ 9GB | 1M MP ≈ 32MB |
| 속도 (1M MP) | 수 초 | 80-300 ms |

**규칙**: 100계약 이하 검토는 `measure()`, 대량 portfolio 는 `value()`.
두 결과는 시점 0 에서 **수치적으로 동일** (parity test 가 자동 검증).

## 1.5 변형 — 회사 / 채널 / 상품 차이

### 가입연령에 따라 사망률 다르게

```python
def mortality_by_age(sex, issue_age, duration):
    # 가입연령에 따른 base 사망률 — 40세에 0.001, 60세에 0.005
    base = 0.001 + 0.0002 * np.maximum(issue_age - 40, 0)
    # 경과년수에 따른 증가
    return base * (1 + 0.05 * duration)

assumptions = fcf.Assumptions(
    mortality_annual=mortality_by_age,
    # 나머지는 동일
    ...
)
```

### 성별에 따라 다르게

```python
def mortality_by_sex(sex, issue_age, duration):
    # sex=0 남자, sex=1 여자. 여자가 더 낮은 사망률
    base = np.where(sex == 0, 0.001, 0.0007)
    return np.broadcast_to(base, issue_age.shape).copy()
```

### 여러 계약 동시에 (portfolio)

`ModelPoints.single` 대신 `ModelPoints` 로 배열 직접:

```python
n_contracts = 1000
rng = np.random.default_rng(42)

portfolio = fcf.ModelPoints(
    issue_age=rng.integers(25, 60, n_contracts),    # 25-60세 랜덤
    sex=rng.integers(0, 2, n_contracts),            # 0 또는 1
    death_benefit=rng.integers(10, 100, n_contracts) * 1_000_000,  # 1억-10억
    level_premium=rng.integers(3, 15, n_contracts) * 10_000,        # 3만-15만
    term_months=np.full(n_contracts, 120),                          # 모두 10년
)

result = fcf.value(portfolio, assumptions)
# 시점 0 의 BEL 등이 (n_contracts,) shape 배열로 나옴
print(f"포트폴리오 BEL 합계: {result.bel.sum():>15,.0f}")
print(f"평균 계약 BEL:        {result.bel.mean():>15,.0f}")
print(f"손실 계약 개수:        {(result.loss_component > 0).sum()}")
```

### 보험료 납입기간 단기납 (보장기간 ≠ 납입기간)

10년 만기, 5년만 보험료 납입:

```python
mp = fcf.ModelPoints.single(
    issue_age=40, death_benefit=100_000_000,
    level_premium=140_000,            # 5년만 내므로 더 큰 금액
    term_months=120,                  # 보장 10년
    premium_term_months=60,           # 납입 5년
)
```

### 보험료 frequency — 분기납 / 반기납 / 연납

월납 외에:

```python
mp = fcf.ModelPoints.single(
    issue_age=40, death_benefit=100_000_000,
    level_premium=70_000,                  # 매 분기 7만원
    term_months=120,
    premium_frequency_months=3,            # 분기납 (3개월에 한 번)
)
```

`premium_frequency_months=12` 이면 연납, `=6` 이면 반기납.

## 1.6 함정 — 흔한 실수와 잡는 방법

### 함정 1 — Rate 를 숫자로 직접 줌

```python
# ✗ 안 됨 -- mortality_annual 은 함수여야
assumptions = fcf.Assumptions(
    mortality_annual=0.001,   # TypeError
    ...
)

# ✓ 함수로 감싸기
assumptions = fcf.Assumptions(
    mortality_annual=lambda sex, age, dur: np.full(dur.shape, 0.001),
    ...
)
```

### 함정 2 — 함수 인자 개수 틀림

`mortality_annual` / `lapse_annual` 은 **3 인자** (sex, issue_age,
duration). 일부 옛 자료가 1-2 인자만 보여주지만 항상 3 인자.

```python
# ✗ 안 됨 -- 2 인자
lapse_annual=lambda issue_age, duration: ...   # 호출 시 에러

# ✓ 3 인자
lapse_annual=lambda sex, issue_age, duration: ...
```

### 함정 3 — rate 의 단위 혼동 (연 vs 월)

`mortality_annual` 은 **연 사망률**. 0.01 은 1% 연 사망률 (=대략 월
0.083%). 엔진이 내부에서 constant-force 방식으로 월율로 환산합니다.

회사 경험률 표가 월율이면 연율로 변환해서 입력:

```python
def annual_from_monthly(q_monthly):
    return 1.0 - (1.0 - q_monthly) ** 12

monthly_q = 0.001
annual_q = annual_from_monthly(monthly_q)   # ≈ 0.01194
```

### 함정 4 — 함수가 항상 같은 shape 의 배열을 돌려줘야

```python
# ✗ 안 됨 -- 스칼라 돌려줌
mortality_annual=lambda sex, age, dur: 0.001    # shape mismatch

# ✓ 입력과 같은 shape 의 배열
mortality_annual=lambda sex, age, dur: np.full(age.shape, 0.001)
```

`np.full(age.shape, value)` 또는 `np.full(dur.shape, value)` 패턴
사용. 일반적으로 **세 인자 중 어느 하나의 shape** 를 따라가면 됩니다 (셋 다 같은 shape).

### 함정 5 — 음수 BEL 을 보고 놀람

음수 BEL 은 "이익 계약" 의미. 정상입니다. 손실 계약은 BEL 양수 + CSM 0
+ loss_component 양수의 조합. 신호 패턴:

| BEL | CSM | loss_component | 의미 |
|---|---|---|---|
| 음수 | 양수 | 0 | 이익 계약 — 정상 |
| 0 | 0 | 0 | 손익분기 계약 |
| 양수 | 0 | 양수 | 손실 계약 (onerous) — 즉시 손실 인식 |

### 함정 6 — `level_premium` 인지 `monthly_premium` 인지

이전 API 에선 `monthly_premium` 이라는 이름이 사용됐으나 현재는
`level_premium` (level = "매 회 동일액") 으로 통일됨. 옛 자료의
`monthly_premium=` 을 보면 `level_premium=` 으로 바꿔서 사용.

### 검증 — 손계산 한 번

신뢰성 빌드업의 가장 쉬운 방법은 **2개월 계약 손계산**:

```python
# 가입 후 2개월, 사망률 1% 월, 사망보험금 12,000, 보험료 100, 할인 0%
mp = fcf.ModelPoints.single(
    issue_age=40, death_benefit=12_000,
    level_premium=100, term_months=2,
)
asmp = fcf.Assumptions(
    mortality_annual=lambda s, a, d: np.full(a.shape, 1 - (1-0.01)**12),
    lapse_annual=lambda s, a, d: np.full(d.shape, 0.0),
    discount_annual=0.0,
    expense_acquisition=0.0, expense_maintenance_annual=0.0,
    expense_inflation=0.0,
    ra_confidence=0.75, mortality_cv=0.0,    # RA = 0 으로 단순화
)
result = fcf.measure(mp, asmp)

# 손계산:
# 월 사망률 0.01, in-force trajectory = [1.0, 0.99]
# PV(claims) = 1.0 × 0.01 × 12000 + 0.99 × 0.01 × 12000 = 238.8
# PV(premiums) = 1.0 × 100 + 0.99 × 100 = 199
# BEL = 238.8 - 199 = 39.8
print(f"엔진 BEL: {result.bel[0, 0]:.4f}")
print(f"손계산 BEL: 39.80")
```

두 값이 일치하면 엔진이 사용자의 의도대로 동작하고 있다는 강한 신호.

## 1.7 인접 레시피

이 챕터를 읽고 나서 자연스럽게 갈 다음 자리들:

- **2장 사망 + 단순 진단 일시금** — 사망보험에 진단보험금 (CI 진단 일시금)
  결합. 첫 번째 rider 도입.
- **3장 보험료 납입면제 (waiver)** — `STATE_MODELS["WAIVER"]` 입문.
  active → waiver 상태 추적. 정기보험에 waiver 옵션 결합한 형태.
- **8장 Excel 워크북 (단일 segment)** — 위 예제의 Python 가정을 Excel
  워크북으로 옮기는 방법. 비프로그래머 actuary 가 자기 데이터로 적용
  하는 가장 빠른 경로.

기본 튜토리얼 (`튜토리얼`) 의 5장 (BEL 계산) / 6장 (RA 계산) /
7장 (CSM 계산) 이 본 챕터의 출력값을 도출하는 IFRS 17 의 자세한 수식과
손계산 예제를 다룹니다.

```{admonition} 가정의 정확성과 결과의 의미
:class: warning

이 챕터의 모든 BEL / RA / CSM 숫자는 **사용자가 입력한 가정** (사망률,
할인율, 위험조정 변동계수) 에 100% 의존합니다. precision (계산 정밀도)
이 높다고 accuracy (현실 정확도) 가 자동으로 보장되지 않습니다.

자기 회사의 best estimate 가정 / 경험률 표 / 시나리오를 입력해야
**자기 상품의** BEL 이 됩니다. 본 예제의 가정 값들은 자동차 매뉴얼의
"60 km/h 정속 주행 시" 수준의 illustration 입니다.
```

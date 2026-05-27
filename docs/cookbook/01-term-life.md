# 1장. 정기보험 평가

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **정기보험** 을 fastcashflow 로 평가하는 가장 짧은 코드
- `Assumptions` / `ModelPoints` 가 무엇을 담는지 — Excel 워크북과의 1:1 매핑
- `measure()` 와 `value()` 의 차이, 언제 어느 것을 쓰나
- 결과의 BEL / RA / CSM 값이 의미하는 바와 부호 규약
- 흔한 실수와 그 자리에서 잡는 방법

이 챕터만 봐도 정기보험 평가는 끝까지 갈 수 있도록 만들었습니다.
다른 자료로 점프할 필요 없음.
```

## 1.1 상품 소개 — 정기보험

**정기보험 (term life insurance)** 은 미리 정한 보험기간 동안 피보험자가
사망하면 사망보험금을 지급하는, 가장 단순한 형태의 생명보험입니다.
기간이 지나 만기 도달 시에는 어떤 환급도 없습니다.

본 챕터의 **단순 정기보험** 은 다음 구조를 가정합니다:

- 보험료는 매월 동일액 (level premium) 납입
- 보험금은 일정 금액 (level death benefit) — 감액 없음
- 사망 이외에 만기시 어떤 지급도 없음 (만기환급금 0)
- 보험료 납입기간 = 보험기간 (전기납), 또는 짧게 (단기납)

이 챕터는 **단순 정기보험** 만 다룹니다. 보험료 납입면제 (waiver) /
다종 진단담보 / 만기환급금 결합 같은 변형은 후속 챕터에서 다룹니다.

## 1.2 모델링 매핑 — 상품 → fastcashflow 입력

실무에서 가정과 보유계약은 **Excel 워크북** 으로 관리됩니다.
fastcashflow 의 입력도 그 워크북 구조를 그대로 따릅니다.

| 상품 mechanic | 워크북 위치 | (참고) Python API |
|---|---|---|
| 피보험자 가입연령 | `policies.csv` 의 `issue_age` 열 | `ModelPoints(issue_age=...)` |
| 피보험자 성별 | `policies.csv` 의 `sex` 열 | `ModelPoints(sex=...)` (0 = 남, 1 = 여) |
| 보험기간 (개월) | `policies.csv` 의 `term_months` 열 | `ModelPoints(term_months=...)` |
| 사망보험금 | `coverages.csv` 의 `benefit` 열 | `ModelPoints(benefits={0: ...})` (DEATH 담보가 코드 0 이라 가정) |
| 월 보험료 | `coverages.csv` 의 `premium` 열 | `ModelPoints(level_premium=...)` |
| 사망률 표 | `mortality_tables` 시트 | `Assumptions(mortality_annual=함수)` |
| 해지율 표 | `lapse_tables` 시트 | `Assumptions(lapse_annual=함수)` |
| 할인율 표 | `discount_tables` 시트 | `Assumptions(discount_annual=...)` |
| 신사업비 | `expense_tables` 시트의 `alpha_fixed` row | `Assumptions(expense_items=(ExpenseItem('acquisition', 'alpha_fixed', ...),))` |
| 유지비 | `expense_tables` 시트의 `gamma_fixed` row | `Assumptions(expense_items=(ExpenseItem('maintenance', 'gamma_fixed', ...),))` |
| 위험조정 신뢰수준 | `segments` 시트의 `ra_confidence` | `Assumptions(ra_confidence=...)` |
| 사망률 변동계수 | `segments` 시트의 `mortality_cv` | `Assumptions(mortality_cv=...)` |

**중요한 점**: 실무에서는 거의 모든 입력이 Excel 워크북의 셀입니다.
Python 의 `Assumptions(...)` / `ModelPoints(...)` 를 직접 채우는 것은
워크북 로더 (`fcf.read_assumptions`) 가 내부에서 알아서 해줍니다.
위 표의 우측 컬럼은 **워크북을 통하지 않고 코드로 직접 평가할 때** 의
형태입니다.

## 1.3 최소 작동 예제 — 샘플 워크북으로

fastcashflow 는 **샘플 워크북** 을 함께 배포합니다. 패키지만 설치하면
바로 실행 가능. 자기 데이터를 준비하기 전에 엔진이 어떻게 동작하는지
먼저 확인하는 용도입니다.

샘플 파일을 Excel 에서 직접 열어 보고 싶다면 (.xlsx 한 개 + .csv 두 개):

```python
import fastcashflow as fcf
print(fcf.sample_data_dir())
# /.../site-packages/fastcashflow/sample_data
```

출력된 폴더에 `sample_assumptions.xlsx`, `sample_policies.csv`,
`sample_coverages.csv` 세 파일이 있습니다. Excel 로 열어 시트 구조를
한 번 보면 다음 절의 코드가 무엇을 읽고 있는지 한눈에 들어옵니다.

### 샘플 워크북의 모양

`mortality_tables` 시트 (사망률 표) 의 처음 몇 행:

```
table_id    sex   age   rate
─────────   ───   ───   ────────
MORTALITY_STD    0     30    0.000500    ← 남성 30세
MORTALITY_STD    0     31    0.000550
MORTALITY_STD    0     32    0.000605
...
MORTALITY_STD    1     30    0.000400    ← 여성 30세 (별도 행)
...
```

`segments` 시트 (상품 × 채널 별 어떤 표를 쓸지 매핑):

```
product   channel   mortality_table   lapse_table   expense_acq   ...
───────   ───────   ───────────────   ───────────   ───────────   ───
defaults  -         MORTALITY_STD          -             -             ...
TERM_LIFE_A    GA        -                 LAPSE_GA      150,000       ...
TERM_LIFE_A    FC        -                 LAPSE_FC       80,000       ...
```

`defaults` 행은 모든 segment 의 공통값. 개별 segment 행은 빈 셀이면
defaults 를 상속, 채워진 셀은 그 segment 만 override.
즉 `TERM_LIFE_A / GA` 와 `TERM_LIFE_A / FC` 는 **같은 사망률 표** 를 쓰지만
**해지율 / 신사업비** 가 다른 두 채널 segment.

```{admonition} 향후 실제 워크시트 스크린샷으로 교체 예정
:class: note

위 두 블록은 ASCII 로 워크북 모양을 흉내낸 것입니다. 후속 챕터에서
실제 워크시트 스크린샷으로 자세히 설명할 예정입니다.
```

```{admonition} 객체 안에 뭐가 들었는지 보고 싶다면
:class: tip

`fcf.describe_assumptions(basis)` 한 줄이면 위 워크북이 메모리에 어떤
트리 구조로 들어왔는지 (segment 키, 위험률 callable, 경제 / 비용,
RA 파라미터, coverages, state_model) 한 번에 출력합니다.
단일 segment 만 보려면 `describe_assumptions(asmp)`.
```

### 코드 — 샘플로 즉시 평가

다음 코드는 그대로 복사해서 실행:

```python
import fastcashflow as fcf

# 샘플 워크북 로드 (패키지 내장)
basis = fcf.load_sample_assumptions()    # {(product, channel): Assumptions}
mp = fcf.load_sample_model_points()      # ModelPoints, 보유계약 8건

# 평가할 segment 선택 (한 상품에 두 채널)
asmp = basis[("TERM_LIFE_A", "GA")]           # 또는 ("TERM_LIFE_A", "FC")

# 측정 -- 두 가지 방법
detail = fcf.measure(mp, asmp)
fast = fcf.value(mp, asmp)

# 결과 출력 (포트폴리오 합계, 시점 0 = 가입 시점)
print("<Detail>")
print(f"BEL : {detail.bel[:, 0].sum():>15,.0f}")     # 최선추정부채
print(f"RA  : {detail.ra[:, 0].sum():>15,.0f}")      # 위험조정
print(f"CSM : {detail.csm[:, 0].sum():>15,.0f}")     # 보험계약마진
print(f"Loss: {detail.loss_component.sum():>15,.0f}")  # 손실요소
print()
print("<Fast path>")
print(f"BEL : {fast.bel.sum():>15,.0f}")
print(f"RA  : {fast.ra.sum():>15,.0f}")
print(f"CSM : {fast.csm.sum():>15,.0f}")
print(f"Loss: {fast.loss_component.sum():>15,.0f}")
```

실행하면 (샘플 데이터 그대로 사용 시):

```
<Detail>
BEL :      34,150,887
RA  :         850,840
CSM :               0
Loss:      35,001,727

<Fast path>
BEL :      34,150,887
RA  :         850,840
CSM :               0
Loss:      35,001,727
```

코드 한 번 돌리면 포트폴리오 8건의 BEL / RA / CSM 합계를 얻습니다.
샘플 portfolio 는 사업비가 보험료 유입을 넘어 **손실** 이 나는 묶음 —
CSM = 0 이고 손실분이 즉시 `loss_component` 로 인식됩니다 (자세한
해석은 다음 절).

### 자기 워크북으로 바꾸기

자사 워크북을 만들었다면 (포맷은 [`assumptions-format`](../assumptions-format)
또는 챕터 2 참조), `load_sample_*` 두 줄을 다음으로 교체:

```python
basis = fcf.read_assumptions("path/to/your_assumptions.xlsx")
asmp  = basis[("your_product", "your_channel")]              # 평가할 segment 선택
mp    = fcf.read_model_points("path/to/your_policies.xlsx",
                              asmp,                          # 특약 코드 해석용
                              coverages="path/to/your_coverages.xlsx")
```

`asmp` 를 `read_model_points` 에 넘기는 이유는 policies / coverages 파일의
**특약 코드** 를 어셈션의 coverage master (workbook 의 `coverages` 시트) 와
매칭하기 위해서입니다. 사망 단독 wide-form (특약 컬럼 없음) 이면 두 번째
인자는 생략 가능하지만, 일반적인 한국 상품 (사망 + 다종 특약) 은 항상
필요합니다.

### 면책 / 감액 컬럼 (optional)

long-form `coverages` 프레임은 담보별 보장 룰을 세 개의 optional
컬럼으로 받습니다. 두 룰은 **모두 가입 시점 (t=0) 에서 시작**하므로
별도의 `*_start` 컬럼은 없습니다.

| 컬럼 | 단위 | 적용 구간 |
|---|---|---|
| `waiting` | 정수 (개월) | `[0, waiting)` 동안 미지급 (면책기간 — 암 특약 90 일, CI 1 년 등) |
| `reduction_end` | 정수 (개월) | `[0, reduction_end)` 동안 부분 지급 (감액기간) |
| `reduction_factor` | 실수 (0..1) | 감액기간 중 지급 비율 (보통 0.5) |

따라서 `t < waiting` → 0%, `t < reduction_end` → `reduction_factor`%,
그 이후 → 100%. 면책과 감액을 함께 두면 `waiting <= reduction_end` 가
일반적입니다 (예: `waiting=3`, `reduction_end=24`, `reduction_factor=0.5`
→ 첫 3 개월 면책, 4~24 개월 50%, 25 개월부터 100%).

`reduction_factor` 만 있고 `reduction_end` 가 없으면 reader 가 거부합니다
(factor 가 영영 발동하지 않으므로). 자세한 동작은 챕터 11 의 검증 절을
참조하세요.

## 1.4 결과 해석 — 숫자가 무엇을 말하는가

### 부호 규약 (sign convention)

fastcashflow 는 **부채 관점에서 유출을 양수, 유입을 음수로**
부호화하는 규약 (outflow-positive) 을 씁니다:

- **유입** (보험사가 받는 돈, 보험료) — 부채를 **감소** 시킴
- **유출** (보험사가 내는 돈, 사망보험금 + 사업비) — 부채를 **증가** 시킴
- 따라서 `BEL = PV(claims) + PV(expenses) - PV(premiums)`

### BEL = +34,150,887 의 의미

BEL 이 **양수** 라는 것은 "예상 미래 사망보험금 + 사업비 유출" 이
"예상 미래 보험료 유입" 보다 크다는 뜻 — 즉 포트폴리오 전체가
보험사 입장에서 **손실이 나는 묶음** (onerous). 가입 시점에 손실분이
즉시 인식되어 `loss_component = 35,001,727` 으로 잡힙니다 (IFRS 17 Sec. 47).

샘플은 의도적으로 사업비가 크게 설정되어 onerous 가 되도록 조정되어
있습니다 — `mortality_cv` 를 낮추거나 `expense_items` 의 `value` 를
낮춰 실험하면 같은 portfolio 가 이익 묶음 (BEL < 0, CSM > 0) 으로
바뀝니다.

### RA = 850,840 — 위험조정

RA (Risk Adjustment = 위험조정) 는 미래 사망률 / 비용 / 해지의
**불확실성** 에 대해 보험사가 받는 보상. 75% 신뢰수준이면
"BEL + RA" 가 75% 백분위 부채 추정치에 해당.

```{admonition} 확인 포인트
:class: note

RA 가 작으면 BEL 의 불확실성이 작다는 뜻. segments 시트의
`mortality_cv` (변동계수) 를 0.10 → 0.50 으로 바꾸면 RA 가 5배 커집니다.
한 번 실험해보세요.
```

### CSM = 0, loss_component = 35,001,727 — 보험계약마진과 손실요소

CSM (Contractual Service Margin = 보험계약마진) 은 IFRS 17 의 핵심
개념. 계약 가입 시점에 "이익이 날 거다" 라고 인식한 부분을 **미래에
걸쳐 분산해서** 손익으로 전환하기 위한 buffer.

- 가입 시점 (t=0): `CSM_0 = max(0, -FCF)` where `FCF = BEL + RA`
- 매 기간 이자 부리 + 보장단위 비례 상각
- 이익이 나는 계약 (FCF < 0) → CSM > 0, 손실요소 = 0
- 손실이 나는 계약 (FCF > 0) → CSM = 0, 손실요소 = FCF

위 예제는 onerous portfolio 이므로 `CSM = 0`, `loss_component = 35,001,727`
(= BEL + RA = 34,150,887 + 850,840).

### measure() 와 value() 의 차이

```{list-table}
:header-rows: 1
:widths: 12 44 44

* -
  - `measure()`
  - `value()`
* - 출력
  - 시간 trajectory 전체 — BEL/RA/CSM/Loss 의 매월 값 + 현금흐름 6갈래
  - 시점 0 의 BEL/RA/CSM/Loss 만
* - 용도
  - 상세 검증 / 변동분석 / 시각화 / 보고용
  - 대량 portfolio 평가, 민감도, 100만+ 계약
* - 메모리
  - 100만 MP x 120개월 ~ 9GB
  - 100만 MP ~ 32MB
* - 속도 (100만 MP)
  - 수 초
  - 80-300 ms
```

**규칙**: 시간 trajectory 가 필요하면 (검증 / 변동분석 / 시각화 / 보고)
`measure()`, 시점 0 의 결과 4개만 필요하면 (대량 portfolio 평가, 민감도)
`value()`. 두 결과는 시점 0 에서 **수치적으로 동일** (parity test 가 자동
검증).

## 1.5 자주 쓰는 변형

### 채널만 바꾸기 — 같은 상품, 다른 channel

같은 상품 (TERM_LIFE_A) 의 GA / FC 두 채널은 해지율과 신사업비가 다릅니다.
segment 키만 바꿔서 비교:

```python
mp = fcf.load_sample_model_points()
basis = fcf.load_sample_assumptions()

for key in basis:
    val = fcf.value(mp, basis[key])
    bel = val.bel.sum()
    ra  = val.ra.sum()
    csm = val.csm.sum()
    print(f"{key}: BEL={bel:>14,.0f}  RA={ra:>9,.0f}  CSM={csm:>14,.0f}")
```

출력:

```
('TERM_LIFE_A', 'FC'): BEL=    21,672,438  RA=1,870,302  CSM=     1,488,802
('TERM_LIFE_A', 'GA'): BEL=    34,150,887  RA=  850,840  CSM=             0
('HEALTH_A', 'FC'): BEL=    21,892,167  RA=1,870,302  CSM=     1,408,900
('HEALTH_A', 'GA'): BEL=    34,480,481  RA=  850,840  CSM=             0
('HEALTH_A', 'TM'): BEL=    32,942,377  RA=  850,840  CSM=             0
('WHOLE_LIFE_A', 'FC'): BEL=    22,990,813  RA=1,870,302  CSM=     1,052,598
('WHOLE_LIFE_A', 'GA'): BEL=    36,348,179  RA=  850,840  CSM=             0
```

같은 보유계약, 같은 사망률·할인율이지만 채널의 해지율·신사업비 차이가
BEL 과 CSM 에 그대로 반영됩니다. 한국 상품 구조에서 **(상품, 채널)** 이
실질적인 가정 단위인 이유.

### 보유계약을 직접 만들기 — 빠른 실험용

샘플 워크북 대신 코드로 가상의 portfolio 를 만들어 빠르게 실험:

```python
import numpy as np

n_contracts = 1000               # 보유계약 1,000건
rng = np.random.default_rng(42)  # 난수 생성기 (시드 42 - 매번 같은 값 재현용)

portfolio = fcf.ModelPoints(
    issue_age=rng.integers(25, 60, n_contracts),                   # 25-60세 랜덤
    sex=rng.integers(0, 2, n_contracts),                           # 0 또는 1
    benefits={0: rng.integers(10, 100, n_contracts) * 1_000_000},  # 1억-10억 (코드 0 = DEATH)
    level_premium=rng.integers(3, 15, n_contracts) * 10_000,       # 3만-15만
    term_months=np.full(n_contracts, 120),                         # 모두 10년
    benefit_patterns=fcf.load_sample_benefit_patterns(),
)

asmp = fcf.load_sample_assumptions()[("TERM_LIFE_A", "GA")]
result = fcf.value(portfolio, asmp)

print(f"Total  : {result.bel.sum():>15,.0f}")                 # 포트폴리오 BEL 합계
print(f"Mean   : {result.bel.mean():>15,.0f}")                # 평균 계약 BEL
print(f"Onerous: {(result.loss_component > 0).sum():>15,d}")  # 손실 계약 개수
```

### 보험료 납입기간 단기납 (보장기간 != 납입기간)

10년 만기, 5년만 보험료 납입:

```python
mp = fcf.ModelPoints.single(
    issue_age=40,                     # 가입연령 40세
    benefits={0: 100_000_000},        # 사망보험금 1억 (코드 0 = DEATH)
    level_premium=140_000,            # 5년만 내므로 더 큰 금액
    term_months=120,                  # 보장 10년
    premium_term_months=60,           # 납입 5년
    benefit_patterns=fcf.load_sample_benefit_patterns(),
)
```

### 보험료 frequency — 분기납 / 반기납 / 연납

월납 외에:

```python
mp = fcf.ModelPoints.single(
    issue_age=40,                     # 가입연령 40세
    benefits={0: 100_000_000},        # 사망보험금 1억 (코드 0 = DEATH)
    level_premium=70_000,             # 매 분기 7만원
    term_months=120,                  # 보장 10년
    premium_frequency_months=3,       # 분기납 (3개월에 한 번)
    benefit_patterns=fcf.load_sample_benefit_patterns(),
)
```

`premium_frequency_months=12` 이면 연납, `=6` 이면 반기납.

```{admonition} 사망률 / 해지율 표를 바꾸려면
:class: note

자사 경험률표로 평가하려면 워크북의 `mortality_tables` /
`lapse_tables` 시트에 행을 추가하고 `segments` 시트의 `mortality_table`
/ `lapse_table` 컬럼에서 그 `table_id` 를 가리키면 됩니다. 자세한
워크북 편집 가이드는 챕터 2 (Excel 워크북 — 한 segment).
```

## 1.6 함정 — 흔한 실수와 잡는 방법

### 함정 1 — 존재하지 않는 segment 키

```python
basis = fcf.load_sample_assumptions()
asmp = basis[("TERM_LIFE_A", "TM")]   # KeyError: 샘플의 TERM_LIFE_A 는 GA / FC 만
```

`basis.keys()` 로 어떤 segment 가 있는지 먼저 확인:

```python
print(sorted(basis.keys()))
# [('HEALTH_A', 'FC'), ('HEALTH_A', 'GA'), ('HEALTH_A', 'TM'),
#  ('TERM_LIFE_A', 'FC'), ('TERM_LIFE_A', 'GA'),
#  ('WHOLE_LIFE_A', 'FC'), ('WHOLE_LIFE_A', 'GA')]
```

자기 워크북에서는 `segments` 시트의 `(product_code, channel_code)` 조합이
그대로 키가 됩니다 (`defaults` 행은 제외).

### 함정 2 — sex 코딩 (0/1)

`fastcashflow` 의 성별 인코딩은 **0 = 남, 1 = 여**. 워크북의 `policies`
시트, `mortality_tables` 시트 모두 동일한 규약을 따라야 합니다. 일부
사내 표준 (예: M/F, 1/2) 과 다르므로 로드 전에 변환 필요.

### 함정 3 — 음수 BEL 을 보고 놀람

음수 BEL 은 **이익 계약** 이라는 의미. 정상입니다. 손실 계약은 BEL 양수
+ CSM 0 + loss_component 양수의 조합. 신호 패턴:

| BEL | CSM | loss_component | 의미 |
|---|---|---|---|
| 음수 | 양수 | 0 | 이익 계약 — 정상 |
| 0 | 0 | 0 | 손익분기 계약 |
| 양수 | 0 | 양수 | 손실 계약 (onerous) — 즉시 손실 인식 |

### 함정 4 — 자기 워크북의 `table_id` 매칭 누락

`segments` 시트의 `mortality_table` 컬럼에 `MORTALITY_STD` 라고 적었는데
`mortality_tables` 시트엔 그런 `table_id` 가 없으면 로드 시 명확한
에러로 알려줍니다. 새 segment 를 추가할 때 자주 발생.

### 함정 5 — Assumptions 를 코드로 직접 채울 때 함수 형식

위 1.3 / 1.5 의 예제는 모두 워크북 로더가 사망률 / 해지율을 함수로
변환해줍니다. 만약 **워크북을 거치지 않고 직접 `Assumptions(...)`** 을
호출한다면 (보통 검증 / 단위테스트 용도), rate 인자는 **숫자가 아닌
함수** 여야 합니다.

```python
# 직접 작성하는 드문 경우 — 검증 / 단위테스트
assumptions = fcf.Assumptions(
    mortality_annual=lambda sex, age, dur: np.full(dur.shape, 0.001),
    lapse_annual   =lambda sex, age, dur: np.full(dur.shape, 0.01),
    discount_annual=0.03,
    # ...
)
```

3 인자 `(sex, issue_age, duration)`, 출력은 입력과 같은 shape 의 배열.
이 형태는 다음 검증 절에서 다시 사용합니다.

### 검증 — 손계산 한 번

신뢰성 빌드업의 가장 쉬운 방법은 **2개월 계약 손계산**.

설정:

- **2개월 계약**, 가입 후 2 시점 (t=0, t=1) 만 평가
- **월 사망률 1%**, 사망보험금 12,000, 월 보험료 100, 할인 0%

엔진은 **연 사망률** 을 받아 내부에서 월로 환산합니다. 손계산 (월 1%)
과 일치시키려면 `1 - (1-0.01)^12` (월 1% 의 연 환산값) 을 넣습니다 —
엔진이 다시 월로 내리면 정확히 0.01 이 됩니다.

```python
import numpy as np
import fastcashflow as fcf

# 가입 후 2개월, 월 사망률 1%, 사망보험금 12,000, 보험료 100, 할인 0%
mort = lambda s, a, d: np.full(a.shape, 1 - (1-0.01)**12)
mp = fcf.ModelPoints.single(
    issue_age=40,                          # 가입연령
    benefits={0: 12_000},                  # 사망보험금 (코드 0 = DEATH)
    level_premium=100,                     # 월 보험료
    term_months=2,                         # 보장 2개월
    benefit_patterns={"DEATH": fcf.BenefitPattern.DEATH},
)
asmp = fcf.Assumptions(
    # 사망률: 연 환산값. 엔진이 월 단위로 내리면 정확히 1% 가 됨
    mortality_annual=mort,

    # 해지율: 해지 없음
    lapse_annual=lambda s, a, d: np.full(d.shape, 0.0),

    # 할인율: 0% (검증 단순화)
    discount_annual=0.0,

    # 사업비: 전부 0 (검증 단순화)
    expense_inflation=0.0,           # 비용 인플레이션

    # 위험조정 (RA = 0 으로 단순화)
    ra_confidence=0.75,              # 신뢰수준 (cv=0 이라 사용 안 됨)
    mortality_cv=0.0,                # 변동계수 0 -> RA = 0

    # 사망 보장 — 사망 사건의 지급 rate (decrement 의 mortality_annual 과
    # 같은 값을 쓰면 손계산 PV(claims) 와 일치)
    coverages=(fcf.CoverageRate("DEATH", mort),),
)
result = fcf.measure(mp, asmp)

# 손계산:
# 월 사망률 0.01, in-force trajectory = [1.0, 0.99]
# PV(claims) = 1.0 x 0.01 x 12000 + 0.99 x 0.01 x 12000 = 238.8
# PV(premiums) = 1.0 x 100 + 0.99 x 100 = 199
# BEL = 238.8 - 199 = 39.8
print(f"Engine   : {result.bel[0, 0]:.2f}")
print(f"Hand-calc: 39.80")
```

출력:

```
Engine   : 39.80
Hand-calc: 39.80
```

두 값이 일치하면 엔진이 사용자의 의도대로 동작하고 있다는 강한 신호.
이 패턴은 1.5 의 함정 5 처럼 **rate 를 직접 함수로 줘서** 의도된 값을
정확히 통제할 수 있는 자리입니다 — 일반 평가에선 워크북 로더가
대신 해주지만, 검증은 직접 작성하는 게 자연스럽습니다.

## 1.7 인접 레시피

이 챕터를 읽고 나서 자연스럽게 갈 다음 자리들:

- **챕터 2 사망 + 단순 진단 일시금** (작성 예정) — 사망보험에 진단보험금
  (CI = Critical Illness = 진단) 일시금 결합. 첫 번째 추가 담보 도입.
- **챕터 3 보험료 납입면제 (waiver)** (작성 예정) — `STATE_MODELS["WAIVER"]`
  입문. active → waiver 상태 추적.
- **챕터 11 검증 도구** — `show_trace` / `show_bel_step` / `show_csm_step`
  으로 본 챕터의 숫자가 어디서 왔는지 풀어 보기.

전체 챕터 라인업은 [쿡북 인덱스](index) 참조.

기본 튜토리얼 (`튜토리얼`) 의 5장 (BEL 계산) / 6장 (RA 계산) /
7장 (CSM 계산) 이 본 챕터의 출력값을 도출하는 IFRS 17 의 자세한 수식과
손계산 예제를 다룹니다.

```{admonition} 가정의 정확성과 결과의 의미
:class: warning

이 챕터의 모든 BEL / RA / CSM 숫자는 **샘플 워크북의 가정** (사망률,
할인율, 위험조정 변동계수) 에 100% 의존합니다. precision (계산 정밀도)
이 높다고 accuracy (현실 정확도) 가 자동으로 보장되지 않습니다.

자사의 best estimate 가정 / 경험률 표 / 시나리오를 입력해야
**자사 상품의** BEL 이 됩니다. 본 예제의 샘플 가정값들은 자동차
매뉴얼의 "60 km/h 정속 주행 시" 수준의 illustration 입니다.
```

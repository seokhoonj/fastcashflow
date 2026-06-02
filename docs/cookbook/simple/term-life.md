# 2.1 정기보험

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **정기보험** 한 계약을 fastcashflow 로 평가하는 가장 짧은 코드 — 20 줄로 끝
- 손계산과 엔진 결과가 정확히 일치하는 것을 확인 (둘 다 `39.11`)
- BEL / RA / CSM 값이 의미하는 바와 부호 규약
- 한 계약 → 보유계약 portfolio (파일) 로 확장
- 흔한 실수와 그 자리에서 잡는 방법

이 챕터만 봐도 정기보험 평가는 끝까지 갈 수 있도록 만들었습니다.
다른 자료로 점프할 필요 없음.
```

## 상품 소개 — 정기보험

**정기보험 (term life insurance)** 은 미리 정한 보험기간 동안 피보험자가
사망하면 사망보험금을 지급하는, 가장 단순한 형태의 생명보험입니다.
기간이 지나 만기 도달 시에는 어떤 환급도 없습니다.

본 챕터의 **단순 정기보험** 은 다음 구조를 가정합니다:

- 보험료는 매월 동일액 (level premium) 납입
- 보험금은 일정 금액 (level death benefit) — 감액 없음
- 사망 이외에 만기 시 어떤 지급도 없음 (만기환급금 0)
- 보험료 납입기간 = 보험기간 (전기납), 또는 짧게 (단기납)

이 챕터는 **단순 정기보험** 만 다룹니다. 보험료 납입면제 (waiver) /
다종 진단담보 / 만기환급금 결합 같은 변형은 후속 챕터에서 다룹니다.

## 한눈에 보기 — 입력 파일과 사용자 함수

평가에 들어가기 전, 어떤 파일을 어떤 함수로 다루는지 전체 그림을 한 번
잡고 갑니다. 본 챕터의 코드는 이 트리를 그대로 따라갑니다. (같은 그림이
기초 part 의 [한눈에 보기](../basics/overview) 에도 정리되어 있습니다.)

```{include} ../_shared/inputs_and_api.md
```

본 챕터는 위 그림 중 **samples.export → read_\* → measure →
print** 정도만 씁니다. 변동분해 / 시각화는 후속 챕터의 자리.

## 한 계약 — 손계산과 엔진 한 번에

가장 단순한 케이스부터 시작합니다. 한 계약, 두 달짜리. 손계산이 그대로
잡히는 작은 예입니다.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 2개월
- 월 사망률 1%, 해지 없음
- 사망보험금 12,000, 월 보험료 100
- 월 할인율 0.5%
- 사업비 0 (할인과 후방재귀에 집중)
```

손계산은 두 시점 (t=0, t=1) 의 cash flow 를 현재가치로 가져와 합칩니다:

| t | 보유계약 | 사망보험금 (월중) | 보험료 (월초) |
|---|---|---|---|
| 0 | 1.0000 | 1.00 × 1% × 12,000 = 120.00 | 100.00 |
| 1 | 0.9900 | 0.99 × 1% × 12,000 = 118.80 | 99.00 |

월 할인율 0.5% 로 할인계수 — 월초 `1 / (1.005)^t`, 월중 `1 / (1.005)^(t+0.5)`:

- PV(보험료) = 100 × 1.000000 + 99 × 0.995025 = **198.51**
- PV(사망보험금) = 120 × 0.997512 + 118.80 × 0.992550 = **237.62**
- **BEL = 237.62 − 198.51 = 39.11**

이 39.11 이 엔진에서도 그대로 나오면 측정이 의도대로 동작하고 있다는
강한 신호입니다. 코드:

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda s, a, d: np.full(d.shape, 0.0)

# 모델 포인트 (계약 하나)
mp = fcf.ModelPoints.single(
    issue_age     = 40,           # 가입연령 40세
    sex           = 0,            # 성별 (0=남, 1=여)
    benefits      = {0: 12_000},  # 0번 보장 (= DEATH) 의 보험금 12,000
    level_premium = 100,          # 월납 보험료 100
    term_months   = 2,            # 보험기간 2개월
)

# 산출기초
basis = fcf.Basis(
    mortality_annual = death_fn,         # 보유계약 감쇠용 사망률 (위 death_fn)
    lapse_annual     = lapse_fn,         # 해지율 (해지 없음)
    discount_annual  = 1.005 ** 12 - 1,  # 연 할인율 (월 0.5% 의 연 환산)
    ra_confidence    = 0.75,             # 위험조정 신뢰수준 75%
    mortality_cv     = 0.10,             # 사망률 변동계수 10%
    coverages        = (
        fcf.CoverageRate("DEATH", death_fn),  # 사망 보장 1종 (청구 rate = death_fn)
    ),
)

m = fcf.gmm.measure(mp, basis)
print(f"BEL  = {m.bel[0]:.2f}")             # 최선추정부채
print(f"RA   = {m.ra[0]:.2f}")              # 위험조정
print(f"CSM  = {m.csm[0]:.2f}")             # 보험계약마진
print(f"Loss = {m.loss_component[0]:.2f}")  # 손실요소
```

출력:

```
BEL  = 39.11
RA   = 16.03
CSM  = 0.00
Loss = 55.14
```

손계산 39.11 과 엔진 39.11 이 정확히 일치합니다. **작은 toy 계약을 손으로
풀어 엔진을 신뢰할 수 있는지 확인하는 것이 cookbook 의 핵심 패턴입니다.**

```{admonition} 두 자리에 같은 사망률 함수를 넘기는 이유
:class: note

`mortality_annual` 은 *보유계약이 사망으로 감쇠* 하는 율, `coverages`
의 `CoverageRate("DEATH", ...)` 는 *그 사망 사건에 사망보험금이 지급*
되는 율 — 엔진 안에서는 별개의 두 양입니다. 손계산은 둘이 같다는
가정에서 답을 도출했으므로, 코드에서도 같은 `death_fn` 을 두 자리에
공유합니다. 한 자리만 바꾸면 두 양이 silent 어긋나 BEL 이 안 맞으니,
한 변수에 lift 후 두 자리에 통과시키는 게 안전한 패턴.
```

```{admonition} gmm.trace 로 한 줄씩 풀어 보기
:class: tip

`fcf.gmm.trace(0, mp, basis)` 한 줄이면 매월 cash flow, BEL 의 후방
재귀, CSM 의 전방 진행이 ASCII 트리로 풀려 나옵니다. 손계산과 엔진이
어긋날 때 어느 단계에서 갈렸는지 한눈에 확인. 자세한 사용은
[검증 패턴](../workflow/validation).
```

## 결과 읽기 — BEL / RA / CSM

### 부호 규약 (sign convention)

fastcashflow 는 **부채 관점에서 유출을 양수, 유입을 음수로**
부호화하는 규약 (outflow-positive) 을 씁니다:

- **유입** (보험사가 받는 돈, 보험료) — 부채를 **감소** 시킴
- **유출** (보험사가 내는 돈, 사망보험금 + 사업비) — 부채를 **증가** 시킴
- 따라서 `BEL = PV(claims) + PV(expenses) - PV(premiums)`

### BEL = +39.11 의 의미

BEL 이 **양수** 라는 것은 "예상 미래 사망보험금 + 사업비 유출" 이
"예상 미래 보험료 유입" 보다 크다는 뜻 — 즉 보험사 입장에서 **손실이
나는 계약** (onerous). 가입 시점에 손실분이 즉시 인식되어
`loss_component` 로 잡힙니다 (IFRS 17 Sec. 47).

위 예제의 39.11 은 작은 toy 숫자지만, 실제 portfolio 의 BEL 도 단위만
크게 같은 의미입니다. 음수 BEL 은 *이익이 예상되는 계약*.

### RA = 16.03 — 위험조정

RA (Risk Adjustment = 위험조정) 는 미래 사망률 / 비용 / 해지의
**불확실성** 에 대해 보험사가 받는 보상. 75% 신뢰수준이면
"BEL + RA" 가 75% 백분위 부채 추정치에 해당.

```{admonition} 확인 포인트
:class: note

RA 가 작으면 BEL 의 불확실성이 작다는 뜻. `mortality_cv` (사망률
변동계수) 를 `0.10 → 0.50` 으로 바꾸면 RA 가 5배 커집니다. 한 번
실험해보세요.
```

### CSM = 0, Loss = 55.14 — 보험계약마진과 손실요소

CSM (Contractual Service Margin = 보험계약마진) 은 IFRS 17 의 핵심
개념. 계약 가입 시점에 "이익이 날 거다" 라고 인식한 부분을 **미래에
걸쳐 분산해서** 손익으로 전환하기 위한 buffer.

- 가입 시점 (t=0): `CSM_0 = max(0, -FCF)` where `FCF = BEL + RA`
- 매 기간 이자 부리 + 보장단위 비례 상각
- 이익이 나는 계약 (FCF < 0) → CSM > 0, 손실요소 = 0
- 손실이 나는 계약 (FCF > 0) → CSM = 0, 손실요소 = FCF

위 예제는 onerous 계약이므로 `CSM = 0`, `Loss = 55.14`
(= BEL + RA = 39.11 + 16.03).

### full=True 와 full=False 의 차이

측정은 함수 하나 — `measure()` — 이고, `full` 인자로 상세도만 고릅니다.
`full=True` (기본) 는 시간 trajectory 전체를, `full=False` 는 시점 0 의
headline 네 숫자만 돌려줍니다.

```{list-table}
:header-rows: 1
:widths: 12 44 44

* -
  - `measure(...)` (`full=True`, 기본)
  - `measure(..., full=False)`
* - 출력
  - 시간 trajectory 전체 — BEL/RA/CSM/Loss 의 매월 값 + 현금흐름 6 갈래
  - 시점 0 의 BEL/RA/CSM/Loss 만
* - 용도
  - 상세 검증 / 변동분해 / 시각화 / 보고용
  - 대량 portfolio 평가, 민감도, 100만+ 계약
* - 메모리
  - 100만 MP × 120개월 ~ 9 GB
  - 100만 MP ~ 32 MB
* - 속도 (100만 MP)
  - 수 초
  - 80–300 ms
```

**규칙**: 시간 trajectory 가 필요하면 (검증 / 변동분해 / 시각화 / 보고)
`full=True` (기본), 시점 0 의 결과 4 개만 필요하면 (대량 portfolio 평가,
민감도) `full=False`. 두 경로는 서로 다른 커널 (`full=True` = rollforward,
`full=False` = fused) 을 쓰지만 시점 0 결과는 **수치적으로 동일** (parity
test 가 자동 검증).

## 포트폴리오 평가 — 파일에서 읽기

위는 한 계약을 코드로 직접 만든 평가였습니다. 실무에서는 보유계약이
수백 ~ 수천만 건이고, 보통 엑셀 / CSV 파일로 들어옵니다. fastcashflow
는 네 갈래의 파일을 입력으로 받습니다:

```{list-table}
:header-rows: 1
:widths: 30 70

* - 파일
  - 내용
* - `calculation_methods.csv`
  - 담보별 산출방식 — 담보 코드 → 산출방식 (DEATH / MORBIDITY / ...)
* - `basis.xlsx`
  - 산출기초 — 사망률 · 해지율 · 할인율 · 사업비 · 위험조정
* - `policies.csv`
  - 보유 계약 — 한 줄 = 한 계약 (가입연령 / 성별 / 보험기간 / 계약수)
* - `coverages.csv`
  - 담보 가입금액 — 한 줄 = 한 (계약, 담보) (long-form)
```

각 파일의 자세한 구조는 [튜토리얼 11장](../../tutorial/11-in-practice) 에
정리되어 있습니다. 본 챕터에서는 패키지에 동봉된 샘플 파일을 그대로
씁니다 — 그대로 paste 하면 11건의 portfolio 가 평가됩니다:

```python
import fastcashflow as fcf

# (1) 샘플 파일을 현재 폴더에 생성 (한 번만 -- 이미 자기 파일이 있으면 생략)
fcf.samples.export(".", template="gmm")   # basis.xlsx + policies / coverages / calculation_methods (+ inforce)

# (2) 읽어서 평가
basis = fcf.read_basis("basis.xlsx")    # {(product_code, channel_code): Basis}
mp    = fcf.read_model_points(
    "policies.csv",                                 # 계약 spec 파일
    coverages="coverages.csv",                      # 담보 가입금액 파일
    calculation_methods="calculation_methods.csv",  # 담보별 산출방식 파일
)

# 한 segment 의 가정을 전체 portfolio 에 적용 — 상세 trajectory
detail = fcf.gmm.measure(mp, basis[("TERM_LIFE_A", "GA")])

# 같은 평가의 빠른 경로 — 시점 0 의 네 숫자만
fast = fcf.gmm.measure(mp, basis[("TERM_LIFE_A", "GA")], full=False)

print(f"<full=True — 시점 0 합계>")
print(f"BEL : {detail.bel.sum():>15,.0f}")
print(f"RA  : {detail.ra.sum():>15,.0f}")
print(f"CSM : {detail.csm.sum():>15,.0f}")
print(f"Loss: {detail.loss_component.sum():>15,.0f}")
print()
print(f"<full=False — 시점 0>")
print(f"BEL : {fast.bel.sum():>15,.0f}")
print(f"RA  : {fast.ra.sum():>15,.0f}")
print(f"CSM : {fast.csm.sum():>15,.0f}")
print(f"Loss: {fast.loss_component.sum():>15,.0f}")
```

출력 (샘플 그대로):

```
<full=True — 시점 0 합계>
BEL :      33,470,642
RA  :         847,965
CSM :               0
Loss:      34,318,606

<full=False — 시점 0>
BEL :      33,470,642
RA  :         847,965
CSM :               0
Loss:      34,318,606
```

`full=True` 와 `full=False` 의 시점 0 결과가 정확히 일치 — parity 가 항상
보장됩니다.

```{admonition} 데모용 라우팅의 한계
:class: note

위 코드는 한 segment 의 가정 (`("TERM_LIFE_A", "GA")`) 을 11건 전체에
적용합니다 — portfolio 안에 다른 (상품, 채널) 의 계약이 섞여 있어도
같은 가정을 씁니다. 실무에서는 각 계약을 자기 segment 의 가정에
라우팅하는 `measure(mp, basis)` 를 씁니다 — 자세한 건 11장.
```

자기 데이터로 돌리려면 (1) 단계를 건너뛰고 (2) 단계의 파일명을 자기
파일 경로로 바꾸면 됩니다. wide-form (담보가 컬럼으로 펼쳐진 한
파일) 이면 `coverages=` 와 `calculation_methods=` 인자가 필요 없습니다.

### 면책 / 감액 컬럼 (optional)

long-form `coverages` 프레임은 담보별 보장 룰을 세 개의 optional
컬럼으로 받습니다. 두 룰은 **모두 가입 시점 (t=0) 에서 시작** 하므로
별도의 `*_start` 컬럼은 없습니다.

| 컬럼 | 단위 | 적용 구간 |
|---|---|---|
| `waiting` | 정수 (개월) | `[0, waiting)` 동안 미지급 (면책기간 — 암 특약 90일, CI 1년 등) |
| `reduction_end` | 정수 (개월) | `[0, reduction_end)` 동안 부분 지급 (감액기간) |
| `reduction_factor` | 실수 (0..1) | 감액기간 중 지급 비율 (보통 0.5) |

따라서 `t < waiting` → 0%, `t < reduction_end` → `reduction_factor`%,
그 이후 → 100%. 면책과 감액을 함께 두면 `waiting <= reduction_end` 가
일반적입니다 (예: `waiting=3`, `reduction_end=24`, `reduction_factor=0.5`
→ 첫 3개월 면책, 4 ~ 24개월 50%, 25개월부터 100%).

`reduction_factor` 만 있고 `reduction_end` 가 없으면 reader 가 거부합니다
(factor 가 영영 발동하지 않으므로). 자세한 동작은
[검증 패턴](../workflow/validation) 절을 참조하세요.

## 자주 쓰는 변형

### 채널만 바꾸기 — 같은 상품, 다른 channel

같은 상품 (`TERM_LIFE_A`) 의 GA / FC 두 채널은 해지율과 신사업비가
다릅니다. segment 키만 바꿔서 비교:

```python
mp    = fcf.samples.model_points()
basis = fcf.samples.basis()

for key in basis:
    val = fcf.gmm.measure(mp, basis[key], full=False)
    print(f"{str(key):22}: BEL={val.bel.sum():>14,.0f}  RA={val.ra.sum():>9,.0f}  CSM={val.csm.sum():>14,.0f}")
```

출력:

```
('TERM_LIFE_A', 'FC') : BEL=    20,955,426  RA=1,854,622  CSM=     1,488,802
('TERM_LIFE_A', 'GA') : BEL=    33,470,642  RA=  847,965  CSM=             0
('HEALTH_A', 'FC')    : BEL=    21,175,155  RA=1,854,622  CSM=     1,408,900
('HEALTH_A', 'GA')    : BEL=    33,800,235  RA=  847,965  CSM=             0
('HEALTH_A', 'TM')    : BEL=    32,262,131  RA=  847,965  CSM=             0
('WHOLE_LIFE_A', 'FC'): BEL=    22,273,802  RA=1,854,622  CSM=     1,052,598
('WHOLE_LIFE_A', 'GA'): BEL=    35,667,934  RA=  847,965  CSM=             0
```

같은 보유계약, 같은 사망률 · 할인율이지만 채널의 해지율 · 신사업비
차이가 BEL 과 CSM 에 그대로 반영됩니다. 한국 상품 구조에서
**(상품, 채널)** 이 실질적인 가정 단위인 이유.

### 보유계약을 직접 만들기 — 빠른 실험용

샘플 워크북 대신 코드로 가상의 portfolio 를 만들어 빠르게 실험:

```python
import numpy as np

n_contracts = 1000               # 보유계약 1,000건
rng = np.random.default_rng(42)  # 난수 생성기 (시드 42 — 매번 같은 값 재현)

portfolio = fcf.ModelPoints(
    issue_age        = rng.integers(25, 60, n_contracts),                   # 25 ~ 60세
    sex              = rng.integers(0, 2, n_contracts),                     # 0 또는 1
    benefits         = {0: rng.integers(10, 100, n_contracts) * 1_000_000}, # 1 ~ 10억
    level_premium    = rng.integers(3, 15, n_contracts) * 10_000,           # 3 ~ 15만원
    term_months      = np.full(n_contracts, 120),                           # 모두 10년
    calculation_methods = fcf.samples.calculation_methods(),
)

basis   = fcf.samples.basis()[("TERM_LIFE_A", "GA")]
result = fcf.gmm.measure(portfolio, basis, full=False)

print(f"Total  : {result.bel.sum():>15,.0f}")                  # 합계
print(f"Mean   : {result.bel.mean():>15,.0f}")                 # 평균
print(f"Onerous: {(result.loss_component > 0).sum():>15,d}")   # 손실 계약 수
```

### 보험료 납입기간 단기납 (보장기간 ≠ 납입기간)

10년 만기, 5년만 보험료 납입:

```python
mp = fcf.ModelPoints.single(
    issue_age           = 40,                  # 가입연령
    benefits            = {0: 100_000_000},    # 사망보험금 1억
    level_premium       = 140_000,             # 5년만 내므로 더 큰 금액
    term_months         = 120,                 # 보장 10년
    premium_term_months = 60,                  # 납입 5년
    calculation_methods    = fcf.samples.calculation_methods(),
)
```

### 보험료 frequency — 분기납 / 반기납 / 연납

월납 외에:

```python
mp = fcf.ModelPoints.single(
    issue_age                = 40,                    # 가입연령
    benefits                 = {0: 100_000_000},      # 사망보험금 1억
    level_premium            = 70_000,                # 매 분기 7만원
    term_months              = 120,                   # 보장 10년
    premium_frequency_months = 3,                     # 분기납
    calculation_methods         = fcf.samples.calculation_methods(),
)
```

`premium_frequency_months=12` 이면 연납, `=6` 이면 반기납.

```{admonition} 사망률 / 해지율 표를 바꾸려면
:class: note

회사 경험률표로 평가하려면 워크북의 `mortality_tables` / `lapse_tables`
시트에 행을 추가하고 `segments` 시트의 `mortality_table` /
`lapse_table` 컬럼에서 그 `table_id` 를 가리키면 됩니다. 자세한 워크북
편집 가이드는 [튜토리얼 11장](../../tutorial/11-in-practice).
```

## 함정 — 흔한 실수와 잡는 방법

### 함정 1 — 존재하지 않는 segment 키

```python
basis = fcf.samples.basis()

# ✗ 없는 segment 키는 KeyError 입니다 (실행하지 마세요):
#     basis[("TERM_LIFE_A", "TM")]   # 샘플의 TERM_LIFE_A 는 GA / FC 만
```

`basis.keys()` 로 어떤 segment 가 있는지 먼저 확인:

```python
print(sorted(basis.keys()))
# [('HEALTH_A', 'FC'), ('HEALTH_A', 'GA'), ('HEALTH_A', 'TM'),
#  ('TERM_LIFE_A', 'FC'), ('TERM_LIFE_A', 'GA'),
#  ('WHOLE_LIFE_A', 'FC'), ('WHOLE_LIFE_A', 'GA')]
```

자기 워크북에서는 `segments` 시트의 `(product_code, channel_code)`
조합이 그대로 키가 됩니다 (`defaults` 행은 제외).

### 함정 2 — sex 코딩 (0 / 1)

fastcashflow 의 성별 인코딩은 **0 = 남, 1 = 여**. 워크북의 `policies`
시트, `mortality_tables` 시트 모두 같은 규약. 일부 사내 표준 (M/F, 1/2)
과 다르므로 로드 전에 변환 필요.

### 함정 3 — 음수 BEL 을 보고 놀람

음수 BEL 은 **이익 계약** 이라는 의미. 정상입니다. 손실 계약은 BEL
양수 + CSM 0 + Loss 양수의 조합. 신호 패턴:

| BEL | CSM | loss_component | 의미 |
|---|---|---|---|
| 음수 | 양수 | 0 | 이익 계약 — 정상 |
| 0 | 0 | 0 | 손익분기 계약 |
| 양수 | 0 | 양수 | 손실 계약 (onerous) — 즉시 손실 인식 |

### 함정 4 — 자기 워크북의 `table_id` 매칭 누락

`segments` 시트의 `mortality_table` 컬럼에 `MORTALITY_STD` 라고 적었는데
`mortality_tables` 시트엔 그런 `table_id` 가 없으면 로드 시 명확한
에러로 알려줍니다. 새 segment 를 추가할 때 자주 발생.

### 함정 5 — `mortality_annual` 과 DEATH coverage rate 의 미스매치

위 "한 계약 평가" 예제에서 `death_fn` 을 두 자리 (`mortality_annual` 과
`coverages` 의 DEATH 행) 에 똑같이 넘긴 이유 — 한 자리만 override 하면
보유계약 감쇠와 사망보험금 청구가 silent 어긋나 손계산과 안 맞습니다.

워크북 로더는 이걸 자동으로 처리해주지만, **직접 `Basis(...)` 를
호출할 때는 항상 같은 callable 을 두 자리에 공유**하는 게 안전한 패턴.

## 인접 레시피

이 챕터를 읽고 나서 자연스럽게 갈 다음 자리들:

- [보장 청구 메커니즘](../basics/coverage-mechanics) — DEATH 외에 다른
  보장 (DIAGNOSIS / MORBIDITY) 이 엔진 안에서 어떻게 다른 알고리즘으로
  처리되는지. 본 챕터가 한 가지 산출방식만 다루는 이유.
- 사망 + 단순 진단 일시금 (작성 예정) — 사망보험에 진단보험금
  (CI = Critical Illness = 진단) 일시금 결합. 첫 번째 추가 담보 도입.
- 보험료 납입면제 (waiver) (작성 예정) — `STATE_MODELS["WAIVER"]`
  입문. active → waiver 상태 추적.
- [검증 패턴](../workflow/validation) — `gmm.trace` / `gmm.trace_bel_step` /
  `gmm.trace_csm_step` 으로 본 챕터의 숫자가 어디서 왔는지 풀어 보기.
- [튜토리얼 11장 — 실무에서의 활용](../../tutorial/11-in-practice) —
  네 갈래 입력 파일의 구조와 결산 워크플로.

전체 챕터 라인업은 [쿡북 인덱스](../index) 참조.

기본 튜토리얼 (`튜토리얼`) 의 5장 (BEL 계산) / 6장 (RA 계산) /
7장 (CSM 계산) 이 본 챕터의 출력값을 도출하는 IFRS 17 의 자세한 수식과
손계산 예제를 다룹니다.

```{admonition} 가정의 정확성과 결과의 의미
:class: warning

본 챕터의 BEL / RA / CSM 숫자는 **샘플 가정 그대로** 의 결과입니다.
실제 회사 평가에서는 mortality_cv / discount_annual / 사업비를 회사
설정에 맞춰야 의미 있는 숫자가 나옵니다. fastcashflow 의 결과는
"입력 가정에 충실한 산출" 이지, 회사 portfolio 의 진실값은 아닙니다.
```

# 7.1 시나리오 / 민감도 분석

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 가정을 **shock** 하는 것 = rate 함수를 교체하는 것. 사망률 +10% 등을 명시
  변수에 lift 한 뒤 배수를 건 새 함수로 바꿔 끼운다
- shock 이 IFRS17 숫자 (BEL / RA / CSM / loss component) 에 미치는 영향을 측정
- **불리한 shock 은 CSM 을 먼저 갉아먹고**, 충분히 크면 계약이 onerous 로
  뒤집혀 **loss component** 가 나타난다 (CSM 은 음수가 못 됨)
- `gmm.trace_diff` 로 한 계약의 shock 전파 (rate -> cash flow -> BEL -> CSM)
  를 ASCII 트리로
- shock 함수는 **3-인자** 로 — 4-인자 default 패턴은 엔진이 잘못 호출 (함정)
```

fastcashflow 의 핵심 가치는 숫자를 한 번 내는 것이 아니라, **가정을 바꾸면
IFRS17 숫자가 어떻게 움직이는지 즉시 보는 것** 입니다. 이 챕터는 그 도구 —
가정을 흔들어 (shock) 보고 그 영향을 읽는 워크플로 — 를 다룹니다.

## shock = rate 함수 교체

가정은 대부분 rate **함수** 입니다 (`mortality_annual`, `lapse_annual`,
담보의 rate). 따라서 "사망률 +10%" 같은 shock 은 그 함수를 *배수를 건 새
함수* 로 바꾸는 것입니다. base rate 를 명시 변수에 lift 해 두면, 배수를 건
함수를 만들어 **같은 자리에 다시 끼웁니다**:

```python
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)  # 사망률 월 1%

def shock_mortality(factor):                       # 사망률에 배수를 건 새 함수
    return lambda s, a, d: np.minimum(death_fn(s, a, d) * factor, 1.0)
```

`death_fn` 은 `mortality_annual` (보유계약 감쇠) *과* DEATH 보장의 청구 rate
**양쪽** 에 들어갑니다. 둘 다 같은 사망률이므로, shock 도 한 번 만들어 두
자리에 함께 먹입니다 — 한쪽만 흔들면 사망 감쇠와 사망보험금이 어긋납니다.

```{admonition} shock 함수는 3-인자로
:class: warning

`shock_mortality` 의 안쪽 lambda 는 `(s, a, d)` **3-인자** 입니다. 무심코
`lambda s, a, d, f=factor: ...` 처럼 default 인자로 배수를 넣으면, 엔진이
rate 를 5-인자 `(sex, age, dur, issue_class, elapsed)` 로 호출하면서 **네
번째 자리 (issue_class) 가 `f` 를 덮어써** shock 이 엉뚱하게 적용됩니다
(issue_class=0 이면 사망률이 0 이 되는 식). 배수는 반드시 **클로저로 캡처**
(위처럼 factory 함수가 `factor` 를 닫아 잡음) 하세요.
```

## 작동 예제 — 사망률 민감도 sweep

흑자 (이익) 계약 하나를 잡고 사망률을 1.0배부터 1.5배까지 흔들어, BEL / RA /
CSM / loss 가 어떻게 움직이는지 봅니다.

```python
import numpy as np
import fastcashflow as fcf

# 계리적 가정 -- 사망률을 명시 변수로 lift
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)  # 사망률 월 1%
lapse_fn = lambda s, a, d: np.full(d.shape, 0.0)                   # 해지 없음

def shock_mortality(factor):                       # 사망률에 배수를 건 새 함수
    return lambda s, a, d: np.minimum(death_fn(s, a, d) * factor, 1.0)

def basis_with_mortality(mort_fn):                 # 한 사망률로 가정 한 벌 조립
    return fcf.Basis(
        mortality_annual = mort_fn,                # 보유계약 감쇠 + 사망보장 공유
        lapse_annual     = lapse_fn,
        discount_annual  = 0.0,                    # 할인 0 (shock 효과에 집중)
        ra_confidence    = 0.75,
        mortality_cv     = 0.10,
        coverages        = (
            fcf.CoverageRate("DEATH", mort_fn),    # 사망 보장에 같은 사망률
        ),
    )

mp = fcf.ModelPoints.single(
    issue_age     = 40,            # 가입연령 40세
    sex           = 0,             # 성별 (0=남, 1=여)
    benefits      = {0: 100_000},  # 사망보험금 100,000
    level_premium = 1_200,         # 월납 보험료 1,200 (claims 보다 커 흑자)
    term_months   = 24,            # 보험기간 2년
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},
)

# 사망률 shock sweep -- decrement 와 claim 양쪽에 같은 배수
print(f"{'factor':>6}  {'BEL':>8}  {'RA':>6}  {'CSM':>7}  {'loss':>7}")
for factor in [1.00, 1.05, 1.10, 1.25, 1.50]:
    v = fcf.gmm.measure(mp, basis_with_mortality(shock_mortality(factor)), full=False)
    print(f"{factor:>6.2f}  {v.bel[0]:>8.0f}  {v.ra[0]:>6.0f}  "
          f"{v.csm[0]:>7.0f}  {v.loss_component[0]:>7.0f}")
```

출력:

```
factor       BEL      RA      CSM     loss
  1.00     -4286    1446     2841        0
  1.05     -3131    1513     1618        0
  1.10     -1983    1581      402        0
  1.25      1422    1780        0     3201
  1.50      6962    2103        0     9065
```

## 결과 읽기 — CSM 이 shock 을 먼저 흡수한다

baseline (factor 1.00) 은 BEL 이 **음수** (-4,286) 입니다 — 보험료가 사망보험금
+ RA 보다 커서 이익이 나는 계약이고, 그 이익이 CSM 2,841 로 잡힙니다
(`CSM_0 = max(0, -(BEL + RA))`, IFRS 17 Sec.38).

사망률을 올리면 사망보험금이 늘어 BEL 이 커지고 (덜 음수가 되고), 그만큼
`FCF = BEL + RA` 가 올라가 **CSM 이 줄어듭니다**:

- **factor 1.10**: CSM 2,841 -> 402. 불리한 가정이 **CSM 을 갉아먹지만** 아직
  이익 범위라 손익(P&L) 에는 안 닿습니다.
- **factor 1.25**: CSM 이 0 으로 **소진** 되고, FCF 가 양수로 넘어가
  **loss component 3,201** 이 나타납니다 — 계약이 onerous (손실부담) 로
  뒤집혀 그 손실이 즉시 인식됩니다 (Sec.47-49). CSM 은 음수가 못 되므로
  (floor 0), 초과분이 loss 로 빠집니다.
- **factor 1.50**: loss 가 9,065 로 더 커집니다.

이것이 민감도 분석의 핵심 그림입니다 — **불리한 변화는 먼저 CSM 이라는
완충재를 소진하고, 그 완충재가 바닥나는 순간부터 손익에 직접 친다.** CSM 이
얼마나 남았는지가 곧 그 계약이 shock 을 얼마나 더 견딜 수 있는지입니다.

## 한 계약 들여다보기 — gmm.trace_diff

sweep 은 *얼마나* 변하는지를 보여주지만, *어디서* 그 변화가 생기는지는
`gmm.trace_diff` 가 두 basis 를 나란히 놓아 보여줍니다 — rate 부터 cash
flow, BEL, CSM 까지 한 줄씩.

```python
base    = basis_with_mortality(death_fn)
shock10 = basis_with_mortality(shock_mortality(1.10))
fcf.gmm.trace_diff(0, mp, base, shock10, label_a="baseline", label_b="mort+10%")
```

출력:

```
diff mp[0]  (-/-, sex=남, issue_age=40, term=24m, premium_term=24m, count=1)
labels: 'baseline'  ->  'mort+10%'
├─ Assumption changes
│   ├─ mortality_annual       : <callable>  ->  <callable>
│   ├─ lapse_annual           : <callable>  ->  <callable>
│   └─ coverage[DEATH].rate   : <callable>  ->  <callable>
├─ Rate deltas (per policy year)
│   ├─ axes: sex=0, issue_age=40, issue_class=0, elapsed_at_issue=0m
│   ├─ year  0
│   │   ├─ mortality(annual)    0.113615  ->    0.124977   ( +0.011362,   +10.00%)
│   │   └─ DEATH(annual)        0.113615  ->    0.124977   ( +0.011362,   +10.00%)
│   └─ year  1
│       ├─ mortality(annual)    0.113615  ->    0.124977   ( +0.011362,   +10.00%)
│       └─ DEATH(annual)        0.113615  ->    0.124977   ( +0.011362,   +10.00%)
├─ Cash flow deltas (annual sum, non-zero rows only)
│   ├─           year          stream   sum(baseline)   sum(mort+10%)               Δ              %Δ
│   ├─              0         premium          13,634          13,555             -79          -0.58%
│   ├─              0           claim          11,362          12,498          +1,136         +10.00%
│   ├─              1         premium          12,085          11,861            -224          -1.85%
│   └─              1           claim          10,071          10,936            +865          +8.59%
├─ Discount factor deltas (key months)
│   ├─ t=   0m: ds  1.000000  ->  1.000000  (+0.000000)
│   ├─ t=  12m: ds  1.000000  ->  1.000000  (+0.000000)
│   └─ t=  24m: ds  1.000000  ->  1.000000  (+0.000000)
├─ BEL deltas (key months)
│   ├─ BEL[   0]        -4,286.44  ->       -1,983.05   (    +2,303.39,   -53.74%)
│   ├─ BEL[  12]        -2,014.13  ->         -925.44   (    +1,088.70,   -54.05%)
│   └─ BEL[  24]             0.00  ->            0.00   (        +0.00,        --)
├─ CSM deltas (key months)
│   ├─ CSM[   0]         2,840.86  ->          402.49   (    -2,438.37,   -85.83%)
│   ├─ CSM[  12]         1,334.88  ->          187.83   (    -1,147.05,   -85.93%)
│   └─ CSM[  24]             0.00  ->           -0.00   (        -0.00,        --)
└─ Final (headline change, per policy)
    ├─ BEL                   -4,286.44  ->       -1,983.05   (    +2,303.39,   -53.74%)
    ├─ RA                     1,445.58  ->        1,580.56   (      +134.98,    +9.34%)
    ├─ FCF = BEL+RA          -2,840.86  ->         -402.49   (    +2,438.37,   -85.83%)
    ├─ CSM = max(0,-FCF)      2,840.86  ->          402.49   (    -2,438.37,   -85.83%)
    └─ loss_component             0.00  ->            0.00   (        +0.00,        --)
```

트리를 위에서 아래로 읽으면 shock 의 **전파 경로** 가 보입니다:

- **Rate deltas** — 사망률이 양쪽 (mortality / DEATH) 에서 +10.00%.
- **Cash flow deltas** — 사망보험금 (claim) 이 +10%, 보험료는 사망 감쇠가
  빨라져 약간 (-0.58%) 줄어듦.
- **BEL deltas** — 청구 증가가 BEL 을 +53.74% 밀어 올림 (덜 음수로).
- **CSM deltas** — 그 BEL 증가가 CSM 을 -85.83% 깎음.
- **Final** — `FCF = BEL + RA`, `CSM = max(0, -FCF)` 의 IFRS17 항등식이
  그대로 드러나, 왜 CSM 이 그만큼 줄었는지를 한 줄로 설명합니다.

`gmm.trace_diff` 는 동일한 항목은 숨기고 *바뀐* 것만 보여주므로, shock 이
어디서 시작해 어디로 흐르는지 추적하는 데 씁니다.

## 변형

### 다른 가정 shock

같은 패턴으로 어느 가정이든 흔듭니다 — `lapse_annual` 배수 (해지 +20%),
`discount_annual` 가감 (금리 -50bp), 사업비 항목 조정. 각각 base 를 명시
변수에 lift 한 뒤 바꿔 끼우고 `measure` (또는 `measure(..., full=False)`) 를 다시 부르면 됩니다.

### 여러 가정 동시 (시나리오)

여러 가정을 한꺼번에 바꾸면 *시나리오* 가 됩니다 (예: 사망률 +10% **와**
해지 -20% 동시). `Basis` 를 그 조합으로 한 벌 만들어 평가하면 됩니다 —
단일 가정 민감도와 달리 cross effect (상호작용) 까지 반영됩니다.

### segment 별 shock

[6.2](../io/workbook-multi) 의 `basis` 사전을 통째로 흔들 수도 있습니다 —
각 segment 의 `Basis` 를 shock 한 새 사전을 만들어 `measure`
에 넘기면 portfolio 전체의 shock 영향이 한 번에 나옵니다.

### 민감도 vs 변동분해

이 챕터는 *가상의* shock 영향 (what-if) 입니다. 분기 사이에 *실제로* 가정 ·
실적이 변해 숫자가 움직인 것을 항목별로 가르는 것은 **변동분해**
(`roll_forward` / `reconcile`) 로, 결산 워크플로의 일부입니다 —
[튜토리얼 11장](../../tutorial/11-in-practice) 참조.

## 함정

### 함정 1 — shock 을 한 자리만 적용

사망률은 `mortality_annual` (감쇠) 과 DEATH 보장 (청구) 두 자리에 들어갑니다.
한쪽만 shock 하면 사망 감쇠와 사망보험금이 어긋나 BEL 이 틀립니다. base 를
명시 변수에 lift 해 **한 shock 을 두 자리에 함께** 먹이세요.

### 함정 2 — shock 함수의 인자 수

위 경고대로, 배수는 클로저로 캡처하고 안쪽 함수는 `(s, a, d)` 3-인자로
두세요. default 인자 (`f=factor`) 로 넣으면 엔진의 5-인자 호출에서 네 번째
자리가 그것을 덮어씁니다.

### 함정 3 — CSM 은 음수가 안 됨

shock 으로 FCF 가 양수가 되면 CSM 은 0 에서 멈추고 초과분이 loss component
로 갑니다. "CSM 이 -3,000 이 됐다" 는 결과는 없습니다 — CSM 0 + loss 3,000
입니다. 민감도 표에서 CSM 과 loss 를 **함께** 봐야 하는 이유입니다.

### 함정 4 — portfolio 합산은 netting 이 아님

계약별 CSM 과 loss 는 상계되지 않습니다 — 한 계약의 CSM 여력이 다른 계약의
loss 를 덮지 못합니다 (각 계약/그룹이 따로 floor). portfolio 민감도는
계약별 CSM 감소와 loss 증가를 **각각** 합산해 봐야 합니다.

## 인접 레시피

- [7.2 검증 패턴](validation) — `gmm.trace` 로 한 계약의 BEL / CSM 계산
  경로를 한 줄씩. `gmm.trace_diff` 의 단일 basis 판.
- [1.3 사망률의 두 가지 역할](../basics/mortality-roles) — 사망률이 감쇠와 청구
  두 자리에 들어가는 구조. shock 을 양쪽에 먹이는 이유.
- [6.2 워크북 — 다 segment](../io/workbook-multi) — `measure` 로
  portfolio 전체 shock.
- [튜토리얼 11장](../../tutorial/11-in-practice) — 실제 분기 변동분해
  (`roll_forward` / `reconcile`).
```

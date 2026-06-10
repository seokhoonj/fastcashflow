# 5.2 변액보험 최저보증의 시간가치 (TVOG)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 보증의 **시간가치 (TVOG = Time Value of Options and Guarantees)** — 결정론
  intrinsic 에는 안 보이는, 변동성이 만드는 보증 비용
- `return_scenarios` 로 펀드 수익률 경로를 넣어 `vfa.measure` 가 시간가치를
  측정하는 법
- 같은 계약의 결정론 intrinsic (작음) 대 시간가치 (큼) 분해 — "단일
  결정론 run 으로 보증을 평가하면 안 되는" 이유
- 시간가치를 GMDB / GMAB 로 분리, 그리고 시간가치의 부호가 보장되지 않는
  까닭
```

[5.1 결정론 측정](gmdb-gmab) 에서 GMDB / GMAB 의 **intrinsic value** — 중앙
시나리오에서 floor 가 무는 비용 — 을 봤습니다. 그 계약의 결정론 보증 비용은
50,594 뿐이었습니다. 하지만 그건 보증 비용의 **일부** 입니다. 펀드 수익률은
중앙값 하나로 굴러가지 않고 변동하며, 그 변동이 만드는 추가 비용이
**시간가치 (TVOG)** 입니다. 이 챕터는 같은 계약에 시나리오를 넣어 시간가치를
드러냅니다.

```{admonition} 이 챕터도 GMDB / GMAB 만 — 적립이율 보증은 끔
:class: note

5.1 과 마찬가지로 `minimum_crediting_rate` (적립이율 보증, 계좌 크레딧
floor) 는 **설정하지 않습니다**. 여기서 측정하는 시간가치는 순수하게 사망 /
만기 floor (GMDB / GMAB) 의 시간가치입니다. 적립이율 보증의 시간가치는 계좌
크레딧을 매월 떠받치는 별개의 메커니즘이라 따로 다룹니다.
```

## 결정론은 시간가치를 못 본다

먼저 5.1 의 계약과 기초를 그대로 다시 세우고, 결정론 측정의 `TVOG = 0` 을
확인합니다.

```python
import numpy as np
import fastcashflow as fcf

# 5.1 과 같은 계약·기초 (펀드 연 3%, 수수료 2.5%, 사망 0.5%, 해지 4%)
basis = fcf.Basis(
    mortality_annual  = 0.005,  # 보유계약 사망률
    lapse_annual      = 0.04,   # 해지율
    discount_annual   = 0.03,   # 연 할인율
    ra_confidence     = 0.95,   # 위험조정 신뢰수준
    mortality_cv      = 0.10,   # 사망률 변동계수
    expense_cv        = 0.10,   # 사업비 변동계수
    investment_return = 0.03,   # 펀드 연 수익률 (중앙값)
    fund_fee          = 0.025,  # 변동수수료
)
mp = fcf.ModelPoints.single(
    issue_age                    = 40,      # 가입연령
    premium                      = 0.0,     # 일시납
    term_months                  = 120,     # 보험기간 10년
    account_value                = 1.0e8,   # 가입 시 계좌가치
    minimum_death_benefit        = 1.02e8,  # GMDB (102%)
    minimum_accumulation_benefit = 1.05e8,  # GMAB (105%)
)

det = fcf.vfa.measure(mp, basis)               # 결정론 (시나리오 없음)
print(f"결정론  CSM  = {det.csm[0]:>14,.0f}")  # intrinsic 흡수 후 마진
print(f"결정론  TVOG = {det.time_value[0]:>14,.0f}")  # 시간가치 (안 보임)
```

출력:

```text
결정론  CSM  =     17,664,772
결정론  TVOG =              0
```

중앙 시나리오에서 계좌는 100,000,000 → 104,933,092 로 평탄하게 굴러갑니다
(5.1 의 궤적). 한 경로만 보면 floor 가 무는 자리가 한정돼 intrinsic 50,594 만
잡히고, 변동성 비용은 잡히지 않습니다.

## 시나리오를 넣으면 시간가치가 드러난다

`return_scenarios` 로 펀드 **월수익률 경로** 를 `(n_scenarios, n_time)` 배열로
건네면, `vfa.measure` 가 각 경로에서 계좌를 굴려 floor 비용을 평가하고 그
기대값에서 intrinsic 을 뺀 **시간가치** 를 돌려줍니다.

```python
# 펀드 월수익률 시나리오 (외부 ESG 산출: 1,000 경로 x 120 개월)
rng  = np.random.default_rng(7)
r_m  = (1 + 0.03) ** (1 / 12) - 1                     # 중앙 월수익률 (연 3%)
vol  = 0.02                                            # 월 변동성 2%
scen = r_m + vol * rng.standard_normal((1000, 120))   # (n_scenarios, n_time)

sto = fcf.vfa.measure(mp, basis, return_scenarios=scen)
print(f"확률론  TVOG = {sto.time_value[0]:>14,.0f}")  # 보증의 시간가치
print(f"확률론  CSM  = {sto.csm[0]:>14,.0f}")         # TVOG 흡수 후 마진
```

출력:

```text
확률론  TVOG =      6,205,470
확률론  CSM  =     11,459,302
```

**보증의 시간가치는 6,205,470 — 결정론 intrinsic 50,594 의 120배가 넘습니다.**
변동성 때문에 일부 경로에서 계좌가 보증액 아래로 더 깊이 떨어지고, 그 풋옵션
비용의 기대값이 시간가치입니다. 이 비용이 CSM 을 17,664,772 → 11,459,302 로
낮춥니다 (계약은 여전히 이익: CSM > 0).

```{admonition} 같은 계약, 두 비용
:class: important

| 측정 | 보증 비용 | CSM |
|---|---:|---:|
| 결정론 (intrinsic) | 50,594 | 17,664,772 |
| 확률론 (결정론 intrinsic + TVOG) | 6,256,064 | 11,459,302 |

결정론만 보면 보증 비용이 5만원 — "거의 공짜" 로 보입니다. 시나리오를
넣어야 6백만의 진짜 비용이 드러납니다. **단일 결정론 run 으로 변액 보증을
평가하면 안 되는** 이유입니다.
```

```{admonition} 시나리오는 외부에서 — fastcashflow 는 엔진
:class: note

`return_scenarios` 는 당신의 ESG (Economic Scenario Generator, 경제 시나리오
생성기) 산출물입니다. fastcashflow 는 시나리오를 생성 하지 않고, 받아 보증
비용을 평가합니다. 한 변수 (여기선 펀드 월수익률) 의 `(n_scenarios, n_time)`
배열로 건네면 됩니다 — 열 수는 투영 개월수와 같아야 합니다. 위 예제의
1,000 경로는 몬테카를로 잡음이 있어 (seed / 경로수에 따라 +/- 5% 흔들림),
실무에서는 더 많은 경로로 수렴시킵니다.
```

## 결과 읽기 — 시간가치를 GMDB / GMAB 로 분리

5.1 의 intrinsic 처럼 시간가치도 보증을 켜고 끄며 분리됩니다:

```python
def tvog_with(gmdb, gmab):
    m = fcf.ModelPoints.single(
        issue_age=40, premium=0.0, term_months=120, account_value=1.0e8,
        minimum_death_benefit=gmdb, minimum_accumulation_benefit=gmab,
    )
    rng2 = np.random.default_rng(7)                       # 같은 시나리오
    s = r_m + vol * rng2.standard_normal((1000, 120))
    return fcf.vfa.measure(m, basis, return_scenarios=s).time_value[0]

print(f"GMDB 시간가치 = {tvog_with(1.02e8, 0.0):>14,.0f}")
print(f"GMAB 시간가치 = {tvog_with(0.0, 1.05e8):>14,.0f}")
print(f"둘 다         = {tvog_with(1.02e8, 1.05e8):>14,.0f}")
```

출력:

```text
GMDB 시간가치 =        236,056
GMAB 시간가치 =      5,969,414
둘 다         =      6,205,470
```

GMDB 236,056 + GMAB 5,969,414 = 6,205,470 — 정확히 합산됩니다.

**시간가치는 GMAB 가 거의 전부 (5,969,414) 입니다.** 만기보증 (GMAB) 은 만기
한 시점에 생존자 **전원** 의 계좌를 105% 와 견주니, 변동성이 그 한 시점의
하방을 크게 키웁니다. 사망보증 (GMDB) 은 매월 소수의 사망자에게만, 그것도
사망률 0.5% 라는 얇은 확률로 무니 시간가치가 훨씬 작습니다 (236,056).

결정론 intrinsic 도 GMAB (31,481) 가 GMDB (19,113) 보다 컸지만, 시간가치에서는
그 격차가 25배로 벌어집니다 — **만기 한 시점에 모인 하방 노출** 이 변동성에
훨씬 민감하다는 신호입니다.

```{admonition} TVOG의 부호는 보장되지 않음
:class: warning

VFA는 위험중립 measure 가 아니라 기초자산 수익률로 할인합니다. 그래서 floor
의 시간가치는 부호가 고정이 아닙니다 — 깊은 in-the-money (보증액이 계좌가치
보다 한참 높은) 보증은 변동성이 오히려 일부 시나리오를 floor 위로 끌어올려
비용을 낮춰, 시간가치가 음수일 수도 있습니다. 위험중립 풋옵션의 "시간가치
>= 0" 직관이 여기선 그대로 통하지 않습니다.
```

## 변형

### onerous 변액 만들어 보기

시나리오 변동성 (`vol`) 을 키우거나 보증액을 올리면 시간가치가 수수료를 넘어
CSM = 0, 손실요소 (loss component) 양수가 됩니다 — onerous 변액계약. 후한 보증
+ 높은 변동성 + 얇은 수수료의 조합이 변액 보증의 손실부담 신호입니다.

```python
hi_vol = r_m + 0.05 * np.random.default_rng(7).standard_normal((1000, 120))
onerous = fcf.vfa.measure(mp, basis, return_scenarios=hi_vol)
print(f"vol 5%  TVOG = {onerous.time_value[0]:>14,.0f}")
print(f"vol 5%  CSM  = {onerous.csm[0]:>14,.0f}")
print(f"vol 5%  loss = {onerous.loss_component[0]:>14,.0f}")
```

출력:

```text
vol 5%  TVOG =     24,986,358
vol 5%  CSM  =              0
vol 5%  loss =      7,321,586
```

월 변동성 5% 에서는 시간가치가 25.0M 으로 수수료 17.7M 을 넘어, CSM 이 0 으로
소진되고 손실요소 7.3M 이 잡힙니다.

### per-MP 보증액 + 시나리오

`return_scenarios` 의 시간가치 패스에서 GMDB / GMAB **보증액 자체는 계약마다
달라도** 됩니다 (`max(AV, 보증)` 의 floor 는 per-MP). 적립이율 보증
(`minimum_crediting_rate`) 에 stochastic 을 적용하는 경우만 v1 에서 portfolio 전체
동일 값을 요구하지만, 이 챕터는 적립이율 보증을 쓰지 않으니 해당되지
않습니다.

## 함정

### 함정 1 — 결정론만 보고 "보증이 싸다" 판단

중앙 성장 가정에선 floor 가 한정된 구간에서만 무려 intrinsic 이 작게
나옵니다. 보증의 진짜 비용 (시간가치) 은 `return_scenarios` 를 넣어야
드러납니다 — 단일 결정론 run 에는 원리상 보이지 않습니다.

### 함정 2 — 시나리오 열 수 != 투영 개월수

`return_scenarios` 의 열 수 (`n_time`) 는 투영 개월수와 같아야 합니다 (여기선
120). 다르면 엔진이 거부합니다. 또한 경로에 빈 배열 / 비유한값 / `<= -1` 의
월수익률이 있으면 거부됩니다 (계좌가 음수가 되는 비현실 경로).

### 함정 3 — 몬테카를로 잡음을 정밀값으로 착각

1,000 경로의 시간가치는 seed / 경로수에 따라 흔들립니다. 보고용 숫자는 충분히
많은 경로로 수렴시키고, 민감도는 같은 시나리오 집합 (같은 seed) 위에서
비교해야 합니다.

## 인접 레시피

- [5.1 변액보험 최저보증 — 결정론 측정](gmdb-gmab) — 같은 계약의 intrinsic
  value 와 GMDB / GMAB 설정 위치, 계좌가치 궤적. 이 챕터의 전제.
- `vfa.trace(mp_index, mp, basis, return_scenarios=scen)` — 이 계약의 계좌가치
  궤적, floor 가 무는 자리, TVOG 까지 트리로 확인 (`return_scenarios` 를 주면
  시간가치 경로도).
- [8.1 시나리오 / 민감도 분석](../workflow/sensitivity) — 가정을 흔들어 CSM
  흡수 / onerous 전환을 보는 워크플로.

# 5.3 적립이율 보증 (크레딧 floor)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- GMDB / GMAB 와는 다른 세 번째 보증 — **최저적립이율 보증**: 계좌가치를 매월
  `max(수익률, 보증이율)` 로 크레딧해 떠받친다
- `minimum_crediting_rate = 0.0` = **원금보존** (마이너스 달에도 계좌가 안 줄어듦) —
  값을 갖는 진짜 보증. 미설정 (`fcf.NO_GUARANTEE_RATE`) 은 "보증 없음"
- 이 보증의 시간가치 (TVOG) 를 `vfa.tvog` 로 따로 보는 법
- **매월 작동하는 floor 는 단일 만기 floor (GMAB) 보다 변동성에 훨씬 비싸다** —
  같은 변동성에서 시간가치가 자릿수로 커지는 이유
```

```{admonition} "floor" 가 뭔가 — 먼저 한 줄로
:class: note

**floor (바닥 / 하한선)** 는 어떤 값이 그 아래로는 못 내려가게 막아 주는 최저선
입니다. 보증은 전부 이 floor 의 한 형태입니다 — 사망보험금이 계좌가치 아래로 못
내려가게 막으면 (GMDB), 만기금이 못 내려가게 막으면 (GMAB), 그리고 이 챕터처럼
**매월 적립이율** 이 못 내려가게 막으면 (크레딧 floor). 보험사는 그 바닥을 메워
주는 대신 비용을 집니다 — 그 비용이 이 챕터의 주제입니다.
```

[5.1](gmdb-gmab) · [5.2](gmdb-gmab-tvog) 의 GMDB / GMAB 는 사망·만기 **한 시점**
에 `max(계좌가치, 보증액)` 을 지급하는 floor 였습니다 — 그 한 시점에 계좌가
보증액 아래면 그 차액을 보험사가 메웁니다. 변액에는 세 번째 보증이 흔히 붙습니다
— **최저적립이율 보증** (크레딧 floor): 한 시점이 아니라 계좌가치 **자체** 를
**매월** 떠받칩니다. 앞 두 챕터는 이 보증을 일부러 꺼 뒀고 (GMDB / GMAB 만), 이
챕터가 그것만 따로 다룹니다.

## 상품 소개 — 매월 떠받치는 적립이율

변액계좌는 보통 `계좌 x (1 + 수익률) x (1 - 수수료)` 로 굴러갑니다. 수익률이
마이너스인 달엔 계좌가 줄어듭니다. **최저적립이율 보증** 은 그 적립률에 바닥을
깔아, 매월 `max(수익률, 보증이율)` 로 크레딧합니다:

- **`0.0` = 0% floor (원금보존)** — 마이너스 달에도 계좌가 줄지 않습니다. 수익이
  난 달은 그대로 받고, 손실 난 달은 0 으로 막아 줍니다.
- **양수 (예: `0.0075`)** — 매월 최소 그 이율만큼은 적립.

GMDB / GMAB 가 사망·만기 한 시점의 floor 라면, 적립이율 보증은 **매월 작동하는**
floor 입니다. 이 "매월" 이 비용을 키웁니다 — 뒤에서 봅니다.

## 모델링 매핑

```{list-table}
:header-rows: 1
:widths: 42 58

* - 자리
  - 무엇
* - `ModelPoints.minimum_crediting_rate`
  - 최저 적립이율 — 계좌 크레딧 floor (`max(수익률, 보증이율)`).
    `0.0` = 0% 바닥, 양수 = 그 이율, 미설정 = 보증 없음
* - `fcf.NO_GUARANTEE_RATE`
  - "보증 없음" 의 이름값 (혼합북을 배열로 직접 만들 때만; 보통은 미설정)
* - `vfa.tvog(mp, basis, return_scenarios)`
  - 적립이율 보증의 시간가치만 따로 측정 (`measure` 의 `time_value` 가 세 보증을
    합산하는 것과 달리, 적립이율 보증을 격리)
```

```{admonition} 주의 — `0.0` 은 "보증 없음" 이 아니다
:class: warning

적립이율 보증에서 `minimum_crediting_rate = 0.0` 은 **진짜 0% 바닥** 입니다 (손실
난 달에도 계좌가 안 줄어드는 원금보존 보증). "보증을 끈다" 는 뜻이 아닙니다 —
보증을 끄려면 인자를 생략하거나 `fcf.NO_GUARANTEE_RATE` 를 씁니다.

GMDB / GMAB 의 보증액은 `0` 이 곧 "작동 안 함" 이라 `0 = 보증 없음` 이지만,
적립이율은 수익률이 음수가 될 수 있어 `0` 자체가 의미 있는 바닥이므로 규칙이
반대가 됩니다.
```

## 한 계약 — 0% floor 의 결정론 측정

GMDB / GMAB 는 끄고 (보증액 0), 0% 적립 floor 만 설정한 계약을 봅니다.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 10년, 일시납, 계좌가치 1억
- `minimum_crediting_rate = 0.0` (0% 원금보존), GMDB / GMAB 없음
- 펀드 연 수익률 3%, 변동수수료 2.5%
```

```python
import numpy as np
import fastcashflow as fcf

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
    issue_age              = 40,
    premium                = 0.0,
    term_months            = 120,
    account_value          = 1.0e8,
    minimum_crediting_rate = 0.0,   # 0% 적립 floor (원금보존), GMDB/GMAB 없음
)

det = fcf.vfa.measure(mp, basis)              # 결정론 (시나리오 없음)
print(f"CSM  = {det.csm[0]:>14,.0f}")
print(f"TVOG = {det.time_value[0]:>14,.0f}")  # 시간가치 (안 보임)
```

출력:

```text
CSM  =     17,715,366
TVOG =              0
```

중앙 시나리오에선 펀드 수익률이 매월 양수 (연 3%) 라 0% floor 가 한 번도
작동하지 않습니다 — 그래서 결정론 intrinsic 도, TVOG 도 0 입니다. 적립이율 보증의
비용은 **변동성** 에서만 나옵니다.

## 시나리오를 넣으면 — 매월 바닥을 메우는 비용

펀드 월수익률을 흔들면, 일부 경로의 마이너스 달마다 floor 가 0 으로 막아 주고
그 차액을 보험사가 부담합니다. 그 기대값이 시간가치입니다.

```python
rng  = np.random.default_rng(7)
r_m  = (1 + 0.03) ** (1 / 12) - 1                    # 중앙 월수익률 (연 3%)
vol  = 0.003                                         # 월 변동성 0.3% (보수적 펀드)
scen = r_m + vol * rng.standard_normal((2000, 120))  # (n_scenarios, n_time)

sto = fcf.vfa.measure(mp, basis, return_scenarios=scen)
print(f"measure.time_value = {sto.time_value[0]:>13,.0f}  (보증 전부 합산)")
print(f"CSM                = {sto.csm[0]:>13,.0f}  (시간가치 흡수 후 마진)")

tvog = fcf.vfa.tvog(mp, basis, scen)
print(f"vfa.tvog           = {tvog.time_value:>13,.0f}  (적립이율 보증만 격리)")
```

출력:

```text
measure.time_value =     2,707,099  (보증 전부 합산)
CSM                =    15,008,267  (시간가치 흡수 후 마진)
vfa.tvog           =     2,707,099  (적립이율 보증만 격리)
```

두 숫자가 같은 것은 우연이 아닙니다 — 둘은 **다른 것을 재는데** 이 계약에선 값이
겹칩니다.

- `measure.time_value` 는 그 계약의 **모든 보증** (적립이율 + GMDB + GMAB) 시간가치를
  **합한** 값입니다.
- `vfa.tvog` 는 그중 **적립이율 보증만** 떼어 봅니다.

이 계약엔 적립이율 보증밖에 없어 (GMDB / GMAB 없음) "모든 보증 합" 이 "적립이율만"
과 같을 수밖에 없습니다. GMDB / GMAB 까지 있으면 `measure.time_value` 가
`vfa.tvog` 보다 크고, 그 차이가 GMDB / GMAB 의 몫 (5.2) 입니다. 둘을 한 번에
나눠 보려면 `vfa.guarantee_tvog` (아래 "유니버설보험 계좌에서" 절) 를 씁니다.

월 변동성 0.3% (연 약 1% — 원금보존형이 까는 보수적 펀드) 만으로도 시간가치가
2,707,099 인 점도 눈여겨볼 대목입니다 — 뒤에서 변동성을 키워 봅니다.

## 결과 읽기 — 왜 매월 floor 가 비싼가

GMDB / GMAB 는 사망·만기 **한 시점** 에만 floor 가 물립니다. 적립이율 floor 는
**120개월 매월** 물립니다 — 마이너스 달마다 차액을 메워 주고, 한 번 막아 준 바닥이
다음 달 계좌의 출발점이 되어 그 효과가 달마다 쌓입니다. 그래서 같은 변동성이라도
단일 만기 floor 보다 자릿수로 비쌉니다.

변동성을 키워 보면 그 민감도가 드러납니다:

```python
for vol in (0.003, 0.005, 0.008):
    rng = np.random.default_rng(7)
    s = r_m + vol * rng.standard_normal((2000, 120))
    m = fcf.vfa.measure(mp, basis, return_scenarios=s)
    print(f"vol {vol:.3f}:  TVOG = {m.time_value[0]:>14,.0f}   CSM = {m.csm[0]:>14,.0f}")
```

출력:

```text
vol 0.003:  TVOG =      2,707,099   CSM =     15,008,267
vol 0.005:  TVOG =      8,108,297   CSM =      9,607,069
vol 0.008:  TVOG =     18,298,399   CSM =              0
```

월 변동성이 0.3% → 0.5% → 0.8% 로 오르자 시간가치가 2.7M → 8.1M → 18.3M 로
뛰고, 0.8% 에선 수수료 (~17.7M) 를 넘어 CSM 이 0 으로 소진됩니다 — onerous.
**5.2 의 GMAB 는 월 변동성 2% 에서 시간가치가 600만 남짓이었습니다**; 적립이율
floor 는 그 **1/4 수준의 변동성 (월 0.5%)** 만으로 이미 그 비용을 넘어섭니다
(위 표의 8.1M). 매월 작동하는
floor 를 깔 때는 백킹 펀드의 변동성을 훨씬 보수적으로 잡아야 하는 이유입니다.

## 변형

### 양수 floor / 보증 없음

`minimum_crediting_rate` 를 양수 (예: `0.0075`) 로 주면 매월 최소 그 이율을
적립하는 더 후한 보증이고, 시간가치는 더 큽니다. 인자를 **생략** 하면 보증
없음 — 시간가치가 0 입니다 (`vfa.tvog` 는 이 경우 측정할 적립이율 보증이 없어
거부합니다).

```python
mp_none = fcf.ModelPoints.single(
    issue_age=40,         # 가입연령
    premium=0.0,          # 거치형
    term_months=120,      # 10년
    account_value=1.0e8,  # 기초 계좌가치
)   # minimum_crediting_rate 생략 = 보증 없음
none = fcf.vfa.measure(mp_none, basis, return_scenarios=scen)
print(f"no-guarantee TVOG = {none.time_value[0]:>14,.0f}")
```

출력:

```text
no-guarantee TVOG =              0
```

### GMDB / GMAB 와 함께

세 보증을 다 설정하면 `measure` 의 `time_value` 는 셋의 시간가치를 합산합니다.
적립이율 보증만 따로 보려면 `vfa.tvog` 를, GMDB / GMAB 의 몫을 보려면 5.2 처럼
보증을 켜고 끄며 차분합니다.

### 유니버설보험 계좌에서 (COI 차감형)

위 예제의 계좌는 적립만 하는 변액계좌 (`account_value` 만) 였습니다.
[5.4](../account/universal-life) 의 **유니버설보험 계좌** — 매월 위험보험료 (COI)
를 순보장금액 (NAR) 에 물려 차감하는 계좌 — 에도 같은 적립이율 보증이 붙고,
`vfa.tvog` 가 **동일하게** 작동합니다. 닫힌형이 아니라 계좌를 시나리오마다 다시
굴려 (COI 피드백 포함) floor 가 떠받친 적립을 평가할 뿐, 호출은 똑같습니다.

```python
ul_mp    = fcf.samples.model_points("ul")   # COI 차감형 계좌 (적립이율 보증 2%)
ul_basis = fcf.samples.basis("ul")
n_time   = int(ul_mp.contract_boundary_months.max())
r_m      = (1 + ul_basis.investment_return) ** (1 / 12) - 1
rng      = np.random.default_rng(7)
scen     = r_m + 0.005 * rng.standard_normal((2000, n_time))

cr = fcf.vfa.tvog(ul_mp, ul_basis, scen)            # 적립이율 floor (유니버설 계좌)
print(f"vfa.tvog (적립이율 floor) = {cr.time_value:>16,.0f}")
```

출력:

```text
vfa.tvog (적립이율 floor) =        6,022,315
```

#### 두 보증을 한 번에 — `vfa.guarantee_tvog`

유니버설 계좌는 적립이율 floor 와 GMDB / GMAB floor 를 **동시에** 가질 수 있고,
둘은 계좌가치의 서로 다른 영역에서 작동해 (적립이율은 계좌를 아래에서 떠받치고,
GMDB / GMAB 는 계좌가 보증액에 못 미칠 때 차액을 메움) 시간가치가 **합산** 됩니다.
`vfa.guarantee_tvog` 는 둘을 한 번에 돌려줍니다:

```python
g = fcf.vfa.guarantee_tvog(ul_mp, ul_basis, scen)
print(f"적립이율 floor = {g.credited_rate_floor:>16,.0f}")
print(f"GMDB/GMAB floor = {g.account_floor:>16,.0f}")
print(f"합계          = {g.total:>16,.0f}")
```

출력:

```text
적립이율 floor =        6,022,315
GMDB/GMAB floor =          -33,520
합계          =        5,988,795
```

여기선 적립이율 floor (6.0M) 가 보증 시간가치를 지배합니다. GMDB / GMAB 몫이
작은 음수인 것은 정상입니다 — 이 계약은 계좌가 사망보장액보다 한참 낮아 GMDB 가
**결정론적으로 깊이 in-the-money** 라, 그 floor 의 *시간가치* 는 기초자산 수익률로
할인하는 VFA 기준에서 부호 제약이 없습니다 ([5.2](gmdb-gmab-tvog) 의 부호 설명).
`credited_rate_floor` 는 `vfa.tvog` 와, `account_floor` 는
`vfa.measure(..., return_scenarios=...).time_value` 의 합과 같습니다.

## 함정

### 함정 1 — 월 floor 와 연 floor 는 다르다

엔진의 적립이율 보증은 **매월** `max(월수익률, 월보증이율)` 로 작동하는 월 단위
바닥입니다. 실무의 "연 최저보증이율" 처럼 한 해 안에서 손익을 상계한 뒤 연말에 한
번 보는 보증과는 비용이 크게 다릅니다 (매월 막아 주는 쪽이 훨씬 비쌈 — 손익 상계
없이 손실 난 달마다 바닥을 메우니까). 백킹 펀드 변동성과 보증 수준을 그 전제 위에서
잡으세요.

### 함정 2 — `0.0` 은 끄는 값이 아니다

`minimum_crediting_rate = 0.0` 은 "보증 없음" 이 아니라 **0% 원금보존 보증**
입니다 (시간가치를 가짐). 끄려면 인자를 생략하거나 `fcf.NO_GUARANTEE_RATE` 를
씁니다. 금액 보증 (GMDB / GMAB) 의 `0 = 보증 없음` 과 규칙이 달라지는 자리입니다
([5.1](gmdb-gmab) 참조).

## 인접 레시피

- [5.1 변액보험 최저보증 — 결정론 측정](gmdb-gmab) — GMDB / GMAB floor 와 세
  보증의 "보증 없음" 표기 공통 원리.
- [5.2 최저보증의 시간가치 (TVOG)](gmdb-gmab-tvog) — GMDB / GMAB 의 시간가치.
  적립이율 floor 와 같은 `return_scenarios` 패스를 쓰지만 작동하는 자리가 다름.
- `vfa.trace(0, mp, basis, return_scenarios=scen)` — 계좌가치 궤적과
  floor 가 작동하는 자리, 시간가치를 트리로 확인.

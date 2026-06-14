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

[5.1](gmdb-gmab) · [5.2](gmdb-gmab-tvog) 의 GMDB / GMAB 는 사망·만기 **한 시점**
에 `max(계좌가치, 보증액)` 을 지급하는 floor 였습니다. 변액에는 세 번째 보증이
흔히 붙습니다 — **최저적립이율 보증** (크레딧 floor): 계좌가치 **자체** 를 매월
떠받칩니다. 앞 두 챕터는 이 보증을 일부러 꺼 뒀고 (GMDB / GMAB 만), 이 챕터가
그것만 따로 다룹니다.

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

```{admonition} `0.0` 과 "미설정" 은 다르다 — 5.1 의 공통 원리
:class: note

금액 보증 (GMDB / GMAB) 은 `0` 이 자연스러운 "작동하지 않는 바닥" 이라 `0 = 보증
없음`. 적립이율은 수익률이 음수가 될 수 있어 `0` 이 **진짜 0% 바닥** 이라,
"보증 없음" 은 0 이 아닌 별도 값 (`fcf.NO_GUARANTEE_RATE`) 으로 표시합니다.
왜 그런지는 [5.1 의 "보증 없음은 어떻게 표현하나"](gmdb-gmab) 절에서 세 보증을
한꺼번에 정리합니다.
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

## 시나리오를 넣으면 — 매월 래칫의 비용

펀드 월수익률을 흔들면, 일부 경로의 마이너스 달마다 floor 가 0 으로 막아 주고
그 차액을 보험사가 부담합니다. 그 기대값이 시간가치입니다.

```python
rng  = np.random.default_rng(7)
r_m  = (1 + 0.03) ** (1 / 12) - 1                    # 중앙 월수익률 (연 3%)
vol  = 0.003                                         # 월 변동성 0.3% (보수적 펀드)
scen = r_m + vol * rng.standard_normal((2000, 120))  # (n_scenarios, n_time)

sto = fcf.vfa.measure(mp, basis, return_scenarios=scen)
print(f"TVOG = {sto.time_value[0]:>14,.0f}")  # 적립이율 보증의 시간가치
print(f"CSM  = {sto.csm[0]:>14,.0f}")         # TVOG 흡수 후 마진

tvog = fcf.vfa.tvog(mp, basis, scen)                   # 적립이율 보증만 격리
print(f"vfa.tvog = {tvog.time_value:>14,.0f}")
```

출력:

```text
TVOG =      2,707,099
CSM  =     15,008,267
vfa.tvog =      2,707,099
```

월 변동성 0.3% (연 약 1% — 원금보존형이 까는 보수적 펀드) 만으로도 시간가치가
2,707,099 입니다. 이 계약엔 적립이율 보증밖에 없으니 `measure` 의 `time_value`
와 `vfa.tvog` 가 같은 값을 가리킵니다 — `vfa.tvog` 는 적립이율 보증을 격리해
보는 도구입니다.

## 결과 읽기 — 왜 매월 floor 가 비싼가

GMDB / GMAB 는 사망·만기 **한 시점** 에만 floor 가 물립니다. 적립이율 floor 는
**120개월 매월** 물립니다 — 마이너스 달마다 차액을 뱉어 내고, 그 바닥이 다음 달
출발점을 떠받쳐 래칫 (ratchet) 처럼 누적됩니다. 그래서 같은 변동성이라도 단일
만기 floor 보다 자릿수로 비쌉니다.

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

## 함정

### 함정 1 — 월 floor 와 연 floor 는 다르다

엔진의 적립이율 보증은 **매월** `max(월수익률, 월보증이율)` 로 작동하는 월 래칫
입니다. 실무의 "연 최저보증이율" 처럼 한 해 안에서 손익을 상계한 뒤 연말에 한 번
보는 보증과는 비용이 크게 다릅니다 (월 래칫이 훨씬 비쌈). 백킹 펀드 변동성과
보증 수준을 그 전제 위에서 잡으세요.

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

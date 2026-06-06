# 5.1 변액보험 최저보증 (GMDB / GMAB)

```{admonition} 이 챕터에서 배우는 것
:class: chapter-brief

- 변액보험이 보장형 (GMM) 과 다른 측정 모델 (VFA) 로 평가되는 이유 —
  계좌가치가 굴러가고 보험사는 수수료를 번다
- `account_value` + 최저보증 (`minimum_death_benefit` /
  `minimum_accumulation_benefit`) 을 모델 포인트에 거는 자리
- `vfa.measure` 의 결정론 측정 — 보증의 intrinsic value 가 BEL에 들어가는 모습
- `return_scenarios` 를 넣으면 드러나는 보증의 시간가치 (TVOG = Time Value of Options and Guarantees) — 단일
  결정론 run 에는 보이지 않는 비용
```

지금까지의 보장형 (GMM) 상품은 위험률 × 보험금으로 청구를 계산했습니다.
변액보험은 다릅니다 — 계약자의 계좌가치가 펀드 수익률로 굴러가고, 사망 /
해지 / 만기에 그 계좌가치를 (또는 최저보증을) 지급합니다. 보험사의 이익은
계좌에서 떼는 수수료입니다. IFRS 17 은 이런 직접참가 계약을 **VFA
(Variable Fee Approach, 변동수수료접근법)** 로 측정합니다 — `vfa.measure`.

## 상품 소개 — 변액보험과 최저보증

**변액보험** (variable insurance) 은 보험료가 펀드에 투자되어 계좌가치
(account value) 가 시장 수익률로 변동합니다. 계약자가 투자위험을 지는 대신,
보험사는 **최저보증** 을 얹어 팝니다:

- **GMDB** (Guaranteed Minimum Death Benefit, 최저사망보증) — 사망 시
  `max(계좌가치, 보증액)` 지급. 계좌가 보증액 아래로 떨어져도 사망보험금은
  보증액 이상.
- **GMAB** (Guaranteed Minimum Accumulation Benefit, 최저적립보증) — 만기
  생존 시 `max(계좌가치, 보증액)` 지급.

보증액을 넘는 초과분 (`보증액 - 계좌가치`, 양수일 때) 이 보험사가 부담하는
보증 비용입니다. 계좌가 보증을 웃돌면 초과분은 0 — 보증은 계좌가치에 대한
**풋옵션** 처럼 하락에만 비용이 발생합니다.

## 모델링 매핑 — VFA

```{list-table}
:header-rows: 1
:widths: 40 60

* - 자리
  - 무엇
* - `vfa.measure(mp, basis)`
  - VFA 측정 (보장형의 `measure` (GMM) 가 아님)
* - `ModelPoints.account_value`
  - 가입 시 계좌가치
* - `ModelPoints.minimum_death_benefit`
  - GMDB 보증액. 기본 0 = 보증 없음 (`max(AV, 0) = AV`)
* - `ModelPoints.minimum_accumulation_benefit`
  - GMAB 보증액. 기본 0 = 보증 없음
* - `ModelPoints.minimum_crediting_rate`
  - 최저 적립이율 — 계좌 크레딧 floor (`max(수익률, 보증이율)`)
* - `Basis.investment_return`
  - 기초자산(펀드) 수익률 — VFA의 할인·적립 basis
* - `Basis.fund_fee`
  - 보험사가 떼는 변동수수료 (= 이익원)
* - `Basis.expense_cv`
  - 위험조정 (변액의 비금융위험 = 사업비위험) 의 변동계수
```

핵심: **사망·만기 exit 은 `max(계좌가치, 보증)`, 해지 exit 은 계좌가치 그대로.**
보험사의 이익 = 수수료 현재가치 = 가입 시 CSM. 보증 비용이 그 마진을 갉아먹습니다.

## 한 계약 — 결정론 측정과 시나리오

계약 하나로 보증 비용이 두 단계로 드러나는 것을 봅니다 — 먼저 결정론
(intrinsic), 그 다음 시나리오 (시간가치).

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 10년 (120개월), 일시납 변액계약
- 계좌가치 1억, GMDB 1.02억 (102%), GMAB 1.05억 (105%)
- 펀드 연 수익률 6%, 변동수수료 연 2.5%, 사망 0.5% / 해지 4%
```

```python
import numpy as np
import fastcashflow as fcf

# 산출기초
death_fn = lambda s, a, d: np.full(np.shape(d), 0.005)   # 연 0.5% 사망률
lapse_fn = lambda s, a, d: np.full(np.shape(d), 0.04)    # 연 4% 해지율
basis = fcf.Basis(
    mortality_annual  = death_fn,   # 보유계약 감쇠용 사망률
    lapse_annual      = lapse_fn,   # 해지율
    discount_annual   = 0.03,       # 연 할인율 (비보증 현금흐름)
    ra_confidence     = 0.95,       # 위험조정 신뢰수준 95%
    mortality_cv      = 0.10,       # 사망률 변동계수
    expense_cv        = 0.10,       # 사업비 변동계수 (VFA의 RA = 사업비위험)
    investment_return = 0.06,       # 기초자산(펀드) 연 수익률
    fund_fee          = 0.025,      # 변동수수료 연 2.5% (= 보험사 이익원)
)

# 모델 포인트 (변액계약 하나: 계좌 1억, GMDB 1.02억, GMAB 1.05억)
mp = fcf.ModelPoints.single(
    issue_age                    = 40,      # 가입연령
    premium                      = 0.0,     # 일시납 (계좌가치로 납입)
    term_months                  = 120,     # 보험기간 10년
    account_value                = 1.0e8,   # 가입 시 계좌가치
    minimum_death_benefit        = 1.02e8,  # GMDB 최저사망보증 (102%)
    minimum_accumulation_benefit = 1.05e8,  # GMAB 최저적립보증 (105%)
)

det = fcf.vfa.measure(mp, basis)                # 결정론 측정 (intrinsic 만)
print(f"BEL  = {det.bel[0]:>14,.0f}")           # 계좌가치 차감 순부채
print(f"fee  = {det.variable_fee[0]:>14,.0f}")  # 수수료 현재가치 (이익원)
print(f"CSM  = {det.csm[0]:>14,.0f}")           # 미실현 수수료 - 보증 intrinsic
print(f"TVOG = {det.time_value[0]:>14,.0f}")    # 시간가치 (시나리오 없으면 0)

# 펀드 수익률 시나리오 (외부 ESG 산출: 1,000 경로 x 120 개월)
rng  = np.random.default_rng(7)
r_m  = (1 + 0.06) ** (1 / 12) - 1                      # 중앙 월수익률
scen = r_m + 0.005 * rng.standard_normal((1000, 120))  # (n_scenarios, n_time)

sto = fcf.vfa.measure(mp, basis, return_scenarios=scen)  # intrinsic + 시간가치
print(f"\nTVOG = {sto.time_value[0]:>14,.0f}")           # 보증의 시간가치 (변동성 비용)
print(f"CSM  = {sto.csm[0]:>14,.0f}")                    # TVOG 흡수 후 마진
```

출력:

```
BEL  =    -17,610,124
fee  =     17,826,387
CSM  =     17,610,124
TVOG =              0

TVOG =      3,433,960
CSM  =     14,176,164
```

**결정론 run 은 보증의 intrinsic value (중앙 시나리오에서의 비용) 만 봅니다.**
6% 성장 가정이면 계좌가 빠르게 보증액 (102% / 105%) 을 넘어 floor 가 거의
안 물립니다 — 그래서 intrinsic 은 작습니다 (수수료 17.83M 와 CSM 17.61M 의
차이 ≈ 0.22M 뿐). 단일 결정론 run 만 보면 "보증이 거의 공짜" 라는 틀린
결론에 이릅니다.

**시나리오를 넣으면 보증의 진짜 비용 — 시간가치 (TVOG) 가 드러납니다.**
변동성 때문에 일부 경로에서 계좌가 보증액 아래로 떨어지고, 그 풋옵션 비용의
기대값이 시간가치입니다. 여기서 3.43M — intrinsic 의 15배가 넘습니다. 이
비용이 CSM을 17.61M → 14.18M 로 낮춥니다 (계약은 여전히 이익: CSM > 0).

```{admonition} 시나리오는 외부에서 — fastcashflow 는 엔진
:class: note

`return_scenarios` 는 당신의 ESG (Economic Scenario Generator, 경제 시나리오
생성기) 산출물입니다. fastcashflow 는 시나리오를 생성 하지 않고, 받아 보증
비용을 평가합니다. 한 변수 (여기선 펀드 월수익률) 의 `(n_scenarios, n_time)`
배열로 건네면 됩니다 — 열 수는 투영 개월수와 같아야 합니다.
```

```{admonition} TVOG의 부호는 보장되지 않음
:class: warning

VFA는 위험중립 measure 가 아니라 기초자산 수익률 로 할인합니다. 그래서
floor 의 시간가치는 부호가 고정이 아닙니다 — 깊은 in-the-money (보증액이 계좌가치보다 한참 높은) 보증은
변동성이 오히려 일부 시나리오를 floor 위로 끌어올려 비용을 낮춰 시간가치가
음수일 수도 있습니다. 위험중립 풋옵션의 "시간가치 >= 0" 직관이 여기선 그대로
통하지 않습니다.
```

## 결과 읽기 — 수수료 vs 보증 비용

VFA 측정의 한 줄 요약: **보험사는 수수료를 벌고, 보증 비용이 그 마진을 깎는다.**

- **BEL** 은 계좌가치를 차감한 순액입니다. 보험사가 들고 있는 계좌는 계약자
  몫 (부채) 이자 운용 자산이라, 그 둘이 상쇄되고 남는 게 BEL.
- **CSM** = 미실현 수수료 − 보증 비용. 위 예제에서 수수료 17.83M 가 보증
  intrinsic (~0.2M) 과 시간가치 (3.43M) 를 흡수하고도 14.18M 남습니다.
- **시간가치가 수수료를 넘으면** CSM이 0 으로 깎이고 손실요소
  (loss component) 가 잡힙니다 — onerous 변액계약.

## 변형

### per-MP 보증률 + 시나리오는 v1 미지원

`return_scenarios` 의 시간가치 패스는 v1 에서 `minimum_crediting_rate` (적립
보증이율) 가 portfolio 전체에서 동일해야 합니다 — per-MP로 다른 적립보증률에
stochastic 을 거는 건 미래 확장입니다. **GMDB / GMAB 보증액 자체는 계약마다
달라도** 됩니다 (`max(AV, 보증)` 의 floor 는 per-MP).

### onerous 변액 만들어 보기

위 예제에서 시나리오 변동성 (`0.005`) 을 키우거나 보증액을 올리면 시간가치가
수수료를 넘어 CSM = 0, 손실요소 양수가 됩니다. 후한 보증 + 높은 변동성 +
얇은 수수료의 조합이 변액 보증의 손실부담 신호입니다.

### 적립이율 보증 (크레딧 floor)

`minimum_crediting_rate` 를 0 보다 크게 주면 계좌가 매월 `max(수익률, 보증
이율)` 로 크레딧됩니다 — 계좌가치 자체를 떠받치는 또 다른 보증이고, 그
시간가치도 같은 `return_scenarios` 패스가 함께 흡수합니다.

## 함정

### 함정 1 — `measure` 가 아니라 `vfa.measure`

변액은 보장형의 `measure` (GMM) 가 아니라 `vfa.measure` (VFA) 로
측정합니다. GMM으로 돌리면 계좌가치 mechanic 이 없어 결과가 틀립니다.

### 함정 2 — 결정론만 보고 "보증이 싸다" 판단

중앙 성장 가정에선 floor 가 거의 안 물려 intrinsic 이 작게 나옵니다. 보증의
진짜 비용 (시간가치) 은 `return_scenarios` 를 넣어야 드러납니다 — 단일
결정론 run 에는 원리상 보이지 않습니다.

### 함정 3 — 보증액 기본 0

`minimum_death_benefit` / `minimum_accumulation_benefit` 를 안 주면 0
입니다 (`max(AV, 0) = AV` = 보증 없음). 보증을 평가하려면 보증액을 명시적으로
줘야 합니다.

## 인접 레시피

- [2.1 정기보험](../simple/term-life) — 보장형 (GMM) 측정의 출발점. 변액은
  같은 in-force 감쇠 위에 계좌가치·수수료·보증을 얹은 다른 측정 모델.
- `vfa.trace(mp_index, mp, basis)` — 이 계약의 계좌가치 궤적, GMDB / GMAB
  floor 가 무는 자리, BEL / CSM 계산 경로를 트리로 확인 (GMM의 `gmm.trace` 에
  대응하는 VFA 버전; `return_scenarios` 를 주면 TVOG까지). 보장형 계약의
  `gmm.trace` 는 [검증 패턴](../workflow/validation) 챕터.

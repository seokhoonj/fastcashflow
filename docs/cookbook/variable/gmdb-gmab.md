# 5.1 변액보험 최저보증 — 결정론 측정 (GMDB / GMAB)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 변액보험이 보장형 (GMM) 과 다른 측정 모델 (VFA) 로 평가되는 이유 —
  계좌가치가 굴러가고 보험사는 수수료를 번다
- `account_value` + 최저보증 (`minimum_death_benefit` /
  `minimum_accumulation_benefit`) 을 모델 포인트에 설정하는 자리
- `vfa.measure` 의 **결정론 측정** — 보증의 intrinsic value (중앙 시나리오
  비용) 가 CSM 을 얼마나 갉아먹는가
- 보증을 켜고 끄는 차이로 GMDB / GMAB 의 비용을 각각 분리하는 법
- 결정론 run 의 `TVOG = 0` — 보증의 **시간가치** 는 여기 안 보이고, 다음
  챕터 [5.2 시간가치](gmdb-gmab-tvog) 에서 시나리오로 드러난다
```

지금까지의 보장형 (GMM) 상품은 위험률 x 보험금으로 청구를 계산했습니다.
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
* - `Basis.investment_return`
  - 기초자산(펀드) 수익률 — VFA의 할인·적립 basis
* - `Basis.fund_fee`
  - 보험사가 떼는 변동수수료 (= 이익원)
* - `Basis.expense_cv`
  - 위험조정 (변액의 비금융위험 = 사업비위험) 의 변동계수
```

핵심: **사망·만기 exit 은 `max(계좌가치, 보증)`, 해지 exit 은 계좌가치 그대로.**
보험사의 이익 = 수수료 현재가치 = 가입 시 CSM. 보증 비용이 그 마진을 갉아먹습니다.

```{admonition} 이 챕터는 GMDB / GMAB 만 — 적립이율 보증은 별개
:class: note

모델포인트에는 적립이율 보증 (`minimum_crediting_rate`, 계좌 크레딧
floor) 자리도 있지만, 이 챕터와 [5.2 시간가치](gmdb-gmab-tvog) 챕터는
**둘 다 그 자리를 비워둡니다** (보증 없음). 적립이율 보증은 계좌가치
**자체** 를 매월 떠받치는 다른 종류의 보증이고, GMDB / GMAB 의 사망·만기
floor 와 비용 구조가 다릅니다. 입력으로 "보증 없음" 을 표현하는 법은 아래
**변형** 절의 "보증 없음은 어떻게 표현하나" 에서 세 보증을 한꺼번에
정리합니다.
```

## 한 계약 — 결정론 측정

계약 하나로 보증의 **intrinsic value** (중앙 시나리오에서 floor 가 무는
비용) 가 CSM 에 들어가는 모습을 봅니다.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 10년 (120개월), 일시납 변액계약
- 계좌가치 1억, GMDB 1.02억 (102%), GMAB 1.05억 (105%)
- 펀드 연 수익률 3%, 변동수수료 연 2.5% (순성장 +0.5%/년), 사망 0.5% / 해지 4%

수익률 3% 에서 수수료 2.5% 를 빼면 계좌는 연 0.5% 로 거의 평탄하게 굴러가,
102% / 105% 보증이 실제로 무는 구간을 만듭니다. 6% 처럼 빠른 성장 가정이면
계좌가 보증을 금세 넘어 floor 가 거의 안 물리고, intrinsic 은 0 에 가깝게
보입니다.
```

```python
import numpy as np
import fastcashflow as fcf

# 산출기초
death_rate = 0.005  # 연 0.5% 사망률
lapse_rate = 0.04   # 연 4% 해지율
basis = fcf.Basis(
    mortality_annual  = death_rate,  # 보유계약 사망률 (in-force 감쇠)
    lapse_annual      = lapse_rate,  # 해지율
    discount_annual   = 0.03,        # 연 할인율 (비보증 현금흐름)
    ra_confidence     = 0.95,        # 위험조정 신뢰수준 95%
    mortality_cv      = 0.10,        # 사망률 변동계수
    expense_cv        = 0.10,        # 사업비 변동계수 (VFA의 RA = 사업비위험)
    investment_return = 0.03,        # 기초자산(펀드) 연 수익률 3%
    fund_fee          = 0.025,       # 변동수수료 연 2.5% (= 보험사 이익원)
)

# 변액계약 하나 (계좌 1억, GMDB 1.02억, GMAB 1.05억)
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
print(f"CSM  = {det.csm[0]:>14,.0f}")           # 미실현 수수료 - 보증 비용
print(f"TVOG = {det.time_value[0]:>14,.0f}")    # 시간가치 (시나리오 없으면 0)
```

출력:

```text
BEL  =    -17,664,772
fee  =     17,737,198
CSM  =     17,664,772
TVOG =              0
```

**결정론 run 은 보증의 intrinsic value — 중앙 시나리오에서 floor 가 무는
비용 — 만 봅니다.** 계좌 궤적을 펼치면 어디서 무는지 보입니다:

```python
av = det.account_value_path[0]                  # 계좌가치 궤적 (월말, 121개)
print("month       account    GMDB    GMAB")
for m in (0, 24, 48, 60, 72, 96, 120):
    gmdb = "물림" if av[m] < 1.02e8 else " - "
    gmab = "물림" if av[m] < 1.05e8 else " - "
    print(f"{m:5d}  {av[m]:>14,.0f}   {gmdb}    {gmab}")
```

출력:

```text
month       account    GMDB    GMAB
    0     100,000,000   물림    물림
   24     100,967,707   물림    물림
   48     101,944,779   물림    물림
   60     102,436,855    -     물림
   72     102,931,306    -     물림
   96     103,927,380    -     물림
  120     104,933,092    -     물림
```

계좌는 연 0.5% 로 100,000,000 → 104,933,092 로 굴러갑니다.

- **GMDB (102%)** 는 계좌가 102,000,000 을 넘는 5년차 (month 60) 전까지만
  무립니다. 그 구간의 사망자에게만 보증 top-up 이 나갑니다.
- **GMAB (105%)** 는 만기 계좌 104,933,092 가 105,000,000 에 못 미쳐 **만기까지
  내내 ITM (in-the-money, 보증액이 계좌가치보다 높은 상태)** — 만기 생존자
  전원에게 보증 top-up 이 나갑니다.

## 결과 읽기 — 보증 비용을 GMDB / GMAB 로 분리

VFA 측정의 한 줄 요약: **보험사는 수수료를 벌고, 보증 비용이 그 마진을 줄인다.**

- **BEL** 은 계좌가치를 차감한 순액입니다. 보험사가 들고 있는 계좌는 계약자
  몫 (부채) 이자 운용 자산이라, 그 둘이 상쇄되고 남는 게 BEL.
- **CSM** = 미실현 수수료 - 보증 비용. 위에서 수수료 17,737,198 이 보증
  비용을 흡수하고도 17,664,772 남았습니다.

보증 비용 (intrinsic) 이 정확히 얼마인지는 **보증을 끄고 다시 재보면** 분리됩니다 —
보증을 켜서 CSM 이 떨어진 만큼이 그 보증의 비용입니다:

```python
def csm_with(gmdb, gmab):
    m = fcf.ModelPoints.single(
        issue_age=40, premium=0.0, term_months=120, account_value=1.0e8,
        minimum_death_benefit=gmdb, minimum_accumulation_benefit=gmab,
    )
    return fcf.vfa.measure(m, basis).csm[0]

csm_off  = csm_with(0.0,    0.0)     # 두 보증 모두 끔
csm_gmdb = csm_with(1.02e8, 0.0)     # GMDB 만
csm_gmab = csm_with(0.0,    1.05e8)  # GMAB 만
csm_both = csm_with(1.02e8, 1.05e8)  # 둘 다

print(f"보증 없음 CSM    = {csm_off:>14,.0f}")
print(f"GMDB intrinsic  = {csm_off - csm_gmdb:>14,.0f}")
print(f"GMAB intrinsic  = {csm_off - csm_gmab:>14,.0f}")
print(f"둘 다 intrinsic  = {csm_off - csm_both:>14,.0f}")
```

출력:

```text
보증 없음 CSM    =     17,715,366
GMDB intrinsic  =         19,113
GMAB intrinsic  =         31,481
둘 다 intrinsic  =         50,594
```

GMDB 19,113 + GMAB 31,481 = 50,594 — 두 보증 비용이 정확히 합산됩니다.
이 계약에서 보증의 결정론 비용은 50,594 뿐 — 수수료 17,737,198 에 견주면
0.3% 입니다.

```{admonition} fee - CSM 은 72,426 인데 보증 비용은 왜 50,594 인가
:class: note

위 출력의 수수료 17,737,198 에서 CSM 17,664,772 을 빼면 72,426 이지만, 보증을
켜고 끄는 차분으로 분리한 보증 비용은 50,594 입니다. 차이 21,832 는 보증과
무관한 **수수료 타이밍 효과** 입니다 — 변동수수료는 월말 생존자에게만 부과되는데,
월중 사망·해지자는 그 달 수수료가 빠지기 전 계좌가치를 받아, 걷지 못한 수수료
만큼이 마진에서 샙니다. 이 효과는 보증을 켜든 끄든 똑같이 있어 차분에서
상쇄되므로, 순수 보증 비용은 50,594 입니다. 보증 비용을 측정할 때는 raw
fee - CSM 이 아니라 **floor 를 켜고 끈 차분** 을 봐야 합니다.
```

```{admonition} GMAB intrinsic 31,481 을 손으로 확인
:class: note

GMAB 는 만기 (month 120) 생존자에게 `max(계좌, 105,000,000)` 을 지급합니다.

- 만기 계좌 = 104,933,092 → 보증 top-up = 105,000,000 - 104,933,092 = **66,908** (계약당)
- 만기 생존율 = `(1 - 0.005 - 0.04)^10` ~ **0.632** (연 사망 0.5% + 해지 4% 감쇠)
- 10년 할인 = `1.03^-10` = **0.744**

66,908 x 0.632 x 0.744 ~ **31,460** ~ 엔진의 31,481. GMDB 는 처음 5년 사망자에게만,
그것도 점점 줄어드는 top-up 으로 무니 훨씬 작은 19,113.
```

```{admonition} 결정론만 보고 "보증이 싸다" 판단하면 안 됩니다
:class: warning

위 50,594 는 **중앙 시나리오 하나** 에서의 비용입니다. 펀드 수익률이
변동하면 일부 경로에서 계좌가 보증액 아래로 더 깊이 떨어지고, 그 풋옵션
비용의 기대값 — **시간가치 (TVOG)** — 은 단일 결정론 run 에 원리상 보이지
않습니다 (`TVOG = 0`). 같은 계약의 시간가치는 [5.2](gmdb-gmab-tvog) 에서
시나리오로 측정하며, intrinsic 50,594 의 120배가 넘습니다.
```

## 변형

### per-MP 보증액

`minimum_death_benefit` / `minimum_accumulation_benefit` 는 계약마다 달라도
됩니다 — `max(AV, 보증)` 의 floor 는 per-MP 로 계산됩니다. 포트폴리오를 numpy
배열로 직접 만들면 계약별로 다른 보증액을 한 번에 평가합니다.

### 보증 없음은 어떻게 표현하나 — 세 보증 공통 원리

GMDB · GMAB · 적립이율 보증은 전부 같은 모양입니다 — **바닥(floor)**:

```text
GMDB     :  사망보험금 = max(계좌가치, minimum_death_benefit)
GMAB     :  만기보험금 = max(계좌가치, minimum_accumulation_benefit)
적립이율 :  적립율     = max(수익률,   minimum_crediting_rate)
```

그래서 **"보증 없음" = "절대 안 무는 바닥"** 이라는 원리도 셋이 같습니다. 다만
**그 바닥이 0일 때 정말 안 무느냐** 가 금액 보증과 율 보증에서 갈립니다.

* **GMDB / GMAB (금액 바닥)** — 계좌가치는 항상 0 이상이라 `max(계좌가치, 0) =
  계좌가치`. 즉 **0이 자연스럽게 "안 무는 바닥"** 이라 `0 = 보증 없음`.
  "보험금이 0 이상" 을 보장하는 상품은 없으니, 0을 "끔" 으로 써도 잃는 표현이
  없습니다.
* **적립이율 (율 바닥)** — 수익률은 음수가 될 수 있어 `max(수익률, 0) !=
  수익률`. 즉 **0이 진짜 0% 바닥** (마이너스 달에 계좌를 떠받치는 실제 상품)
  이라, 0을 "끔" 으로 쓸 수 없습니다. 그래서 "보증 없음" 은 0 이 아닌 별도
  값으로 표시합니다.

입력은 이렇게 합니다:

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - 보증
  - 보증 없음
  - 실제 보증
* - GMDB / GMAB (금액)
  - `0` 또는 미설정
  - 보증액 (예: `1.1e8`)
* - 적립이율 (율)
  - 미설정 / 빈칸
  - `0.0` = 0% 바닥, 또는 양수 (예: `0.0075`)
```

* **CSV** — 적립이율 칸을 **빈칸** 으로 두면 보증 없음. 0% / 양수 바닥은 숫자로
  명시.
* **`ModelPoints.single(...)`** — `minimum_crediting_rate` 인자를 **생략** 하면
  보증 없음 (이 챕터의 예제가 그렇습니다).
* **배열로 직접 (혼합북)** — 일부는 보증, 일부는 무보증인 포트폴리오를 numpy
  배열로 만들 때만, 무보증 행에 이름값 `fcf.NO_GUARANTEE_RATE` 를 넣습니다:

```text
minimum_crediting_rate = [fcf.NO_GUARANTEE_RATE,  0.0,    0.0075]
#                          보증 없음              0% 바닥  0.75% 바닥
```

일상 입력에서 이 값을 숫자로 직접 쓸 일은 없습니다 — 빈칸이나 인자 생략이면
충분합니다.

```{admonition} 왜 0 이 아닌 별도 값인가 — 속도 때문
:class: note

엔진은 모든 모델포인트를 한 배열 식 `max(값, 바닥)` 으로 **동시에** 계산합니다
(행마다 식을 바꾸면 루프가 되어 느려짐). 그래서 "이 행은 보증 없음" 을 식에서
빼는 대신, **절대 안 무는 바닥** (사실상 음의 무한대) 을 깔아 `max` 를
무력화합니다. 음의 무한대는 배열에 저장할 수 없으니, 그것을 가리키는 안전한
유한 표지값 `fcf.NO_GUARANTEE_RATE` 를 둔 것입니다. 금액 보증은 0 자체가 이미
"안 무는 바닥" 이라 이런 표지값이 필요 없었습니다.
```

## 함정

### 함정 1 — `measure` 가 아니라 `vfa.measure`

변액은 보장형의 `measure` (GMM) 가 아니라 `vfa.measure` (VFA) 로
측정합니다. GMM으로 돌리면 계좌가치 mechanic 이 없어 결과가 틀립니다.

### 함정 2 — 보증액 기본 0

`minimum_death_benefit` / `minimum_accumulation_benefit` 를 안 주면 0
입니다 (`max(AV, 0) = AV` = 보증 없음). 보증을 평가하려면 보증액을 명시적으로
줘야 합니다.

### 함정 3 — 결정론 intrinsic 을 보증의 전체 비용으로 착각

결정론 run 의 intrinsic (여기선 50,594) 은 보증 비용의 **일부** — 중앙
시나리오 몫 — 일 뿐입니다. 보증의 진짜 비용은 여기에 **시간가치 (TVOG)** 를
더한 것이고, 시간가치는 시나리오를 넣어야 드러납니다 ([5.2](gmdb-gmab-tvog)).

## 인접 레시피

- [5.2 변액보험 최저보증의 시간가치 (TVOG)](gmdb-gmab-tvog) — 같은 계약에
  `return_scenarios` 를 넣어 보증의 시간가치를 측정. 결정론 intrinsic 50,594
  대 시간가치 6백만의 분해.
- [2.1 정기보험](../simple/term-life) — 보장형 (GMM) 측정의 출발점. 변액은
  같은 in-force 감쇠 위에 계좌가치·수수료·보증을 얹은 다른 측정 모델.
- `vfa.trace(mp_index, mp, basis)` — 이 계약의 계좌가치 궤적, GMDB / GMAB
  floor 가 무는 자리, BEL / CSM 계산 경로를 트리로 확인 (GMM의 `gmm.trace` 에
  대응하는 VFA 버전). 보장형 계약의 `gmm.trace` 는 [검증 패턴](../workflow/validation) 챕터.

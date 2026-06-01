# 3.2 paid-up 분리 (3-state)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 완납 (paid-up) 를 active / waiver 와 **별도 상태** 로 두는 이유 —
  유일한 실무 동기는 *납입후 해지율이 납입중과 달라지는 것*
- 그 방향은 상품에 따라 갈린다 — **보장성** (환급금 없음) 은 완납 후
  해지율이 *낮아지고*, **저축성·고환급** 은 완납 시점에 해약이 *spike*
- 3-state 모델 `STATE_MODELS["WAIVER_PAIDUP"]` 의 wiring 과
  `lapse_paidup_annual` 자리
- paidup 상태는 **전이로 들어가지 않는다** — model point 를 paidup 으로
  *자리 지정 (seating)* 하는, 즉 "이미 완납된 보유계약을 결산 시점에
  평가" 하는 시나리오
- 환급금이 없을 때 해지율이 낮을수록 runoff 가 느려 BEL 이 *커지는* 메커니즘
```

[3.1 납입면제](waiver) 의 2-state 모델은 active / waiver 만 다뤘습니다.
이 챕터는 **완납 (paid-up)** 를 세 번째 상태로 추가합니다 — 보험료
납입이 끝난 계약을 별도로 추적하는 자리입니다.

## 상품 소개 — 완납 상태

한국 상품은 대개 **납입기간** (예: 20년납) 이 **보험기간** (예: 종신)
보다 짧습니다. 납입기간이 끝나면 계약은 보험료를 더 내지 않으면서 보장은
그대로 유지하는 **완납 (paid-up)** 상태가 됩니다.

완납을 별도 상태로 나누는 **유일한 실무 이유는 해지율이 납입중과
달라지기 때문** 입니다. 납입중과 납입후에 같은 해지율을 쓴다면 굳이 상태를
나눌 이유가 없습니다 — 나누는 의미는 두 상태에 **서로 다른 해지율** 을
거는 데 있습니다.

그 **방향은 상품에 따라 갈립니다**:

- **보장성 (환급금 없음)** — 완납 후 해지율이 *낮아집니다*. 보험료 부담이
  사라지고 보장이 사실상 "공짜" 로 유지되니 나갈 유인이 거의 없고, 납입을
  끝까지 버틴 계약자라는 self-selection (완납까지 유지한 집단은 애초에 잘
  안 나감) 효과도 같은 방향입니다.
- **저축성·고환급** — 완납 *시점* 에 해약이 *spike* 합니다. 환급률이 정점에
  닿아 적립된 가치를 빼서 나가는 cash-out 행동. 완납 직후 한 번 튀고 이후
  안정되는 형태라, 평탄 상수가 아니라 경과 의존이 자연스럽습니다.

이 챕터의 예제는 **보장성** (사망보험금만, 환급금 없음) 이라 완납 후
해지율이 *낮아지는* 쪽을 씁니다. 저축성 spike 의 wiring 은 아래 **변형**
절에서 다룹니다.

엔진 관점에서 한 계약은 세 상태 중 하나에 있습니다:

- **active** — 정상 납입 중. 사망 / 해지 / *납입면제 진입* 에 노출.
- **waiver** — 납입면제 상태 ([3.1](waiver) 참조).
- **paidup** — 완납. 보험료를 안 내고 보장은 계속. 사망 *과* 납입후
  해지로 빠져나감 — 이 해지율이 active 와 다르다는 게 핵심.

완납을 엔진에서 다루는 길은 **측정 대상이 신계약이냐 보유계약이냐** 에 따라
갈립니다. 이 구분이 paid-up 모델을 이해하는 핵심입니다.

```{list-table}
:header-rows: 1
:widths: 22 42 36

* - 측정 대상
  - 납입후 해지율을 다루는 법
  - paidup 상태
* - **신계약** (가입 시점부터 측정)
  - `lapse_annual` 을 가입경과 (policy duration) 의 단계함수로 — 납입기간
    동안과 이후에 다른 값. 보험료 중단은 `premium_term_months` 가 처리.
  - **불필요** (active 한 상태로 끝)
* - **보유계약** (이미 완납된 납입후 코호트의 결산)
  - `state = STATE_PAIDUP` 으로 **자리 지정** 해 잔여 기간만 평가
  - **필요** — 이 챕터의 예제
```

```{admonition} 왜 신계약은 paidup 상태로 "전이" 하지 않나
:class: important

엔진의 상태 전이는 모두 **확률적 (rate-driven)** 입니다 — "몇 월이 되면
무조건 active -> paidup" 같은 *시점 트리거 전이* (특정 시점에 확률이 아니라
결정론적으로 일어나는 전이) 가 없습니다. 그래서 active 로 시작한 신계약을
`term_months` 만큼 굴려도 **자동으로 paidup 으로 넘어가지 않고**, 끝까지
active 상태에 남아 `lapse_annual` 을 계속 씁니다 (보험료만
`premium_term_months` 로 멈춤). 따라서:

- **신계약** 에서 납입후 해지율 하락을 반영하려면 `lapse_annual` 자체를
  가입경과 의존 단계함수로 줍니다 (아래 **변형** 참조). paidup 상태가
  필요 없습니다.
- **paidup 상태 + 자리 지정** 은 *이미 완납된 보유계약* 을 결산일에 평가할
  때의 도구입니다 — 지나간 납입기간을 다시 굴리지 않고 잔여 구간만 납입후
  해지율로 봅니다. 이 챕터의 예제가 이 경우입니다. (보유계약 평가의 입력
  구조는 [튜토리얼 11장](../../tutorial/11-in-practice) 참조.)

두 경로는 **서로 다른 측정 모드** 라 바꿔 쓸 수 없습니다 — 신계약 CSM 을
paidup 자리 지정으로 구하면 납입기간을 건너뛴 꼬리만 평가돼 틀립니다.
```

## 모델링 매핑 — 3-state

```{list-table}
:header-rows: 1
:widths: 32 68

* - 자리
  - 무엇
* - `Basis.state_model`
  - `STATE_MODELS["WAIVER_PAIDUP"]` — active / waiver / paidup 3-state 모델
* - `Basis.lapse_annual`
  - active 상태의 해지율 (납입중)
* - `Basis.lapse_paidup_annual`
  - paidup 상태의 해지율 (납입후). 지정 안 하면 `lapse_annual` 로 fallback
* - `Basis.waiver_incidence_annual`
  - active -> waiver 연 전이율
* - `ModelPoints.state`
  - 각 계약의 시작 상태. 완납 보유계약은 `STATE_PAIDUP` 으로 자리 지정
```

핵심은 **해지율이 상태별로 갈린다** 는 점입니다. `lapse_annual` 은
active 에만, `lapse_paidup_annual` 은 paidup 에만 적용됩니다. 사망률은
세 상태 공통입니다.

## 한 계약 — 손계산과 엔진

납입후 해지율의 효과를 또렷이 보려고, **완납된 보장성 계약** 을 평가합니다
(보험료 = 0, 사망보험금만, 환급금 없음). 여기서는 두 번 측정합니다 — 한
번은 완납 상태의 납입후 해지율 (월 2%) 로, 한 번은 같은 계약을 *납입중*
해지율 (월 10%) 로 — 둘의 유일한 차이는 해지율이라 격차가 전부 해지율에서
옵니다. 보장성이라 완납 후 해지율 (2%) 이 납입중 (10%) 보다 *낮습니다*.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 3개월, **완납된 보유계약** 으로 평가
- 월 사망률 1%, 사망보험금 100,000, 보험료 0 (이미 완납), 환급금 없음
- 납입후 해지 월 2% vs 납입중 해지 월 10% (보장성 → 완납 후 낮아짐)
- 월 할인율 0 (상태/해지 동학에 집중)
```

```python
import numpy as np
import fastcashflow as fcf
from fastcashflow import STATE_MODELS, STATE_PAIDUP, STATE_ACTIVE

# 계리적 가정 -- 모든 rate 는 (sex, issue_age, duration) 시그니처
death_fn        = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)  # 사망률 월 1%
lapse_fn        = lambda s, a, d: np.full(d.shape, 1 - (1 - 0.10) ** 12)  # 납입중 해지 월 10%
lapse_paidup_fn = lambda s, a, d: np.full(d.shape, 1 - (1 - 0.02) ** 12)  # 납입후 해지 월 2%
waiver_fn       = lambda s, a, d: np.full(a.shape, 0.0)                   # 납입면제 없음

basis = fcf.Basis(
    mortality_annual        = death_fn,         # 보유계약 감쇠용 사망률 (월 1%)
    lapse_annual            = lapse_fn,          # active 해지율 (납입중 월 10%)
    lapse_paidup_annual     = lapse_paidup_fn,   # paidup 해지율 (납입후 월 2%)
    waiver_incidence_annual = waiver_fn,         # active -> waiver 전이율 (없음)
    discount_annual         = 0.0,               # 연 할인율 0 (검증 단순화)
    ra_confidence           = 0.75,              # 위험조정 신뢰수준 75%
    mortality_cv            = 0.10,              # 사망률 변동계수 10%
    state_model             = STATE_MODELS["WAIVER_PAIDUP"],  # 3-state
    coverages               = (
        fcf.CoverageRate("DEATH", death_fn),     # 사망 보장 1종 (청구 rate = death_fn)
    ),
)

# 같은 계약을 시작 상태만 바꿔 두 번 평가
def measure_in(state):
    mp = fcf.ModelPoints.single(
        issue_age     = 40,            # 가입연령 40세
        sex           = 0,             # 성별 (0=남, 1=여)
        benefits      = {0: 100_000},  # 0번 보장 (= DEATH) 의 보험금 100,000
        level_premium = 0,             # 보험료 0 (이미 완납)
        term_months   = 3,             # 잔여 보험기간 3개월
        state         = state,         # 시작 상태 (자리 지정)
        calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},
    )
    return fcf.gmm.measure(mp, basis)

m_paidup = measure_in(STATE_PAIDUP)  # 완납 (해지 월 2%)
m_active = measure_in(STATE_ACTIVE)  # 납입중 (해지 월 10%) -- 대조용

print(f"paidup inforce  = {m_paidup.cashflows.inforce[0, :3]}")    # 보유계약 (해지 2%)
print(f"paidup claim_cf = {m_paidup.cashflows.claim_cf[0, :3]}")   # 사망보험금
print(f"paidup BEL      = {m_paidup.bel[0]:.2f}")               # 최선추정부채
print(f"active inforce  = {m_active.cashflows.inforce[0, :3]}")    # 보유계약 (해지 10%)
print(f"active BEL      = {m_active.bel[0]:.2f}")               # 대조용 BEL
```

출력:

```
paidup inforce  = [1.         0.9702     0.94128804]
paidup claim_cf = [1000.       970.2      941.28804]
paidup BEL      = 2911.49
active inforce  = [1.       0.891    0.793881]
active BEL      = 2684.88
```

손계산으로 두 상태의 보유계약을 따라갑니다. 두 경우 모두 매월 사망 (1%)
*과* 해지로 빠지지만, 해지율이 다릅니다:

| t | paidup inforce (사망1% × 해지2%) | paidup 사망보험금 | active inforce (사망1% × 해지10%) | active 사망보험금 |
|---|---|---|---|---|
| 0 | 1.000000 | 1,000.00 | 1.000000 | 1,000.00 |
| 1 | 0.970200 |   970.20 | 0.891000 |   891.00 |
| 2 | 0.941288 |   941.29 | 0.793881 |   793.88 |

- `paidup inforce[t] = (0.99 × 0.98)^t` — 사망 0.99 와 납입후 해지 0.98
- `active inforce[t] = (0.99 × 0.90)^t` — 사망 0.99 와 납입중 해지 0.90
- 보험료가 0 이므로 BEL = PV(사망보험금):
  - paidup BEL = 1,000 + 970.2 + 941.29 = **2,911.49**
  - active BEL = 1,000 + 891 + 793.88 = **2,684.88**

완납 후 해지율 (2%) 이 납입중 (10%) 보다 낮아 보유계약이 더 천천히
소멸하고, 그만큼 미래 사망보험금이 더 남아 **BEL 이 커집니다**
(2,911.49 > 2,684.88).

```{admonition} 보장성과 저축성에서 해지의 방향이 반대
:class: warning

이 예제 (보장성) 는 해지 시 지급액 (해약환급금) 이 없습니다. 그래서 해지는
순수하게 부채를 *방출* 하기만 해 해지율이 낮을수록 runoff 가 느려져 BEL 이
커집니다 — 위에서 완납 BEL 이 더 큰 이유입니다. **저축성·고환급 상품은
방향이 반대** 입니다: 완납 시점에 해약환급금이 정점이라 해지가 환급금
유출을 일으켜 BEL 을 *키우고*, 해지율 자체도 완납 시점에 spike 합니다.
이때는 해약환급금 (surrender value) 을 함께 넣어야 BEL 방향이 맞습니다.
```

## 결과 읽기 — 해지율이 갈리는 자리

3-state paid-up 모델의 한 줄 요약: **`lapse_paidup_annual` 이 paidup 점유에만
작동해 납입후 runoff 속도를 바꾼다.**

- 위 두 측정의 유일한 차이는 시작 상태입니다. 같은 사망률 · 같은 보험금
  · 같은 보험료 (0) 인데도 BEL 이 226.61 만큼 다른 것은 **전적으로
  해지율** (2% vs 10%) 때문입니다.
- (보장성에서) 납입후 해지율을 낮출수록 paidup 보유계약이 천천히 빠져
  BEL 이 커집니다 (환급금 없을 때).
- `lapse_paidup_annual` 을 **생략** 하면 paidup 도 `lapse_annual` 을 써서
  active 와 같은 runoff 가 됩니다 — 그러면 paidup 을 별도 상태로 둔 의미가
  사라집니다.

```{admonition} 상태 점유는 trajectory 에 직접 노출되지 않음
:class: note

[3.1](waiver) 과 마찬가지로 `Measurement.cashflows.inforce` 는 세 상태의
합입니다. 위 예제는 보험료가 0 이고 시작 상태가 paidup 뿐이라 inforce 가
곧 paidup 점유와 같지만, 일반적으로 상태별 점유가 필요하면
[검증 패턴](../workflow/validation) 의 `gmm.trace` 로 확인합니다.
```

## 변형 — 해지율 축과 워크북 wiring

### 신계약 — duration-step lapse 로 납입후 하락 표현

본문 예제는 *이미 완납된 보유계약* 을 자리 지정으로 평가했습니다.
**신계약을 가입 시점부터** 측정하면서 완납 시점의 해지율 하락을 반영하려면,
paidup 상태가 아니라 `lapse_annual` 을 가입경과 (policy duration) 의
단계함수로 줍니다 — 납입기간 동안 한 값, 납입후 다른 값. 보험료 중단은
`premium_term_months` 가 처리하므로 active 한 상태로 한 번에 투영합니다.

```python
import numpy as np
import fastcashflow as fcf

# 계리적 가정 -- 해지율을 가입경과 (policy duration, 연 단위) 의 단계함수로
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)  # 사망률 월 1%
lapse_fn = lambda s, a, d: np.where(                              # 완납 시점에 해지율 하락
    d < 1,
    1 - (1 - 0.30) ** 12,   # 납입중 (가입 1년 이내) 월 30%
    1 - (1 - 0.02) ** 12,   # 납입후 (1년 이후)      월 2%
)

basis = fcf.Basis(
    mortality_annual = death_fn,   # 보유계약 감쇠용 사망률 (월 1%)
    lapse_annual     = lapse_fn,   # 해지율 (납입중 30% -> 납입후 2%)
    discount_annual  = 0.0,        # 연 할인율 0
    ra_confidence    = 0.75,       # 위험조정 신뢰수준 75%
    mortality_cv     = 0.10,       # 사망률 변동계수 10%
    coverages        = (
        fcf.CoverageRate("DEATH", death_fn),  # 사망 보장 1종
    ),
)

mp = fcf.ModelPoints.single(
    issue_age           = 40,            # 가입연령 40세
    sex                 = 0,             # 성별 (0=남, 1=여)
    benefits            = {0: 100_000},  # 사망보험금 100,000
    level_premium       = 1_000,         # 월납 보험료 1,000
    term_months         = 24,            # 보험기간 2년
    premium_term_months = 12,            # 납입기간 1년 (이후 완납)
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},
)

m   = fcf.gmm.measure(mp, basis)
ifc = m.cashflows.inforce[0]
pcf = m.cashflows.premium_cf[0]
print(f"premium month 11 / 12        = {pcf[11]:.4f} / {pcf[12]:.4f}")  # 보험료 중단
print(f"inforce ratio m11/m10 (납입중) = {ifc[11] / ifc[10]:.4f}")        # 해지 30%
print(f"inforce ratio m13/m12 (납입후) = {ifc[13] / ifc[12]:.4f}")        # 해지 2%
```

출력:

```
premium month 11 / 12        = 17.7038 / 0.0000
inforce ratio m11/m10 (납입중) = 0.6930
inforce ratio m13/m12 (납입후) = 0.9702
```

납입기간이 끝나는 12개월째에 보험료가 0 으로 멈추고 (`premium_term_months`),
같은 시점에 해지율이 떨어집니다:

- 납입중 (가입 1년 이내): 월 생존비 0.6930 = 사망 0.99 × 해지 0.70 (해지 30%)
- 납입후 (1년 이후):      월 생존비 0.9702 = 사망 0.99 × 해지 0.98 (해지 2%)

상태를 나누지 않고 `lapse_annual` 한 함수로 납입후 하락을 표현했습니다.
신계약 측정에서는 이 길이 가장 단순합니다 — paidup 을 별도 상태로 두는 것은
보유계약 결산처럼 *시작부터 완납 상태인* 계약을 평가할 때입니다.

```{admonition} 단계는 연 경계에서만
:class: note

`lapse_annual` 의 `duration` 인자는 **연 단위 가입경과** 입니다 (엔진이
해지율을 가입연도별 격자로 평가). 따라서 단계함수의 전환점도 연 경계에만
놓입니다 — 납입기간이 12 / 24 / 36개월 처럼 정수 연이면 정확히 맞고, 18개월
처럼 연 중간이면 그 해 평균으로 흡수됩니다. 월 단위 정밀도가 필요하면 별도
확장이 필요합니다.
```

### 완납 후 해지율의 현실적 수준

본문 예제의 "월 2%" 는 손계산 대비를 또렷이 하려는 **과장값** 입니다.
실제 장기 보장성 · 무·저해지 상품의 완납 후 해지율은 그보다 훨씬 낮습니다.
금융위원회가 제4차 보험개혁회의에서 마련한 IFRS 17 주요 계리가정
가이드라인 (2024년 말 결산부터 적용) 은 무·저해지보험 해지율에 다음을
제시합니다:

- **원칙모형 = 로그-선형 모형** — 납입 완료 시점에 가까워질수록 해지율이
  **0% 로 수렴** (실무 수렴점 약 0.1%). *"저해지상품은 납입 중 해지 시
  환급금이 없거나 적은 특성상 실제 해지율이 낮을 것으로 예상"* 한다는
  논리입니다.
- **종국해약률** — 경험통계가 부족하면 **해외 통계 기준 0.8%** 를 준용하며,
  무·저해지는 표준형 상품보다 낮게 적용합니다.

(출처: 금융위원회 보도자료 ["IFRS17 계리적 가정 가이드라인 마련"](https://www.fsc.go.kr/no010101/80080).)

```{admonition} 장기 건강보험에서 완납 후 해지율이 특히 낮은 이유
:class: note

20년납·가입 40세·60세 완납 같은 장기 건강보험에서는, 완납 시점의 가입자가
보험료 부담이 0 인데 연령 상승으로 암·수술·입원 발생률은 오히려 올라갑니다.
보장을 계속 유지할 유인이 강해 해지 유인이 작습니다. 가이드라인이 무·저해지
해지율을 표준형보다 낮게, 완납 시점에 0% 로 수렴시키는 것도 같은 방향입니다.
보장성의 완납 후 해지율은 **방향뿐 아니라 절대 수준도 낮게** 잡는 게
일반적입니다.
```

### 완납 이벤트를 regime 으로 쪼개기

장기 건강보험의 해지 가정을 제대로 만들 때 중요한 것은 단일 "완납 후
해지율" 한 숫자보다, **완납 이벤트 전후를 별도 regime 으로** 보는 것입니다:

```{list-table}
:header-rows: 1
:widths: 28 72

* - regime
  - 성격
* - 완납 직전
  - 마지막 납입을 앞두고 끝까지 유지하려는 구간. 해지율이 이미 낮아짐
* - 완납 시점
  - 이벤트가 발생하는 해. 상품에 따라 spike (저축성) 또는 급락 (보장성)
* - 완납 후 1~3년
  - 이벤트 직후의 과도 구간. 경험이 가장 불확실
* - 완납 후 장기
  - 안정된 종국 수준 (보장성은 해외 통계 기준 0.8% 또는 그 이하로 수렴)
```

엔진에서는 `lapse` (납입중) 와 `lapse_paidup_annual` (완납) 의 두 자리에,
각각 `duration` 의존 표를 넣어 이 regime 을 근사합니다. 다만 완납 시점의
*월* 단위 spike 는 연 단위 표가 그 해 평균으로 흡수하므로, 그 정밀도가
필요하면 별도 확장이 필요합니다.

### 단기납 종신 — 반대 극단

5년납·7년납 같은 단기납 종신은 완납 후 해지율이 *낮아지는* 보장성과
정반대입니다. 짧은 납입 후 높은 환급률에 도달하는 구조라 **완납 시점에
해약이 급증** 할 수 있습니다 (cash-out). 이런 상품은 `lapse_paidup_annual`
에 완납 직후 duration 에서 높은 값을 넣고, **해약환급금 (surrender value)
을 반드시 함께 모델링** 해야 BEL 방향이 맞습니다.

### 납입중 / 납입후를 segment 축으로 쪼개기

상태 모델 대신 **segment 키 (product, channel, 납입상태)** 로 납입중 /
납입후를 나누는 길도 있습니다 — 그러면 보유계약을 두 묶음으로 분리해
각각 다른 해지율표로 평가합니다. 상태 모델은 한 계약 안에서 active <->
waiver 의 *동적 전이* 까지 필요할 때, segment 분리는 납입후가 이미 확정된
정적 묶음일 때 더 단순합니다.

## 함정

### 함정 1 — paidup 으로 전이가 일어나길 기대함

`STATE_MODELS["WAIVER_PAIDUP"]` 에는 paidup 을 목적지로 삼는 전이가
없습니다. active 계약을 `term_months` 만큼 굴려도 *자동으로* paidup 으로
넘어가지 않습니다. paidup 계약은 반드시 `state = STATE_PAIDUP` 으로 시작
상태를 자리 지정해야 합니다.

### 함정 2 — `lapse_paidup_annual` 을 빠뜨림

이 필드를 생략하면 paidup 이 `lapse_annual` 로 fallback 해 active 와 같은
해지율을 씁니다. 동작은 정상이지만 paidup 을 별도 상태로 둔 효과가
사라집니다 (cash flow 가 active 와 동일). 납입중과 다른 납입후 해지율을
반영하려면 이 필드를 명시적으로 채워야 합니다.

### 함정 3 — 완납 계약에 보험료를 그대로 둠

완납 계약은 보험료를 안 냅니다. `level_premium` 을 0 으로 두지 않으면
이미 완납된 계약에서 보험료 수입이 잡혀 BEL 이 틀립니다. (paidup state
자체도 `premium=False` 라 보험료가 0 으로 계산되지만, 입력에서도 0 으로
두는 것이 의도를 드러냅니다.)

## 인접 레시피

- [3.1 보험료 납입면제 (waiver)](waiver) — 2-state 입문. 본 챕터의 직접
  출발점.
- [1.3 사망률의 두 가지 역할](../basics/mortality-roles) — decrement (사망 +
  해지) 와 보장 청구의 분리.
- [2.1 정기보험](../simple/term-life) — 상태 전이 없는 정액형.
- [검증 패턴](../workflow/validation) — `gmm.trace` 로 상태별 점유와
  cash flow 를 한 줄씩 확인.

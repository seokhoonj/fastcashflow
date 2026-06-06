# 1.3 사망률의 두 가지 역할

```{admonition} 이 챕터에서 배우는 것
:class: chapter-brief

- 엔진 전반을 관통하는 `mortality_annual` (보유계약 감쇠) 의 역할
- 사망 보장 청구가 **별도의 슬롯** 으로 계산되는 이유
- 단순 정기보험 (일반사망) 에서 두 율이 같고, 질병 / 재해 같은
  cause-specific 사망 보장에서 명시적으로 달라지는 이유
- 손계산 / cookbook 예제가 두 슬롯에 같은 callable 을 공유하는 안전한
  입력 연결 패턴
```

이 챕터를 읽고 나면 다음 챕터 (1.4 보장 청구 메커니즘) 의 `inforce ×
rate × benefit` 식에서 각 슬롯의 rate 가 무엇을 의미하는지 명확해집니다.

## 1.3.1 사망률 = 엔진의 보편 감쇠

엔진의 모든 cash flow 는 한 가지 양에 비례합니다 — 그 시점의 **보유계약
수 `inforce`**. 그리고 `inforce` 를 매월 줄이는 단 하나의 사망 관련
입력이 `mortality_annual` 입니다.

```
inforce[t+1] = inforce[t] × (1 - mortality_annual_monthly) × (1 - lapse_monthly)
```

contract 당 한 값. 어느 원인으로든 사람이 in-force 에서 빠지는 **전체**
율. 사람이 죽으면 보험료 납입도 멈추고 사업비도 발생을 멈추고 만기금도
사라지고 청구도 더 이상 못 받습니다. 그래서 `mortality_annual` 은 엔진
어디서도 비켜갈 수 없는 보편적 감쇠 — 보험료 cash flow, 사업비 cash
flow, 만기금 cash flow, 그리고 모든 보장의 청구 cash flow 가 같은
`inforce` 에 묶여 함께 줄어듭니다.

```text
basis = fcf.Basis(
    mortality_annual = lambda s, a, d: ...,  # all-cause 감쇠
    ...
)
```

워크북에서는 `segments` 시트의 `mortality_table` 컬럼이 가리키는
`mortality_tables` 시트가 이 입력의 출처입니다. 회사별 / 성별 / 연령별
all-cause 사망률 테이블 한 장.

이 한 슬롯이 가장 큰 가정 — `mortality_annual` 을 +10% shock 하면
거의 모든 cash flow 가 함께 흔들립니다.

## 1.3.2 사망 보장의 청구는 별도 슬롯

위 1.3.1 까지가 엔진의 보편 감쇠 이야기입니다. 그런데 **사망 보장의
청구 금액** 을 계산하는 자리는 따로 있습니다.

생각해보면 그래야 합니다. 사람이 죽는 사건은 `mortality_annual` 로 이미
`inforce` 에서 빠지는 효과를 냈습니다. 그 사람에게 *얼마의 사망보험금이
지급되느냐* 는 별개의 질문 — 보장의 약관, 면책 / 감액, 보험금액 등의
함수. 그래서 엔진은 사망 보장의 청구를 그 보장의 약관에 묶인 별도 rate
로 계산합니다.

```
claim_DEATH[t] = inforce[t] × DEATH.rate × DEATH.benefit
```

이 `DEATH.rate` 는 1.3.1 의 `mortality_annual` 과 같은 mortality 의
일종이지만, **다른 입력 슬롯**:

```text
basis = fcf.Basis(
    mortality_annual = ...,                                   # 1.3.1 의 감쇠
    coverages        = (fcf.CoverageRate("DEATH", rate_fn),)  # 사망 보장 청구율
)
```

워크북에서도 다른 schema path — `coverages` 시트의 `rate_table` 컬럼이
가리키는 `mortality_tables` 또는 `incidence_rate_tables` 시트. 같은
워크북에서도 **별개의 슬롯** 입니다.

이 별개의 슬롯이 필요한 이유는 다음 두 절에서 사례로 확인합니다.

## 1.3.3 같은 숫자일 때 — 일반사망 보장 하나

가장 단순한 정기보험을 봅시다. 사망 보장이 일반사망 (all-cause) 하나만
붙은 경우. 어느 원인으로든 사람이 죽으면 두 일이 동시에 일어납니다 —
보유계약에서 한 명 빠지고 (`mortality_annual` 의 일), 그 사람에게
사망보험금이 지급됨 (DEATH 보장 청구). 같은 사건의 같은 빈도라 두 율이
자연히 같은 숫자가 됩니다.

cookbook 의 손계산 예제 (1.4 / 2.1) 가 바로 이 경우라, 두 슬롯에 같은
callable 을 넘깁니다:

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda s, a, d: np.full(d.shape, 0.0)

basis = fcf.Basis(
    mortality_annual = death_fn,                              # 보유계약 감쇠 (all-cause)
    lapse_annual     = lapse_fn,                              # 해지율 (없음)
    discount_annual  = 0.03,                                  # 할인율
    ra_confidence    = 0.75,                                  # 위험조정 신뢰수준
    mortality_cv     = 0.10,                                  # 사망률 변동계수
    coverages        = (fcf.CoverageRate("DEATH", death_fn),) # 일반사망 청구율
)
```

한 변수 `death_fn` 을 두 슬롯에 공유. "두 율이 같다" 는 사실을 코드에
명시하는 동시에, 한 쪽만 바꾸는 silent 어긋남이 구조적으로 차단됩니다.

## 1.3.4 다른 숫자일 때

두 율이 명시적으로 달라지는 경우는 흔합니다. 핵심은 *all-cause vs
cause-specific* 의 구분.

**경우 1 — cause-specific 사망 보장 하나만 있을 때**

사망 보장이 질병사망 / 재해사망 같은 cause-specific 이면, 보장이
하나뿐이어도 두 율이 다릅니다.

```
mortality_annual = rate_all_cause                # 보유계약 감쇠 (all-cause)

coverages = (
    fcf.CoverageRate("ADB", rate_accident),      # 재해사망만 (예: 연 0.3%)
)
```

보유계약은 **어느 원인으로든** 사람이 죽으면 빠지니까 감쇠율은 all-cause
(예: 연 1.2%). 하지만 사망보험금 청구는 **재해 사고로 죽은 경우만** —
청구율 (0.3%) 이 all-cause 보다 훨씬 낮음. 두 율이 다른 숫자.

**경우 2 — 다종 사망 보장이 함께 붙을 때**

한 계약에 일반사망 + 질병사망 + 재해사망 세 보장이 함께 붙으면:

```
mortality_annual = rate_all_cause                         # all-cause (예: 연 1.2%)

coverages = (
    fcf.CoverageRate("DEATH",         rate_all_cause),    # 일반사망 (1.2%)
    fcf.CoverageRate("DISEASE_DEATH", rate_disease),      # 질병사망 (0.9%)
    fcf.CoverageRate("ADB",           rate_accident),     # 재해사망 (0.3%)
)
```

사람은 한 번만 죽으므로 `inforce` 는 all-cause 율로 한 번 감쇠. 그 한
번의 사망 사건이 여러 보장 의 청구를 동시 트리거 (재해로 죽으면 일반
사망금 + 재해사망금 둘 다 지급). 두 cause-specific 의 합 (0.9 + 0.3 =
1.2) 이 all-cause 와 일치 — 같은 사망 사건의 원인별 분해라서.

`mortality_annual` 을 분리하지 않고 보장 청구 식 안에 묶었다면 세 사망
보장 중 어느 하나의 rate 로만 in-force 를 감쇠해야 했을 겁니다. 분리되어
있으니 감쇠는 contract 전체에 한 번, 청구는 보장마다 자기 율.

**경우 3 — calibration 출처가 다를 때**

같은 일반사망 보장이어도 두 율의 source 가 다른 경우. 보유계약 감쇠는
경험률표 (실측 자료) 로 calibrate, 사망보험금 지급률은 *공시 표준
mortality table* (규제 의무) 로. 두 표가 다른 숫자라 두 슬롯도 다른
숫자.

## 1.3.5 손계산 입력 연결의 안전 패턴

위 사례들이 fastcashflow 가 분리한 이유. 단순 정기보험 손계산은 두 값이
같지만, 두 슬롯에 별개의 값을 실수로 넘기면 결과가 silent 로 어긋납니다.

| 잘못된 연결 | 결과 |
|---|---|
| `mortality_annual` = 0%, `DEATH` rate = 1% | 보유계약 안 줄고 매월 사망보험금 청구. 무한 사망. BEL 과대평가. |
| `mortality_annual` = 1%, `DEATH` rate = 0% | 보유계약은 줄고 사망보험금 없음. BEL 과소평가. |

두 패턴 모두 에러 없이 통과 — 조용히 어긋나는 함정. 한 변수 공유 패턴 (`death_fn`
을 두 슬롯에) 이 구조적 방어.

```{admonition} 워크북 로더는 자동 처리
:class: note

`fcf.read_basis("basis.xlsx")` 로 워크북에서 가정을 읽으면,
`segments` 시트의 `mortality_table` 컬럼이 두 슬롯 (`mortality_annual` /
`coverages` 의 DEATH `rate_table`) 를 같은 `table_id` 로 자동 매핑.
손계산 / 단위테스트로 직접 `Basis(...)` 를 채울 때만 주의.
```

## 1.3.6 다음 챕터로

이 분리를 이해하고 [1.4 보장 청구 메커니즘](coverage-mechanics) 으로
가면, `inforce × rate × benefit` 식의 `inforce` 는 `mortality_annual`
이 감쇠시키는 공유 풀, 각 보장의 `rate` 는 그 보장의 청구 빈도임이
자연스럽게 읽힙니다. DIAGNOSIS 가 별도 `undiagnosed` 풀을 갖는 이유도
같은 결.

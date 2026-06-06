# 1.3 서로 다른 두 입력 — 보유계약 사망률과 사망보험금 발생률

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 엔진에는 **이름은 비슷하지만 하는 일이 완전히 다른 두 입력**이 있다 —
  보유계약(in-force)을 줄이는 **보유계약 사망률**과, 사망보험금 지급액만
  계산하는 **사망보험금 발생률**
- "한 사망률의 두 역할"이 아니라 **별개의 입력 슬롯**이라는 것
- 모든 입력을 *어떤 급부를 만들고 / 어떤 풀을 줄이는가* 한 축으로 보면
  사망·진단·반복급부가 한 표에 정리된다는 것
- 단순 정기보험에서 둘이 우연히 같은 숫자가 되는 경우 vs 명시적으로
  달라지는 경우
- 이 분리가 1.4의 "한 번 vs 반복" 청구 메커니즘으로 이어지는 길
```

엔진 안의 일을 "사망률" 한 단어로 뭉뚱그리면 반드시 혼란이 생깁니다.
사실 엔진에는 하는 일이 다른 여러 입력이 있고, 핵심은 *각 입력이 어떤
급부를 만들고, 어떤 풀(pool)을 줄이는가* 입니다. 이 한 축으로 보면 전부
한 표에 정리됩니다.

| 코드 입력 | 용어 | 급부 | **줄이는 풀** |
|---|---|---|---|
| `mortality_annual` | **보유계약 사망률** | 직접 지급 없음 | 계약 전체 in-force |
| `lapse_annual` | **해지율** | 없음 | 계약 전체 in-force |
| DEATH 담보의 `rate` | **사망보험금 발생률** | 사망보험금 | 없음 |
| DIAGNOSIS 담보의 `rate` | **진단 발생률** | 진단보험금 | 그 담보의 미진단 풀 |
| MORBIDITY 담보의 `rate` | **반복급부 발생률** | 반복 보험금 | 없음 |

이 챕터는 이 중 가장 헷갈리는 한 쌍 — **보유계약 사망률**과 **사망보험금
발생률** — 을 갈라 둡니다. 둘 다 "사망"에 묶여 있어 같은 양처럼 보이지만,
하나는 *누가 보유계약에 남아 있는가* 를, 다른 하나는 *얼마의 사망보험금을
지급하는가* 를 정하는 **별개의 입력**입니다.

```{admonition} 표준 용어로는 — 탈퇴(decrement) vs 발생(incidence)
:class: note

위 표의 최상위 축 *어떤 풀을 줄이는가* 는 다중탈퇴모형(multiple-decrement
model)의 교과서 구분 그대로입니다.

- **풀을 줄이는 입력 = 탈퇴(decrement)**. `mortality_annual`은 표준 용어로
  **사망탈퇴율**(mortality decrement, q_x^(d)) — 연단위 *확률*이라 **율**이지
  **력**(force, μ)이 아니고, 엔진은 `inforce x (1 - q)`로 깎습니다.
  `lapse_annual`은 같은 줄의 해지탈퇴율.
- **풀을 안 줄이고 급부만 트리거하는 입력 = 발생(incidence)**. 담보의
  `rate`가 그것 — 사망 보장이면 mortality incidence, 진단/입원이면
  morbidity incidence.

엔진 코드도 같은 어휘입니다 — `mortality_annual`의 docstring은 "in-force
decrement", 전이율은 `waiver_incidence_annual` / `ci_incidence_annual`처럼
`_incidence` 접미사. 그래서 이 챕터가 쓰는 친근형 **보유계약 사망률**과
**사망보험금 발생률**은 각각 **사망탈퇴율(decrement)** 과 **사망 발생률
(incidence)** 의 다른 이름일 뿐입니다.
```

## 1.3.1 보유계약 사망률 — 계약 전체 in-force를 줄임

엔진의 모든 cash flow는 한 가지 양에 비례합니다 — 그 시점의 **보유계약 수
`inforce`**. 그리고 `inforce`를 매월 줄이는 사망 입력이 **보유계약 사망률**
(`mortality_annual` — 표준 용어로 **사망탈퇴율**, mortality decrement q_x^(d))입니다.

```
inforce[t+1] = inforce[t] × (1 - 보유계약사망률_월) × (1 - 해지율_월)
```

contract당 한 값. 어느 원인으로든 사람이 in-force에서 빠지는 **전체
(all-cause)** 율입니다. 사람이 죽으면 보험료 납입도 멈추고, 사업비도 멈추고,
만기금도 사라지고, 청구도 더 못 받습니다. 그래서 보유계약 사망률은 엔진
어디서도 비켜갈 수 없는 보편적 감쇠 — 보험료·사업비·만기금, 그리고 모든
보장의 청구가 같은 `inforce`에 묶여 함께 줄어듭니다.

```text
basis = fcf.Basis(
    mortality_annual = lambda s, a, d: ...,  # 보유계약 사망률 (all-cause 감쇠)
    ...
)
```

워크북에서는 `segments` 시트의 `mortality_table` 컬럼이 가리키는
`mortality_tables` 시트가 이 입력의 출처입니다. 회사별·성별·연령별
all-cause 사망률 테이블 한 장.

이 한 슬롯이 가장 큰 가정입니다 — 보유계약 사망률을 +10% shock하면 거의
모든 cash flow가 함께 흔들립니다.

## 1.3.2 사망보험금 발생률 — 사망보험금만 계산하는 별도 입력

보유계약 사망률이 *누가 남아 있는가* 를 정했다면, **사망보험금의 지급액**
을 계산하는 자리는 완전히 따로 있습니다 — **사망보험금 발생률**.

생각해보면 그래야 합니다. 사람이 죽는 사건은 보유계약 사망률로 이미
`inforce`에서 빠지는 효과를 냈습니다. 그 사람에게 *얼마의 사망보험금이
지급되느냐* 는 별개의 질문 — 보장 약관, 면책·감액, 보험금액의 함수입니다.
그래서 엔진은 사망 보장의 청구를 그 보장에 묶인 **별도 rate(사망보험금
발생률)** 로 계산합니다.

```
사망보험금[t] = inforce[t] × 사망보험금발생률 × DEATH.benefit
```

이 발생률은 보유계약 사망률과 **이름만 둘 다 "사망"일 뿐, 서로 다른 입력
슬롯**입니다:

```text
basis = fcf.Basis(
    mortality_annual = ...,                                   # 보유계약 사망률 (in-force 감쇠)
    coverages        = (fcf.CoverageRate("DEATH", rate_fn),)  # 사망보험금 발생률 (청구)
)
```

워크북에서도 schema path가 다릅니다 — `coverages` 시트의 `rate_table`
컬럼이 가리키는 `mortality_tables` 또는 `incidence_rate_tables` 시트.
같은 워크북 안에서도 **별개의 슬롯**입니다.

**중요한 비대칭**: 사망보험금 발생률은 *어떤 풀도 줄이지 않습니다*. in-force를
줄이는 건 어디까지나 보유계약 사망률이고, 발생률은 그 위에서 *지급액만*
계산합니다. 이 "발생률은 풀을 안 줄인다"가 1.4의 반복급부(MORBIDITY)와
직접 이어집니다.

## 1.3.3 둘이 같은 숫자일 때 — 일반사망 보장 하나

가장 단순한 정기보험을 봅시다. 사망 보장이 일반사망(all-cause) 하나만 붙은
경우. 어느 원인으로든 사람이 죽으면 두 일이 **동시에** 일어납니다 — 보유계약
에서 한 명 빠지고(보유계약 사망률의 일), 그 사람에게 사망보험금이 지급됨
(사망보험금 발생률의 일). 같은 사건의 같은 빈도라 두 율이 자연히 같은 숫자가
됩니다.

**같은 숫자이지만 여전히 두 개의 입력입니다.** cookbook의 손계산 예제
(1.4 / 2.1)가 이 경우라, 두 슬롯에 같은 callable을 넘깁니다:

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda s, a, d: np.full(d.shape, 0.0)

basis = fcf.Basis(
    mortality_annual = death_fn,                              # 보유계약 사망률 (all-cause)
    lapse_annual     = lapse_fn,                              # 해지율 (없음)
    discount_annual  = 0.03,                                  # 할인율
    ra_confidence    = 0.75,                                  # 위험조정 신뢰수준
    mortality_cv     = 0.10,                                  # 사망률 변동계수
    coverages        = (fcf.CoverageRate("DEATH", death_fn),) # 사망보험금 발생률 (일반사망)
)
```

한 변수 `death_fn`을 두 슬롯에 공유합니다. "두 율이 같다" 는 사실을 코드에
명시하는 동시에, 한 쪽만 바꾸다 생기는 silent 어긋남이 구조적으로
차단됩니다.

## 1.3.4 둘이 다른 숫자일 때

두 율이 명시적으로 달라지는 경우는 흔합니다. 핵심은 *all-cause vs
cause-specific* 의 구분.

**경우 1 — cause-specific 사망 보장 하나만 있을 때**

사망 보장이 질병사망 / 재해사망 같은 cause-specific이면, 보장이 하나뿐이어도
두 율이 다릅니다.

```text
mortality_annual = rate_all_cause                # 보유계약 사망률 (all-cause)

coverages = (
    fcf.CoverageRate("ADB", rate_accident),      # 사망보험금 발생률: 재해사망만 (예: 연 0.3%)
)
```

보유계약은 **어느 원인으로든** 사람이 죽으면 빠지니까 보유계약 사망률은
all-cause(예: 연 1.2%). 하지만 사망보험금 발생률은 **재해 사고로 죽은
경우만** — 발생률(0.3%)이 보유계약 사망률보다 훨씬 낮습니다. 두 율이 다른
숫자.

**경우 2 — 다종 사망 보장이 함께 붙을 때**

한 계약에 일반사망 + 질병사망 + 재해사망 세 보장이 함께 붙으면:

```text
mortality_annual = rate_all_cause                         # 보유계약 사망률: all-cause (예: 연 1.2%)

coverages = (
    fcf.CoverageRate("DEATH",         rate_all_cause),    # 사망보험금 발생률: 일반사망 (1.2%)
    fcf.CoverageRate("DISEASE_DEATH", rate_disease),      # 사망보험금 발생률: 질병사망 (0.9%)
    fcf.CoverageRate("ADB",           rate_accident),     # 사망보험금 발생률: 재해사망 (0.3%)
)
```

사람은 한 번만 죽으므로 `inforce`는 보유계약 사망률(all-cause)로 한 번
감쇠. 그 한 번의 사망 사건이 여러 보장의 청구를 동시 트리거(재해로 죽으면
일반사망금 + 재해사망금 둘 다 지급). 두 cause-specific의 합(0.9 + 0.3 =
1.2)이 all-cause와 일치 — 같은 사망 사건의 원인별 분해라서.

보유계약 사망률을 보장 청구 식 안에 묶었다면 세 사망 보장 중 어느 하나의
율로만 in-force를 감쇠해야 했을 겁니다. 분리되어 있으니 감쇠는 contract
전체에 한 번, 청구는 보장마다 자기 발생률.

**경우 3 — calibration 출처가 다를 때**

같은 일반사망 보장이어도 두 율의 source가 다른 경우. 보유계약 사망률은
경험률표(실측 자료)로 calibrate, 사망보험금 발생률은 *공시 표준 mortality
table*(규제 의무)로. 두 표가 다른 숫자라 두 슬롯도 다른 숫자.

## 1.3.5 손계산 입력 연결의 안전 패턴

위 사례들이 fastcashflow가 두 입력을 분리한 이유입니다. 단순 정기보험
손계산은 두 값이 같지만, 두 슬롯에 별개의 값을 실수로 넘기면 결과가 silent로
어긋납니다.

| 잘못된 연결 | 결과 |
|---|---|
| 보유계약 사망률 = 0%, 사망보험금 발생률 = 1% | in-force가 안 줄고 매월 사망보험금 청구 — **MORBIDITY와 똑같은 메커니즘**. 무한 청구. BEL 과대평가. |
| 보유계약 사망률 = 1%, 사망보험금 발생률 = 0% | in-force는 줄고 사망보험금 없음. BEL 과소평가. |

두 패턴 모두 에러 없이 통과 — 조용히 어긋나는 함정. 한 변수 공유 패턴
(`death_fn`을 두 슬롯에)이 구조적 방어입니다. (위 첫 행이 곧 1.3.6의
"풀이 안 줄면 반복"임을 보여줍니다.)

```{admonition} 워크북 로더는 자동 처리
:class: note

`fcf.read_basis("basis.xlsx")`로 워크북에서 가정을 읽으면, `segments`
시트의 `mortality_table` 컬럼이 두 슬롯(`mortality_annual` / `coverages`의
DEATH `rate_table`)을 같은 `table_id`로 자동 매핑. 손계산 / 단위테스트로
직접 `Basis(...)`를 채울 때만 주의.
```

## 1.3.6 풀로 보면 — DEATH·DIAGNOSIS·MORBIDITY가 한눈에

이 "어떤 풀을 줄이는가" 관점이 다음 챕터(1.4)의 핵심을 미리 정리해 줍니다.
세 담보 모두 청구식은 `inforce × 발생률 × benefit`로 **똑같고**, "한 번이냐
반복이냐"는 *그 발생률로 어떤 풀이 줄어드느냐* 만의 문제입니다:

- **DEATH** — "한 번". 별도의 **보유계약 사망률**이 공유 in-force를 같은
  율로 깎아, 죽은 사람이 다시 청구하지 못함. (사망보험금 발생률 자체는 풀을
  안 줄임 — in-force를 줄이는 건 보유계약 사망률.)
- **DIAGNOSIS** — "한 번". **진단 발생률**이 *그 담보의 미진단 풀* 을
  `(1 - 진단발생률)`로 직접 깎음. 한 번 진단받으면 그 풀에서 빠짐. (계약은
  안 끝남 — 사람은 in-force에 남아 보험료 내고 다른 담보도 작동.)
- **MORBIDITY** — "반복". **반복급부 발생률**은 *어떤 풀도 줄이지 않음*.
  공유 in-force만 보유계약 사망률·해지율로 줄 뿐이라, 살아있는 한 매월 새로
  청구.

즉 셋의 차이는 청구 식이 아니라 *발생률이 어떤 풀을 (또는 어떤 풀도 안)
줄이는가* 입니다. [1.4 보장 청구 메커니즘](coverage-mechanics)이 이 셋을
toy 예제로 한 페이지에 보여줍니다.

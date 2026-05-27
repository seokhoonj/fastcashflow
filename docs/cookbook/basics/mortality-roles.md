# 1.3 사망률의 두 역할

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 엔진이 사망률을 *두 자리* 에 받는 이유 — `mortality_annual` (보유계약
  감쇠) 과 `coverages` 의 사망보장 rate (보장 청구율)
- 두 양이 *서로 다른 역할* 인 이유: 하나는 *모든 cash flow 의 기반*,
  하나는 *그 보장만의 청구 빈도*
- *단순 정기보험* 에서 두 값이 *우연히 같은 숫자* 인 사정
- 다종 사망보장이 붙은 contract 에서 두 값이 *명시적으로 달라지는* 이유
- 손계산 / cookbook 예제가 *같은 callable 을 두 자리에 공유* 하는
  안전한 wiring 패턴

이 챕터를 읽고 나면 다음 챕터 (1.4 보장 청구 메커니즘) 의 `(A) in_force
× rate × benefit` 식에서 *어느 자리의 rate* 가 무엇을 의미하는지 명확
해집니다.
```

## 두 가지 다른 역할

엔진의 입력 두 자리 — `mortality_annual` 과 `coverages` 의 사망 보장
rate — 은 **이름이 둘 다 "사망률" 같지만 하는 일이 완전히 다릅니다.**

```{list-table}
:header-rows: 1
:widths: 35 65

* - 입력 자리
  - 역할
* - `mortality_annual`
  - **보유계약의 감쇠** — 사람이 *얼마나 죽나*. `in_force` 풀의 동적
    변화.
* - `coverages` 의 DEATH rate
  - **그 사망 보장의 청구 빈도** — 그 보장이 *얼마나 자주 지급되나*.
    `claim_cf` 한 자리에만 영향.
```

이 *역할 차이* 가 핵심이고, 분리해 받는 design 의도의 이유입니다.

## `in_force` 는 *모든* cash flow 의 기반

엔진은 매월 `in_force[t]` — 현재 살아 있는 보유계약 수 — 를 가지고
있습니다. 그리고 그 `in_force` 가 **거의 모든 cash flow 식에 곱해
들어갑니다**:

```
premium[t]   = in_force[t] × level_premium        ← 보험료 유입
expense[t]   = in_force[t] × per_policy_expense   ← 유지비
annuity[t]   = in_force[t] × annuity_payment      ← 생존연금
claim[t]     = in_force[t] × coverage_rate × benefit  ← 사망보험금
maturity[T]  = in_force[T] × maturity_benefit     ← 만기금 (term 시점)
```

→ `in_force` 가 *공유 풀* 입니다. 한 번 줄어들면 *모든 cash flow 식* 의
값이 함께 줄어듦. 보험료도 줄고, 사업비도 줄고, 사망보험금도 줄고, 만기금도 줄음.

이 *공유 풀* 의 *동적 감쇠* 를 책임지는 게 **`mortality_annual`**:

```
in_force[t+1] = in_force[t] × (1 - mortality_annual_monthly[t]) × (1 - lapse_monthly[t])
```

매월 *얼마의 비율로 보유계약이 줄어드나* 의 단일 결정자. 어느 *원인* 으로든
사람이 in-force 에서 빠지는 *전체 율*. **fastcashflow 의 contract 당
mortality_annual 은 한 값** (다종 사망보장이 붙어도 마찬가지).

## 각 보장의 rate 는 *그 보장만의* 청구 빈도

`coverages` 안의 각 보장은 자기 `rate` 를 가집니다. 이 rate 는 **그 보장
하나의 청구 식에만 등장**:

```
claim_for_coverage_X[t] = in_force[t] × coverage_X.rate[t] × coverage_X.benefit
```

- 사망 보장 (DEATH) 이라면: 사망보험금 지급 빈도
- 진단 보장 (DIAGNOSIS) 이라면: 진단보험금 지급 빈도
- 입원 보장 (MORBIDITY) 이라면: 입원 1 회당 지급 빈도

각 보장이 *자기 청구식만* 책임집니다. `in_force` 의 감쇠에는 *영향 없음*.
보험료 / 사업비 / 만기금에도 영향 없음.

## 단순 정기보험 — 두 값이 *우연히* 같다

가장 단순한 사례: **사망 보장 하나만** 붙은 정기보험.

- 사람이 죽는 사건이 *유일한 in-force 감쇠 사건* (해지는 별도)
- 그 사망 사건이 *유일한 사망보험금 청구 사건*

같은 사건이 *한 자리 (in-force 감쇠)* 와 *다른 자리 (사망보험금 청구)* 둘
다를 트리거. 그래서 *수학적으로* `mortality_annual` 과 DEATH 보장의 rate
가 **같은 숫자** 가 됩니다 — 우연이 아니라 *사건이 같으니까*.

cookbook 의 손계산 예제 (1.4 / 2.1) 는 이 case 라서 두 자리에 *같은
callable* 을 넘깁니다:

```python
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

asmp = fcf.Assumptions(
    mortality_annual = death_fn,                              # in_force 감쇠
    coverages        = (fcf.CoverageRate("DEATH", death_fn),) # 사망보장 청구율
)
```

한 변수 `death_fn` 을 *두 자리에 공유*. 둘이 *같다는 사실* 을 코드에 명시.

## 다종 사망보장 — 두 값이 *명시적으로* 다르다

한 계약에 사망 보장이 *세 개* 붙어 있다면:

```text
asmp = fcf.Assumptions(
    mortality_annual = rate_all_cause,        <- all-cause (예: 연 1.2%)
    coverages = (
        fcf.CoverageRate("DEATH",         rate_all_cause),    <- 일반사망 (1.2%)
        fcf.CoverageRate("ADB",           rate_accident),     <- 재해사망 (0.3%)
        fcf.CoverageRate("DISEASE_DEATH", rate_disease),      <- 질병사망 (0.9%)
    )
)
```

엔진 안에서 일어나는 일:

1. **in_force 감쇠**: `mortality_annual` (= all-cause 1.2%) 하나가 풀 전체를
   감쇠. 사람은 *한 번* 만 죽음 (in-force 는 그 한 번 분만 줄어듦).
2. **사망보험금 청구**: 세 보장이 *각자 자기 rate* 로 청구식 계산.
   - 일반사망 청구 = in_force × 1.2% × benefit_일반
   - 재해사망 청구 = in_force × 0.3% × benefit_재해
   - 질병사망 청구 = in_force × 0.9% × benefit_질병
   세 청구가 *동시에 발생*. 같은 사망 사건이 *여러 보장* 의 청구를 동시
   트리거 (재해로 죽으면 일반 + 재해 둘 다 지급).

여기서 *내적 일관성*: 청구 rate 들의 합 (0.3 + 0.9 = 1.2) 이 all-cause
(1.2) 와 일치. 같은 사망 사건의 *원인별 분해* 라서.

만약 `mortality_annual` 이 *분리되지 않고* 보장 청구 식 안에 묶여 있었다면
— 세 사망 보장 중 *어느 하나의 rate* 로만 in-force 를 감쇠해야 했을 겁니다.
어떤 보장을 골라야 할지 ambiguous. 분리되어 있으니 **감쇠는 contract
전체에 한 번, 청구는 보장마다 자기 율** — 자연스러움.

## 두 값이 다른 또 하나의 사정 — 출처 분리

또 다른 사례: 회사가 두 값을 *다른 source* 에서 calibrate.

- 보유계약 감쇠는 *경험률표* (실측 자료, 회사 내부)
- 사망보험금 청구율은 *공시 표준 mortality table* (KIDI 표, 규제 의무)

두 율이 *다른 표* 에서 나오므로 다른 숫자. 한 자리에 묶어두었다면 이런
calibration 자유도가 없음.

## 손계산 wiring — silent footgun 방어

위 사례들이 fastcashflow 가 *분리한* 이유. 단순 정기보험 손계산은 두 값이
같지만, 두 자리에 *별개의 값* 을 실수로 넘기면 *silent* 어긋남:

| 잘못된 wiring | 결과 |
|---|---|
| `mortality_annual = 0`, DEATH rate = 1% | 보유계약 안 줄어드는데 매월 사망보험금 청구. 같은 사람이 무한 사망 → BEL 과대평가. |
| `mortality_annual = 1%`, DEATH rate = 0 | 보유계약은 줄어드는데 사망보험금 한 푼도 안 나감 → BEL 과소평가. |

두 패턴 모두 *에러 없이 통과* — 그래서 silent. 코드로 직접
`Assumptions(...)` 를 호출할 때 한 변수 공유 패턴 (`death_fn` 을 두 자리
에) 이 *구조적 방어*.

```{admonition} 워크북 로더는 자동 처리
:class: note

`fcf.read_assumptions("assumptions.xlsx")` 로 워크북에서 가정을 읽으면,
`segments` 시트의 `mortality_table` 컬럼이 두 자리 (`mortality_annual` /
`coverages` 의 DEATH `rate_table`) 를 같은 `table_id` 로 자동 매핑.
손계산 / 단위테스트로 직접 Assumptions 를 채울 때만 주의.
```

## 다음 챕터로

이 분리를 이해하고 [1.4 보장 청구 메커니즘](coverage-mechanics) 으로
가면:

- `(A) in_force × rate × benefit` 식의 **in_force** 는 `mortality_annual`
  이 감쇠시키는 공유 풀
- 같은 식의 **rate** 는 *그 보장의* coverage rate (DEATH / MORBIDITY)
- DIAGNOSIS 가 별도 `(B)` 식인 이유는 — *그 보장만의 별도 풀
  (`undiagnosed`)* 이 필요해서

`mortality_annual` 의 감쇠는 모든 보장이 *공유* 하지만, 각 보장의 청구
식은 *자기 rate × 자기 풀 (in_force 또는 undiagnosed)* 로 독립. 이게
엔진의 핵심 구조.

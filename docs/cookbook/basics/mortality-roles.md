# 1.3 사망률의 두 역할

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 엔진이 사망률을 서로 다른 두 입력 슬롯에서 받는 이유 —
  `mortality_annual` (보유계약 감쇠) 과 `coverages` 의 사망보장 rate
  (보장 청구율)
- 단순 정기보험에서 두 값이 같은 숫자가 되는 사정과, 다종 사망보장에서
  명시적으로 달라지는 이유
- 손계산 / cookbook 예제가 두 슬롯에 같은 callable 을 공유하는 안전한
  wiring 패턴
```

이 챕터를 읽고 나면 다음 챕터 (1.4 보장 청구 메커니즘) 의 `in_force ×
rate × benefit` 식에서 각 슬롯의 rate 가 무엇을 의미하는지 명확해집니다.

## 1.3.1 두 가지 역할

같은 "사망률" 이라는 단어지만, 엔진 안에서 하는 일은 완전히 다릅니다.

`mortality_annual` 은 **보유계약의 감쇠율**. 매월 보유계약이 얼마의 비율로
줄어드는지를 결정합니다. 어느 원인으로든 사람이 in-force 에서 빠지는
전체 율 — contract 당 한 값.

```
in_force[t+1] = in_force[t] × (1 - mortality_annual_monthly) × (1 - lapse_monthly)
```

`coverages` 안의 사망보장 rate 는 **그 보장의 청구 빈도**. 사망보험금이
얼마나 자주 지급되는지의 율 — 보장마다 개별.

```
claim_DEATH[t] = in_force[t] × DEATH.rate × DEATH.benefit
```

워크북에서도 두 입력은 서로 다른 schema path 로 들어옵니다 —
`segments` 시트의 `mortality_table` 컬럼이 가리키는 `mortality_tables`
시트 (decrement 용) 와, `coverages` 시트의 `rate_table` 컬럼이 가리키는
`mortality_tables` 또는 `incidence_rate_tables` 시트 (보장 청구 용). 두
입력은 같은 워크북에서도 *별개의 슬롯*.

두 식의 차이가 핵심입니다. `in_force` 가 줄어들면 *모든* cash flow (보험료,
사업비, 만기금, 청구 모두) 가 함께 줄어들고, 그 감쇠는 `mortality_annual`
한 슬롯이 책임집니다. 반면 각 보장의 rate 는 *그 보장의 청구* 식 한
자리에만 나타나고, 보유계약 감쇠에는 영향을 주지 않습니다.

## 1.3.2 같은 숫자일 때 — 단순 정기보험

사망보장 하나만 붙은 정기보험을 생각해봅시다. 사람이 죽는 사건이 두
일을 동시에 일으킵니다 — 보유계약이 한 명 빠지고, 그 사람에게 사망보험금
이 지급됨. 같은 사건이 트리거하니까 두 율이 자연히 같은 숫자가 됩니다.

cookbook 의 손계산 예제 (1.4 / 2.1) 가 바로 이 경우라, 두 슬롯에 같은
callable 을 넘깁니다:

```python
death_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)

asmp = fcf.Assumptions(
    mortality_annual = death_fn,                              # 보유계약 감쇠
    coverages        = (fcf.CoverageRate("DEATH", death_fn),) # 사망보장 청구율
)
```

한 변수 `death_fn` 을 두 슬롯에 공유. "두 율이 같다" 는 사실을 코드에
명시하는 동시에, 한 쪽만 바꾸는 silent 어긋남이 구조적으로 차단됩니다.

## 1.3.3 다른 숫자일 때 — 다종 사망보장

한 계약에 사망보장이 세 개 붙으면 이야기가 달라집니다.

```
asmp = fcf.Assumptions(
    mortality_annual = rate_all_cause,             # all-cause (예: 연 1.2%)
    coverages = (
        fcf.CoverageRate("DEATH",         rate_all_cause),  # 일반사망 (1.2%)
        fcf.CoverageRate("ADB",           rate_accident),   # 재해사망 (0.3%)
        fcf.CoverageRate("DISEASE_DEATH", rate_disease),    # 질병사망 (0.9%)
    )
)
```

사람은 한 번만 죽으므로 `in_force` 는 all-cause 율로 한 번 감쇠합니다.
하지만 그 한 번의 사망 사건이 *여러 보장* 의 청구를 동시에 트리거 (재해로
죽으면 일반사망금 + 재해사망금 둘 다 지급). 세 청구 rate 의 합 (0.3 + 0.9
= 1.2) 이 all-cause 와 일치 — 같은 사망 사건의 원인별 분해라서.

여기서 `mortality_annual` 을 분리하지 않고 보장 청구 식 안에 묶었다면
세 사망 보장 중 어느 하나의 rate 로만 in-force 를 감쇠해야 했을 겁니다.
분리되어 있으니 감쇠는 contract 전체에 한 번, 청구는 보장마다 자기 율.

또 다른 경우는 calibration 출처가 다른 사정 — 보유계약 감쇠는 경험률표
(실측 자료) 로, 사망보험금 지급률은 공시 표준 mortality table (규제
의무) 로. 두 율이 다른 source 에서 나오니 다른 숫자.

## 1.3.4 손계산 wiring 의 안전 패턴

위 사례들이 fastcashflow 가 분리한 이유. 단순 정기보험 손계산은 두 값이
같지만, 두 슬롯에 별개의 값을 실수로 넘기면 결과가 silent 로 어긋납니다.

| 잘못된 wiring | 결과 |
|---|---|
| `mortality_annual = 0`, DEATH rate = 1% | 보유계약 안 줄고 매월 사망보험금 청구. 무한 사망. BEL 과대평가. |
| `mortality_annual = 1%`, DEATH rate = 0 | 보유계약은 줄고 사망보험금 없음. BEL 과소평가. |

두 패턴 모두 에러 없이 통과 — silent footgun. 한 변수 공유 패턴 (`death_fn`
을 두 슬롯에) 이 구조적 방어.

```{admonition} 워크북 로더는 자동 처리
:class: note

`fcf.read_assumptions("assumptions.xlsx")` 로 워크북에서 가정을 읽으면,
`segments` 시트의 `mortality_table` 컬럼이 두 슬롯 (`mortality_annual` /
`coverages` 의 DEATH `rate_table`) 를 같은 `table_id` 로 자동 매핑.
손계산 / 단위테스트로 직접 `Assumptions(...)` 를 채울 때만 주의.
```

## 1.3.5 다음 챕터로

이 분리를 이해하고 [1.4 보장 청구 메커니즘](coverage-mechanics) 으로
가면, `in_force × rate × benefit` 식의 `in_force` 는 `mortality_annual`
이 감쇠시키는 공유 풀, 각 보장의 `rate` 는 그 보장의 청구 빈도임이
자연스럽게 읽힙니다. DIAGNOSIS 가 별도 `undiagnosed` 풀을 갖는 이유도
같은 결.

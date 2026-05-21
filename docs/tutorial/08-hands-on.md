# 8장. 직접 돌려보기

```{admonition} 이 장에서 배우는 것
:class: tip

- 모델포인트와 계리적 가정을 코드로 만들기
- measure()로 측정을 실행하고 BEL·RA·CSM 읽기
- value()로 대규모 포트폴리오를 빠르게 평가하기
- 튜토리얼을 마치며
```

1장부터 7장까지, 보험계약부채를 측정한다는 것이 무엇인지 — 추정,
할인, 위험조정, 이익 분리 — 를 손으로 따라왔습니다. 이 장에서는 그
계산을 fastcashflow 엔진으로 직접 돌려 봅니다. 5~7장에서 손으로 구한
바로 그 계약을, 이번엔 코드로요.

## 8.1 입력 만들기

엔진의 입력은 둘뿐입니다(1.6절) — **계리적 가정**과 **모델포인트**.

계리적 가정은 `Assumptions`로 만듭니다. 5~7장 예제의 가정 — 월 사망률
1%, 해지 없음, 월 할인율 0.5%, 사업비 0, 신뢰수준 75%, 사망위험
변동계수 0.10 — 을 그대로 옮깁니다.

```python
import numpy as np
import fastcashflow as fcf

asmp = fcf.Assumptions(
    mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.01),
    lapse_monthly=lambda duration: np.full(duration.shape, 0.0),
    discount_annual=1.005 ** 12 - 1,     # 월 0.5%의 연 환산
    expense_acquisition=0.0,
    expense_maintenance_annual=0.0,
    expense_inflation=0.0,
    ra_confidence=0.75,
    mortality_cv=0.10,
)
```

사망률과 해지율은 숫자가 아니라 **함수**로 줍니다. 나이와 경과기간에
따라 달라질 수 있기 때문이죠(3.2절). 여기서는 어디서나 같은 값을
돌려주는 간단한 함수를 썼습니다. 할인율은 엔진이 연 단위로 받으므로,
5장의 월 0.5%에 해당하는 연율을 넣었습니다.

모델포인트는 `ModelPointSet`으로 만듭니다. 계약 한 건이면 `single()`이
편합니다.

```python
mps = fcf.ModelPointSet.single(
    issue_age=40, death_benefit=12_000,
    monthly_premium=100, term_months=2,
)
```

2개월 정기보험 한 건, 5~7장의 그 계약입니다.

## 8.2 측정 실행과 결과 읽기

입력이 준비됐으면 측정은 한 줄입니다.

```python
m = fcf.measure(mps, asmp)
```

`measure()`는 1.2절의 4단계 — 추정, 할인, 위험조정, 이익 분리 — 를
모두 수행하고 측정 결과 객체를 돌려줍니다. 거기서 BEL·RA·CSM과
손실요소를 꺼냅니다.

```python
print(m.bel[0, 0])           # BEL
print(m.ra[0, 0])            # RA
print(m.csm[0, 0])           # CSM
print(m.loss_component[0])   # 손실요소
```

```
39.1082
16.0269
0.0
55.1351
```

`m.bel`은 시점별 BEL을 담은 궤적이라(5.3절의 BEL 곡선), `[0, 0]`은 첫
모델포인트의 **최초 인식 시점** 값입니다.

이 숫자들을 알아보시겠습니까? **5장에서 손으로 구한 BEL 39.10,
6장에서 구한 RA 16.01**입니다. 손계산은 할인계수를 반올림해 썼으니
끝자리만 미세하게 다를 뿐, 엔진이 내놓은 39.11과 16.03은 같은
값입니다. FCF = BEL + RA가 양수라 7.1절에서 본 대로 **손실부담계약** —
CSM은 0, 손실요소는 55.14입니다.

일곱 장에 걸쳐 손으로 따라온 측정을, 엔진은 다섯 줄로 똑같이 해냅니다.

## 8.3 대규모 평가

계약 한 건이 아니라 수백만 건이라면 어떨까요? `measure()`는 시점별
궤적까지 다 담아 무겁습니다. 그럴 땐 **`value()`**를 씁니다.
BEL·RA·CSM·손실요소 네 숫자만 모델포인트마다 하나씩 돌려주는 빠른
경로입니다.

```python
v = fcf.value(mps, asmp)
print(v.bel, v.ra, v.csm, v.loss_component)
```

대규모 포트폴리오는 보통 파일에서 읽어 옵니다.

```python
mps  = fcf.read_model_points("portfolio.parquet")
asmp = fcf.read_assumptions("basis.xlsx")
val  = fcf.value(mps, asmp)
fcf.write_valuation(val, "results.parquet")
```

빠르기는 이 패키지의 존재 이유입니다. `examples/benchmark.py`로 재면,
100만 계약을 120개월 평가하는 데 약 0.05초, 500만 건이면 약
0.3초입니다. 메모리에 다 올리기 버거운 규모는 `value_file()`이 파일을
조각조각 흘려 가며 처리합니다.

## 8.4 마치며

여기까지가 튜토리얼입니다. 되짚어 보면:

- **1~2장** — IFRS 17이 무엇이고, 보험계약부채가 BEL·RA·CSM으로
  이뤄진다는 것
- **3~4장** — 모델포인트와 계리적 가정이라는 입력, 그리고 엔진이
  현금흐름을 만들어 내는 방식
- **5~7장** — BEL·RA·CSM을 차례로 계산하고 손으로 검증
- **8장** — 엔진으로 직접 측정

이제 `measure()` 한 줄 안에서 무엇이 어떤 순서로 일어나는지를 끝까지
펼쳐 본 셈입니다. 1.6절에서 한 약속을 지킨 거죠.

마지막으로 한 가지. 엔진이 내놓는 BEL·RA·CSM은 **넣어 준 가정만큼만**
정확합니다. 엔진은 모형을 충실하고 빠르게 계산하지만, 그 가정이
현실에 맞는지는 말해 주지 못합니다. 가정을 세우고 검증하는 일 —
계리사의 판단 — 이야말로 측정에서 가장 어렵고 중요한 몫입니다. 이
엔진은 그 판단을 숫자로 옮기는 빠르고 투명한 도구입니다.

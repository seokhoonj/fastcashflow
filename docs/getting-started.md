# 시작하기

## 설치

PyPI에는 아직 올라가 있지 않습니다. GitHub에서 직접 설치합니다:

```bash
pip install git+https://github.com/seokhoonj/fastcashflow.git
```

## 첫 평가

평가에는 입력이 둘 필요합니다 — 모델포인트(측정할 계약들)와 산출기초
(가정). 가장 빠른 시작은 fastcashflow에 들어 있는 샘플 데이터입니다.
파일을 따로 준비할 필요가 없습니다.

```python
import fastcashflow as fcf

basis        = fcf.samples.basis()[("TERM_LIFE_A", "GA")]   # 한 세그먼트의 산출기초(가정)
model_points = fcf.samples.model_points()

m = fcf.gmm.measure(model_points, basis)
print(m)   # BEL · RA · CSM · 손실요소 한눈에 (계약별 + 합계)
```

출력:

```text
GMMMeasurement -- 11 model points
                   BEL            RA           CSM          loss
    mp 0     2,690,944        34,842             0     2,725,785
    mp 1     2,698,298        63,722             0     2,762,020
    mp 2       757,630        39,358             0       796,988
    mp 3     5,052,908        86,369             0     5,139,277
    mp 4     1,594,718       118,529             0     1,713,247
    mp 5     3,946,952        59,748             0     4,006,700
    mp 6     1,630,562       114,642             0     1,745,204
    mp 7     2,915,954        69,259             0     2,985,213
    mp 8     6,211,504       123,522             0     6,335,025
    mp 9     2,679,645        85,006             0     2,764,651
     ...  (1 more model points)
   Total    33,470,642       847,965             0    34,318,606
```

`measure`는 각 계약을 월 단위로 추정해 IFRS 17 보험계약부채를 시점별로
펼쳐 냅니다. 한 줄을 더하면 그 결과를 차트로 볼 수 있습니다.

```python
fcf.plot_liability(m)
```

```{image} images/first-valuation.png
:alt: 계약 기간에 걸친 BEL·RA·CSM 궤적
:class: hero
```

BEL·RA·CSM·손실요소 네 값만 빠르게 얻으려면 `measure(..., full=False)` 를
씁니다. 시점별 궤적을 만들지 않는, 메모리를 거의 안 쓰는 빠른 경로입니다.

## 다음으로

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 튜토리얼
:link: tutorial/index
:link-type: doc

IFRS 17 측정을 개념부터 실무까지 차근차근 익힙니다.
:::

:::{grid-item-card} API 레퍼런스
:link: api
:link-type: doc

모든 함수와 결과 타입, 전체 시그니처.
:::

::::

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
<GMMMeasurement -- 11 model points>
                   BEL            RA           CSM          loss
    mp 0       460,519        40,285             0       500,804
    mp 1      -892,100       150,809       741,291             0
    mp 2      -243,286        33,283       210,004             0
    mp 3       698,622       111,521             0       810,143
    mp 4      -491,236       136,118       355,118             0
    mp 5       309,596        45,544             0       355,139
    mp 6       666,248       103,621             0       769,869
    mp 7      -978,975       150,596       828,379             0
    mp 8   -10,913,881       207,244    10,706,636             0
    mp 9   -10,286,113       144,186    10,141,927             0
     ...  (1 more model points)
   Total   -22,295,026     1,210,877    23,520,103     2,435,955
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

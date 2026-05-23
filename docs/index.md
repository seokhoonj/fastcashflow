# {{ fcf }}

현금흐름 추정부터 손익 리포트까지 — 오픈소스 IFRS 평가 엔진

```{image} images/hero.png
:alt: fastcashflow가 산출하는 IFRS 17 수치 — 부채 구성요소, 보험계약마진 상각, 변동분석
:class: hero
```

{{ fcf }}는 보험계약부채 — 최선추정부채(BEL), 위험조정(RA),
보험계약마진(CSM) — 를 계약 건별로, IFRS 17의 세 가지 회계모형
모두에 대해 측정합니다.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} 속도
numba로 컴파일한 수치 커널 — 계약 건별 대규모 평가를 위한
설계.
:::

:::{grid-item-card} 폭넓은 기능
세 가지 회계모형(GMM·PAA·VFA)을 모두 — 재보험, 확률론적 평가,
기말 변동분석.
:::

:::{grid-item-card} 투명성
읽기 쉬운 코드, 교육 수준의 docstring, 이해를 돕는 내장 차트.
:::

:::{grid-item-card} 오픈소스
자유로운 오픈소스, MPL-2.0 라이선스.
:::

::::

```{toctree}
:hidden:

시작하기 <start>
튜토리얼 <tutorial/index>
쿡북 <cookbook/index>
api
```

# 시작하기

## 설치

PyPI에는 아직 올라가 있지 않습니다. GitHub에서 직접 설치합니다:

```bash
pip install "git+https://github.com/seokhoonj/fastcashflow.git#egg=fastcashflow[viz]"
```

`[viz]`는 아래에서 쓰는 내장 차트에 필요한 matplotlib을 함께 설치합니다.
차트가 필요 없으면 `[viz]`를 빼고 `pip install
git+https://github.com/seokhoonj/fastcashflow.git` 로 설치합니다.

## 첫 평가

평가에는 입력이 둘 필요합니다 — 모델포인트(측정할 계약들)와 계리적
가정. 가장 빠른 시작은 fastcashflow에 들어 있는 샘플 데이터입니다.
파일을 따로 준비할 필요가 없습니다.

```python
import fastcashflow as fcf

basis        = fcf.load_sample_assumptions()       # {(product, channel): Assumptions}
assumptions  = basis[("term_a", "GA")]             # 한 세그먼트 선택
model_points = fcf.load_sample_model_points()

m = fcf.measure(model_points, assumptions)
print(m.bel[:, 0])   # 최선추정부채
print(m.ra[:, 0])    # 위험조정
print(m.csm[:, 0])   # 보험계약마진
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

BEL·RA·CSM·손실요소 네 값만 빠르게 얻으려면 `measure` 대신 `value`를
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

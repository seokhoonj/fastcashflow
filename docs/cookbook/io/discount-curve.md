# 7.3 할인율 곡선 구성 — Smith-Wilson

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 관측 시장금리(국고채)에서 **만기별 할인율 곡선**을 만드는 법 — `fcf.smith_wilson`
- **Smith-Wilson**: 관측 구간은 정확히 통과시키고, 장기 구간은 **UFR(ultimate
  forward rate, 장기수렴 선도금리)** 로 외삽 — K-IFRS17/K-ICS가 보험부채 할인율에
  규정하는 방법
- 입력 — 관측 만기·금리 + UFR + `alpha`(수렴속도) — 하나로 **원화·외화 모두** 구성
- 만든 곡선을 그대로 `Basis.discount_annual` 로 넣기
:::

앞 챕터(7.1/7.2)는 `discount_annual` 곡선을 **이미 만들어진 데이터**로 받았습니다.
이 챕터는 그 곡선을 **시장금리에서 직접 구성**합니다.

## 왜 Smith-Wilson 인가

K-IFRS17/K-ICS 보험부채 할인율 곡선은 **관찰 구간(국고채) + 보간/외삽(LTFR 수렴)**
으로 만듭니다. 그 **만기수익률→현물 전환과 보간을 Smith-Wilson 으로** 하도록
금감원 「책임준비금 외부검증 가이드라인」이 규정합니다(외화 곡선도 동일 방법).

Smith-Wilson 은 금리 *모형*이 아니라 곡선 *구축 알고리즘* 입니다:

- **관측 구간** — 입력 금리를 **정확히 재현**(근사 아님).
- **장기 구간** — 선도금리가 **UFR 로 수렴**. 수렴 속도는 `alpha`.
- **통화 무관** — 알고리즘은 하나. 원화/외화 차이는 **입력**(관측금리, LLP, UFR,
  alpha) 뿐.

## 모델링 매핑

:::{list-table}
:header-rows: 1
:widths: 34 66

* - 입력
  - 무엇
* - `maturities`
  - 관측 만기(년). 가장 큰 값이 LLP(last liquid point, 최종관찰만기 — 원화 20년)
* - `rates`
  - 그 만기의 관측 무위험 현물금리(원화=국고채)
* - `ufr`
  - 장기수렴 선도금리(LTFR/UFR). 금감원 공시 — 2026년 원화 4.05%
* - `alpha`
  - 수렴 속도(모형 밖에서 선택, 통화별 공시값)
:::

:::{note}
`smith_wilson` 은 **무위험 곡선**만 만듭니다. 유동성프리미엄(LP)은 규제 기준에
따라 입력 금리나 산출 곡선에 **따로 가산**하세요. 곡선은 연 1회 결정·고정입니다.
:::

## 작동 예제 — 국고채에서 곡선 구성

ECOS 국고채 현물금리(관측)에 UFR 4.05%, `alpha` 0.10 으로 100년 곡선을 만듭니다.

```python
import numpy as np
import fastcashflow as fcf

mat  = np.array([1, 2, 3, 5, 10, 20], dtype=float)        # 관측 만기 (년), LLP=20
rate = np.array([0.0310, 0.0355, 0.0368, 0.0390, 0.0408, 0.0410])  # 국고채 현물

curve = fcf.smith_wilson(mat, rate, ufr=0.0405, alpha=0.10, years=100)

for y in (1, 5, 10, 20, 30, 50, 100):
    print(f"{y:>4d}y spot = {curve[y - 1]:>8.4%}")
```

출력:

```text
   1y spot =   3.1000%
   5y spot =   3.9000%
  10y spot =   4.0800%
  20y spot =   4.1000%
  30y spot =   4.0892%
  50y spot =   4.0753%
 100y spot =   4.0628%
```

1·5·10·20년은 관측금리를 **정확히** 통과하고(3.10/3.90/4.08/4.10%), LLP(20년)를
지난 30·50·100년은 UFR 4.05% 로 **수렴**합니다.

만든 곡선은 바로 `Basis.discount_annual` 입니다:

```python
basis = fcf.Basis(
    mortality_annual = 0.004,
    lapse_annual     = 0.03,
    discount_annual  = curve,        # Smith-Wilson 곡선을 그대로
    ra_confidence    = 0.75,
    mortality_cv     = 0.10,
    coverages        = (fcf.CoverageRate("DEATH", 0.004),),
)
```

## 검산 — 곡선의 두 성질

- **관측점 정확 재현** — `smith_wilson` 은 보간이 아니라 **fit** 이라, 관측 만기에서
  입력 금리를 기계정밀도로 재현합니다.
- **UFR 수렴** — LLP 이후 1년 선도금리가 UFR 로 수렴합니다. `alpha` 가 클수록 빨리
  수렴(장기 데이터에 가중 ↓), 작을수록 천천히(관측 데이터에 가중 ↑).

## 함정

### 함정 1 — `alpha` 와 UFR 가 결과를 좌우한다

Smith-Wilson 자체보다 **UFR 와 LLP·alpha 선택**이 장기 곡선에 훨씬 큰 영향을
줍니다. 원화는 EUR 만큼 LLP/UFR 이 규제로 못박혀 있지 않아 더 민감합니다 —
**금감원 공시값**을 쓰세요.

### 함정 2 — LP 는 따로

`smith_wilson` 출력은 무위험 곡선입니다. IFRS17(×100%) / K-ICS(변동성조정 ×80%)
유동성프리미엄은 규제 기준에 맞춰 **별도 가산**해야 합니다.

## 인접 레시피

- [7.1 워크북 — 단일 segment](workbook-single) — `discount_annual` 을 워크북
  `discount_tables` 시트로 받는 경로(이미 만들어진 곡선).
- [2.2 BEL · RA · CSM](../../tutorial/02-bel-ra-csm) — 이 곡선으로 할인하는 측정.

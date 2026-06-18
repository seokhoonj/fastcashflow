# 경제 시나리오 생성 — Hull-White + 변액

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 변액·UL 보증평가(TVOG)와 확률적 측정이 먹는 **위험중립 시나리오** 를 fcf 안에서
  직접 만드는 법 — `fcf.esg.simulate`
- **Hull-White 1-factor** 단기금리(`gmm.stochastic` 입력)와 **로그정규(GBM)** 펀드수익률
  (`vfa.measure` 입력)을 **상관** 시켜 한 번에 생성
- 곡선을 정확히 재현하도록 **Smith-Wilson 곡선에 캘리브레이션** 하고, 무차익 성질
  (**마팅게일**)으로 검증하는 법
- 생성한 시나리오를 그대로 `vfa.measure(return_scenarios=...)` 에 먹여 TVOG 평가
:::

fcf는 TVOG/확률측정에서 시나리오를 **소비** 하지만, 번들 샘플(`samples.return_scenarios`)은
스스로 "calibrated economic scenario generator가 아니다"라고 적은 toy 입니다. 이 챕터의
`fcf.esg` 는 [할인율 곡선](discount-curve)에서 만든 곡선을 받아 **그 곡선을 재현하는
위험중립 시나리오** 를 만듭니다 — "곡선 -> 시나리오 -> TVOG" 가 fcf 안에서 닫힙니다.

## 두 요인 — 금리와 펀드

- **Hull-White 1-factor 단기금리** `dr = (theta(t) - a*r)dt + sigma*dW`. `theta(t)` 를
  Smith-Wilson 할인곡선에 캘리브레이션해 모델이 그 곡선을 **정확히 재현**. 출력은
  월별 **연율** 로, `gmm.stochastic` 의 금리시나리오 입력.
- **로그정규(GBM) 펀드수익률** -- 위험중립 drift = 단기금리. 출력은 월 **단순수익률** 로,
  `vfa.measure` / `measure_tvog` 의 `return_scenarios` 입력. 항상 > -1.
- 두 요인의 **상관**(rate/equity) 을 Cholesky로 한 번에 생성.

평균회귀 `a` 와 변동성(`rate_vol`, `equity_vol`)은 **입력** 입니다(스왑션 곡면 캘리브레이션은
범위 밖 -- 아래 범위 참조). fcf는 곡선 캘리브레이션·마팅게일 검증·분산축소를 책임집니다.

## 최소 작동 예제 — 생성 + 무차익 검증

```python
import numpy as np
import fastcashflow as fcf

mat  = np.array([1, 2, 3, 5, 10, 20], dtype=float)   # observed maturities (years)
rate = np.array([0.0310, 0.0355, 0.0368, 0.0390, 0.0408, 0.0410])  # spot rates

es = fcf.esg.simulate(
    mat, rate, ufr=0.0405, alpha=0.10,    # the Smith-Wilson curve to calibrate to
    mean_reversion=0.10, rate_vol=0.01,   # Hull-White 1-factor short rate
    equity_vol=0.15, correlation=-0.20,   # lognormal fund return, rate/equity corr
    n_scenarios=20000, n_time=240, seed=20260619)

bond_err, equity_err = es.martingale_error()
print(f"rates    {es.rates.shape}   annual rate per month -> gmm.stochastic")
print(f"returns  {es.returns.shape}   monthly fund return  -> vfa.measure")
print(f"bond reprice error     {bond_err:.2e}")
print(f"equity martingale err  {equity_err:.2e}")
```

출력:

```text
rates    (20000, 240)   annual rate per month -> gmm.stochastic
returns  (20000, 240)   monthly fund return  -> vfa.measure
bond reprice error     3.28e-04
equity martingale err  1.34e-03
```

`martingale_error()` 가 **확률 생성기의 정확성 검증** 입니다(손계산이 아니라 무차익 성질로
검증 -- Smith-Wilson을 exact-fit으로 검증한 것과 같은 결):

- **채권 마팅게일** — 시나리오의 확률할인계수 평균 `mean exp(-sum r/12)` 이 곡선의 `P(0,T)`
  를 재현. drift를 이산 단계에서 정확 캘리브레이션해 **편향 0**, 남는 건 MC 노이즈뿐이라
  시나리오를 늘리면 오차가 `1/sqrt(n)` 로 줄어듭니다.
- **주식 마팅게일** — 할인된 펀드가치 평균이 1(펀드는 거래가능자산이라 할인가치가 마팅게일).

작은 오차는 시나리오 수의 MC 노이즈입니다 -- 정밀 평가엔 `n_scenarios` 를 키우고 antithetic
(대조변량, 기본 on)으로 분산을 줄입니다.

## TVOG에 먹이기 — 변액 보증 시간가치

생성한 펀드수익률을 그대로 `vfa.measure` 의 `return_scenarios` 로 넣으면 변액 보증의 시간가치
(TVOG)가 나옵니다(샘플 변액 계약의 보장기간에 맞춰 재생성):

```python
import numpy as np
import fastcashflow as fcf
mat  = np.array([1, 2, 3, 5, 10, 20], dtype=float)
rate = np.array([0.0310, 0.0355, 0.0368, 0.0390, 0.0408, 0.0410])

vmp, vbasis = fcf.samples.model_points("vfa"), fcf.samples.basis("vfa")
n_time = int(np.asarray(vmp.term_months).max())

es = fcf.esg.simulate(mat, rate, ufr=0.0405, alpha=0.10, mean_reversion=0.10,
                      rate_vol=0.01, equity_vol=0.15, correlation=-0.20,
                      n_scenarios=2000, n_time=n_time, seed=20260619)
vm = fcf.vfa.measure(vmp, vbasis, return_scenarios=es.returns)
print(f"VFA guarantee time value (TVOG)  {float(np.sum(vm.time_value)):,.0f}")
```

출력:

```text
VFA guarantee time value (TVOG)  947,191,493
```

`es.rates` 는 같은 방식으로 `gmm.stochastic(model_points, basis, es.rates)` 에 들어갑니다
(금리시나리오, 연율). `EconomicScenarios.rates`/`.returns` 가 곧 소비측 입력이라 추가 변환이
없습니다.

## 함정 / 범위 (v1 = 위험중립 최소판)

- **결정론** — 같은 `seed` 는 동일 배열. 평가 재현성을 위해 seed를 고정·기록하세요.
- **단위** — `rates` 는 **연율**(엔진이 `(1+r)^(1/12)-1` 로 월율 변환), `returns` 는 **월
  단순수익률**(`> -1` 강제). 헷갈리면 안 됩니다.
- **넣음**: HW1F(연율 금리) + GBM(월수익률), Smith-Wilson 캘리브레이션, 상관, 마팅게일
  검증, 대조변량.
- **뺌(연기)**: 스왑션 vol-곡면 캘리브레이션(여기선 `a`/`sigma` 를 입력으로 받음),
  준난수(Sobol), 실세계(real-world) 측도, 다요인/확률변동성(LMM/Heston) -- 전체판 ESG는
  별도 제품 영역.

## 인접 레시피

- [할인율 곡선 — Smith-Wilson](discount-curve) — 이 생성기가 캘리브레이션하는 곡선.
- [변액 보증 / 크레딧 floor TVOG](../variable/crediting-floor) — `return_scenarios` 를
  받는 보증 시간가치 평가.

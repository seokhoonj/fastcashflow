# 10장. 실제 업무에서의 활용 (2)

```{admonition} 이 장에서 배우는 것
:class: tip

- 측정 결과를 차트로 보기
- 보고기간별 변동분석 (analysis of change)
- IFRS 17 공시 — 보험손익 만들기
```

9장에서 입력을 파일로 읽어 평가하고 저장하는 흐름을 봤습니다. 이 장은
그 측정 결과로 실무에서 하는 세 가지 — 그림으로 보기, 기간별 변동을
분석하기, 공시로 만들기 — 를 다룹니다.

세 절 모두 같은 측정 결과를 씁니다. 먼저 한 번 만들어 둡니다.

```python
import fastcashflow as fcf

mps  = fcf.load_sample_model_points()
asmp = fcf.load_sample_assumptions()
m    = fcf.measure(mps, asmp)
```

## 10.1 결과를 그림으로

`measure()`가 돌려준 `m`은 시점별 궤적을 모두 담고 있어, 숫자만으로는
한눈에 들어오지 않습니다. 내장 차트가 그것을 한 줄로 보여 줍니다.
차트 기능은 8.1절의 `[viz]` 추가 설치에 들어 있는 matplotlib을 씁니다.

```python
fcf.plot_liability(m)               # BEL·RA·CSM 궤적
fcf.plot_cashflows(m)               # 현금흐름의 여섯 갈래
fcf.plot_csm_runoff(m)              # CSM 런오프
fcf.plot_risk_adjustment(m, asmp)   # 위험조정
```

- `plot_liability` — 5·6·7장의 BEL·RA·CSM이 시간에 따라 어떻게
  변하는지.
- `plot_cashflows` — 4장에서 본 현금흐름 여섯 갈래.
- `plot_csm_runoff` — 7장의 CSM이 보장기간에 걸쳐 풀려 나가는 모습.
- `plot_risk_adjustment` — 위험조정. 가정도 함께 넘깁니다.

각 함수는 차트 한 장을 그리고 그 객체를 돌려줍니다 — 화면에 띄우거나
파일로 저장할 수 있습니다.

```python
ax = fcf.plot_liability(m)
ax.figure.savefig("liability.png")
```

## 10.2 변동분석

IFRS 17 공시는 부채의 *잔액*만 보이는 것으로 끝나지 않습니다. 한
보고기간 동안 부채가 **어떻게 움직였는지** — 기초잔액에서 이자, 상각,
경험조정 등을 거쳐 기말잔액에 이르기까지 — 를 보여야 합니다. 이것이
변동분석(analysis of change)입니다.

`roll_forward`가 측정 결과를 보고기간으로 자르고, `reconcile`이 각
기간의 움직임을 기초에서 기말까지 정확히 맞아떨어지는 변동분석표로
만듭니다.

```python
movements = fcf.roll_forward(m, period_months=12)   # 12개월 기간으로
recon = fcf.reconcile(movements)
print(recon[0])                                     # 첫 보고기간
```

`recon`은 보고기간 수만큼의 변동분석이고, `recon[0]`은 첫 기간의
표입니다 — BEL·RA·CSM 각각의 기초·이자·상각·기말이 한 줄씩.
`plot_analysis_of_change`가 그 표를 그림으로 그립니다.

```python
fcf.plot_analysis_of_change(recon[0])
```

## 10.3 공시

측정과 변동분석이 *부채* 쪽이라면, 공시는 *손익* 쪽입니다. IFRS 17의
보험손익 — **보험수익**에서 **보험서비스비용**을 뺀 **보험서비스결과**
— 을 `report()`가 측정 결과에서 만들어 냅니다.

```python
rep = fcf.report(m)
annual = rep.annual()               # 연도별 포트폴리오 합계
print(annual["insurance_revenue"])
print(annual["insurance_service_result"])
```

`report()`가 돌려주는 `Report`에는 보험수익, 보험서비스비용,
보험서비스결과, 보험금융비용, 그리고 CSM 변동내역이 들어 있습니다.
`.annual()`은 이를 모델포인트 전체로 합산하고 연 단위로 묶어 줍니다 —
공시 표에 바로 쓸 수 있는 모습입니다.

## 10.4 다음 장

8~10장에서 엔진을 직접 돌려 측정하고, 파일로 다루고, 결과를 그림과
공시로 풀어냈습니다. 다음 장은 본문에서 한 걸음 비켜서서, 이 엔진이 왜
그렇게 빠른지 그리고 그 속도가 무엇을 대가로 지불했는지를 봅니다.

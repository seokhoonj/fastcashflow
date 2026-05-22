# 10장. 실무에서의 활용 (2)

```{admonition} 이 장에서 배우는 것
:class: tip

- 측정 결과를 차트로 보기
- 보고기간별 변동분석 (analysis of change)
- 보험손익 리포트 만들기
```

9장에서 입력을 파일로 읽어 평가하고 저장하는 흐름을 봤습니다. 이 장은
그 측정 결과로 실무에서 하는 세 가지 — 그래프로 보기, 기간별 변동을
분석하기, 리포트로 정리하기 — 를 다룹니다.

세 절 모두 같은 측정 결과를 씁니다. 8장에서 쓴 내장 샘플 — 패키지에
저장돼 있는 계약 8건과 그에 맞는 가정 — 을 `load_sample_*`로 그대로
불러와 한 번 측정해 둡니다. 따로 파일을 준비할 필요가 없습니다.

```python
import fastcashflow as fcf

mps  = fcf.load_sample_model_points()   # 패키지에 저장된 샘플 포트폴리오
asmp = fcf.load_sample_assumptions()    # 그에 맞는 샘플 가정
m    = fcf.measure(mps, asmp)
```

## 10.1 결과를 그래프로

`measure()`가 돌려준 `m`은 시점별 궤적을 모두 담고 있어, 숫자만으로는
한눈에 들어오지 않습니다. 내장 차트가 그것을 그래프로 보여 줍니다.
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

예를 들어 `plot_liability(m)`는 위 샘플 포트폴리오의 BEL·RA·CSM이
보험기간에 걸쳐 어떻게 변해 가는지를 이렇게 그립니다.

```{image} ../images/liability-trajectory.png
:alt: 내장 샘플 포트폴리오의 BEL·RA·CSM이 보험기간에 걸쳐 변해 가는 궤적
:class: hero
```

각 함수는 차트를 그려 matplotlib의 `Axes` 객체로 돌려줍니다. 함수 자체는
화면에 띄우지는 않습니다 — Jupyter 노트북에서는 셀 결과로 바로 그래프가
나오고, 스크립트에서는 `matplotlib.pyplot.show()`를 불러야 창이 뜹니다.
파일로 저장할 때는 돌려받은 객체를 씁니다.

```python
import matplotlib.pyplot as plt

ax = fcf.plot_liability(m)
ax.figure.savefig("liability.png")   # 파일로 저장
plt.show()                           # 스크립트에서 화면에 띄우기
```

## 10.2 변동분석

IFRS 17은 부채의 잔액만 보이는 것으로 끝나지 않습니다. 한
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

실행하면 첫 보고기간(1차년도)의 변동분석표가 나옵니다.

```
Reconciliation -- months 0-12
                               BEL                RA               CSM
Opening                -17,618,497         1,269,709        16,348,789
Future service                   0                 0                 0
Finance                   -539,428            36,563           463,678
Release                  1,526,019          -108,963        -1,919,700
Closing                -16,631,907         1,197,309        14,892,767
```

`recon`은 보고기간 수만큼의 변동분석입니다. 표는 BEL·RA·CSM을 열로,
한 보고기간의 움직임을 다섯 행으로 보여 줍니다.

```{list-table}
:header-rows: 0
:widths: 30 70

* - Opening (기초잔액)
  - 보고기간이 시작될 때의 잔액. 한 기간의 Closing이 다음 기간의 Opening이 됩니다.
* - Future service (미래서비스 관련 변동)
  - 아직 제공하지 않은 미래 보장에 관한 변동(가정 재산정, 미래 현금흐름 추정 변경, 신계약 등). GMM에서는 손익이 아니라 CSM을 조정합니다. 위 표에서 0인 것은 가정을 고정한 결정론적 투영을 그대로 굴려 바뀔 것이 없기 때문입니다.
* - Finance (보험금융손익)
  - 시간이 흐르며 생기는 할인·이자 효과. BEL·RA는 할인이 풀리고, CSM은 최초 인식 시점에 고정된 이자율로 부리됩니다.
* - Release (당기 서비스 제공분)
  - 이번 기간에 보장을 제공한 만큼 풀려 나간 부분. CSM 상각, RA 해제, 당기 예상 현금흐름의 런오프.
* - Closing (기말잔액)
  - 보고기간이 끝날 때의 잔액. Opening에 위 세 변동을 더하면 정확히 Closing이 됩니다.
```

`plot_analysis_of_change`가 그 표를 그래프로 그립니다.

```python
fcf.plot_analysis_of_change(recon[0])
```

기본값은 CSM 성분이며(`component`로 BEL·RA도 선택할 수 있습니다), 표의
다섯 행 — 기초잔액에서 이자·상각을 거쳐 기말잔액까지 — 을 폭포꼴
막대로 잇습니다.

```{image} ../images/analysis-of-change.png
:alt: 첫 보고기간 CSM의 변동분석 폭포 차트 — 기초잔액에서 이자·상각을 거쳐 기말잔액까지
:class: hero
```

## 10.3 리포트

측정과 변동분석이 *부채* 쪽이라면, 리포트는 *손익* 쪽입니다. IFRS 17의
보험손익 — **보험수익**에서 **보험서비스비용**을 뺀 **보험서비스결과**
— 을 `report()`가 측정 결과에서 만들어 냅니다.

```python
rep = fcf.report(m)
print(rep)
```

실행하면 연 단위로 묶은 손익 리포트가 표로 나옵니다.

```
IFRS 17 report -- annual portfolio totals (first 5 of 20 years)
                        Year 1      Year 2      Year 3      Year 4      Year 5
Insurance revenue    6,208,072   3,481,125   3,227,525   3,034,716   2,893,120
Service expense      4,179,678   1,652,084   1,555,724   1,486,056   1,439,495
Service result       2,028,394   1,829,042   1,671,801   1,548,660   1,453,625
Finance expense        -39,187       4,970      45,833      79,226     106,415
CSM accretion          463,678     422,734     385,750     351,747     319,924
CSM release          1,919,700   1,727,699   1,575,954   1,456,707   1,364,169
```

`print`은 `Report`를 모델포인트 전체로 합산하고 연 단위로 묶어, 앞 다섯
보고연도를 여섯 행으로 보여 줍니다.

```{list-table}
:header-rows: 0
:widths: 30 70

* - Insurance revenue (보험수익)
  - 그 해에 보험서비스를 제공하고 인식한 수익. 예상 보험서비스비용에 RA 해제분과 CSM 상각분을 더한 값입니다(IFRS 17 B120~B124).
* - Service expense (보험서비스비용)
  - 그 해에 실제 발생한 보험서비스비용: 보험금, 진단·입원 등 급부, 사업비.
* - Service result (보험서비스결과)
  - 보험수익에서 보험서비스비용을 뺀 보험서비스 손익. 곧 RA 해제분과 CSM 상각분의 합입니다.
* - Finance expense (보험금융손익)
  - 할인효과로 생기는 금융손익. BEL·RA의 할인 풀림과 CSM 부리를 합한 것으로, 10.2 표의 Finance 행과 같은 성격입니다.
* - CSM accretion (CSM 부리)
  - CSM에 최초 인식 시점의 이자율로 붙는 이자.
* - CSM release (CSM 상각)
  - 그 해의 보장 제공분만큼 CSM이 손익으로 풀려 나간 금액.
```

첫 해가 가장 크고 해마다 줄어드는 것은 보장이 풀려 나가며 포트폴리오가
점차 소멸하기 때문입니다. 모든 연도의 수치가 필요하면 `rep.annual()`이
각 항목을 연도별 배열로 돌려줍니다 — 표 출력은 앞 다섯 연도만 보여
주지만 배열에는 모든 연도가 들어 있습니다.

```{admonition} 재보험은 빠져 있습니다
:class: note

이 리포트는 원수(발행)계약만 다룹니다. 재보험을 보유한 회사라면 IFRS 17
보험서비스결과에 보유재보험계약의 순손익이 더해지지만, 이 예제에는
재보험이 없어 그 항이 없습니다. fastcashflow는 보유재보험을
`measure_reinsurance`로 따로 측정합니다.
```

## 10.4 다음 장

8~10장에서 엔진을 직접 돌려 측정하고, 파일로 다루고, 결과를 그래프와
리포트로 풀어냈습니다. 다음 장은 본문에서 한 걸음 비켜서서, 이 엔진이
속도를 높이려고 한 일들을 봅니다.

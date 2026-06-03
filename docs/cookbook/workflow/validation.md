# 8.2 검증 패턴

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 한 계약의 BEL / RA / CSM이 **어떤 테이블, 어떤 rate, 어떤 cash flow**
  로 계산되는지 추적하기 — `gmm.trace`
- 월 단위 BEL backward recursion 의 **항별 분해** 와 손계산과의 매칭
  — `gmm.trace_bel_step`
- 월 단위 CSM forward recursion 의 **이자부리 / 환입** 분해
  — `gmm.trace_csm_step`
- 가정 변경 (mortality +10%) 이 **각 단계에 어떻게 전파** 되는지 비교
  — `gmm.trace_diff`

이 챕터는 회사 데이터로 평가를 돌려보고 **결과를 신뢰** 하기 위한
검증 워크플로를 모읍니다. "엔진이 뭘 하는지 보이지 않는다" 는 가장
흔한 도입 마찰을 정면으로 푸는 도구들입니다.
```

## 검증 워크플로 — 왜 / 언제

평가 엔진은 정의상 **블랙박스** 입니다. fastcashflow 도 예외가 아닙니다.
한 줄의 `measure()` 호출이 수십 개의 테이블 lookup, 월별 cash flow 산출,
backward / forward recursion 을 거쳐 BEL / RA / CSM 하나의 숫자로
요약됩니다. 그 숫자를 사내 계리위원회 / 감독당국 / 회계감사인에게
설명해야 할 때, 그리고 회사 손계산 결과와 어긋날 때, **어디서 어떻게
계산됐는지 한눈에 보이는 도구** 가 필요합니다.

전형적인 검증 시나리오는 셋입니다:

1. **신규 도입** — 처음 회사 데이터를 넣고 BEL 값을 받았는데,
   사내 별도 산출치와 X% 차이. 어디서 어긋났는가?
2. **분기 결산** — 직전 분기 대비 BEL이 Y% 움직였는데, 어느 가정 /
   어느 segment 가 주범인가?
3. **시나리오 / 민감도** — "사망률을 10% 올리면 BEL이 얼마나 움직이나"
   — 결과 숫자만 보지 말고 **어디서 어떻게** 변했는지 추적.

세 시나리오 모두 같은 출발점에서 풉니다: **단일 계약 (one model point) 의
계산 경로 전체** 를 보는 것입니다. 포트폴리오 전체를 한꺼번에 보면 노이즈
가 너무 많아 원인이 가려집니다. fastcashflow 의 검증 도구 4종은 **한 계약**
에 초점을 맞춥니다.

## 모델링 매핑 — 4개의 도구

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - 도구
  - 출력
  - 언제 쓰나
* - `gmm.trace`
  - 한 계약의 전체 계산 트리 — segment / 테이블 / coverage / 연도별 rate
    / 월별 cash flow / discount / BEL roll-forward / CSM / 최종 headline
  - 첫 검증. "엔진이 어느 segment 의 어느 테이블을 적용했나"
* - `gmm.trace_bel_step`
  - 월 t 의 BEL 식을 항별로 풀어 표시 — `premium`, `claim`, `expense`,
    `half = (1+i)^(-1/2)`, `BEL[t+1]`, **recomputed vs engine 잔차**
  - 손계산과 엔진 값이 어긋날 때. 어느 항이 다른지
* - `gmm.trace_csm_step`
  - 월 t 의 CSM 식 — 이전 CSM / 이자부리 / coverage_unit 환입비율 /
    환입액 / 잔차
  - CSM 흐름 검증. 손실부담계약 (onerous = 손실부담) 의 floor 가 작동
    하는지 확인
* - `gmm.trace_diff`
  - 두 가정 (basis) 비교 — 바뀐 테이블 / 연도별 rate 변화 / cash flow 의
    전파 / BEL / RA / CSM의 절대·% 변화
  - 시나리오 / 민감도. mortality +10% 가 BEL에 14.2%, RA에 9.8% 라면
    각각 어디서 왔는지
```

모두 ASCII 트리로 출력합니다. 외부 의존성 없고, Jupyter / REPL /
파이프로 파일에 넣기 자유롭습니다. 실제 계산은 새로 하지 않고
`measure()` 의 결과를 **슬라이스해서** 보여줄 뿐이므로 1M 계약 포트폴리오
에서 한 계약만 추적해도 그 한 계약의 비용만 듭니다.

## 최소 작동 예제 — gmm.trace 부터

샘플 워크북의 첫 계약 (TERM_LIFE_A / FC 채널, 가입연령 35, 보험기간 20년)
을 추적합니다.

```python
import fastcashflow as fcf

mp = fcf.samples.model_points()
basis = fcf.samples.basis()

fcf.gmm.trace(0, mp, basis)
```

`mp_index=0` 은 `model_points` 의 첫 행 (0-based) 입니다. `basis` 가
dict 일 때 — `read_basis()` / `samples.basis()` 가
돌려주는 형태 — 그 행의 `(product_code, channel_code)` 로 자동 라우팅합니다.
단일 `Basis` 개체를 직접 넣어도 됩니다.

출력 (발췌):

```
mp[0]  (TERM_LIFE_A/FC, sex=남, issue_age=35, term=240m, premium_term=240m, count=1)
├─ Basis (segment-level)
│   ├─ mortality_annual     -> MORTALITY_STD
│   ├─ lapse_annual         -> LAPSE_FC
│   ├─ waiver_incidence     -> WAIVER_STD
│   ├─ discount_annual      = ndarray len=1 [0.03, ..., 0.03]
│   ├─ expense_inflation    = ndarray len=1 [0.02, ..., 0.02]
│   ├─ expense_items        = tuple  (len=2)
│   │   ├─ ExpenseItem('acquisition', basis='alpha_fixed', value=80000)
│   │   └─ ExpenseItem('maintenance', basis='gamma_fixed', value=60000)
│   ├─ ra: method='confidence_level', conf=0.75
│   └─ cv: mort=0.1 morb=0.12 long=0 disab=0
├─ Coverages (rate-driven, n=5)
│   ├─ 'DEATH'           method=DEATH      risk=0  is_diagnosis=False  rate -> MORTALITY_STD
│   ├─ 'INPATIENT'       method=MORBIDITY  risk=1  is_diagnosis=False  rate -> INPATIENT_STD
│   ├─ 'CANCER'          method=DIAGNOSIS  risk=1  is_diagnosis=True   rate -> CANCER_STD
│   ├─ 'ADB'             method=DEATH      risk=0  is_diagnosis=False  rate -> ADB_STD
│   └─ 'DISEASE_DEATH'   method=DEATH      risk=0  is_diagnosis=False  rate -> DISEASE_DEATH_STD
├─ Rates (annual, evaluated for this MP)
│   ├─ axes: sex=0, issue_age=35, issue_class=0, elapsed_at_issue=0m
│   ├─         year      mort(an)     lapse(an)    waiver(an)     DEATH(an)  INPATIENT(an)    CANCER(an)       ADB(an)  DISEASE_DEATH(an)
│   ├─            0      0.000805      0.100000      0.002000      0.000805      0.030000      0.001469      0.000350      0.000345
│   ├─            1      0.000886      0.096000      0.002000      0.000886      0.030000      0.001587      0.000350      0.000354
│   ...
│   └─           19      0.004925      0.024000      0.002000      0.004925      0.030000      0.006341      0.000350      0.000516
├─ Cash flows (annual sum over 240m horizon)
│   ├─         year       premium         claim     morbidity       expense ...
│   ├─            0       719,782        61,395             0       137,691 ...
│   ├─            1       647,253        60,901             0        53,033 ...
│   ...
├─ Undiagnosed share (key months, per coverage)
│   └─ 'CANCER':
│       ├─ t=   0m: undiagnosed=1.000000
│       ├─ t=  12m: undiagnosed=0.998531
│       ...
│       └─ t= 240m: undiagnosed=0.934845
├─ Discount factors (key months)
│   ├─ t=   0m: ds=1.000000
│   ├─ t=  12m: ds=0.970874
│   ...
│   └─ t= 240m: ds=0.553676
├─ BEL roll-forward (key months)
│   ├─ BEL[t] = annuity[t] - premium[t] + (claim+morbidity+disability+expense+surrender)[t] * (1+i)^(-1/2) + BEL[t+1] * (1+i)^(-1)
│   ├─ BEL[ 240] =    2,720,414.15  (maturity seed -- a single payment at term)
│   ├─ BEL[ 228] =    2,743,895.72
│   ...
│   └─ BEL[   0] =      724,174.53
├─ CSM roll-forward (key months)
│   ├─ FCF[0]    = BEL[0] + RA[0] = 724,174.53 + 73,104.88 = 797,279.41
│   ├─ CSM[0]    = max(0, -FCF[0]) = 0.00
│   ├─ loss_comp = max(0,  FCF[0]) = 797,279.41
│   ...
└─ Final (headline numbers, per policy)
    ├─ BEL              =      724,174.53
    ├─ RA               =       73,104.88
    ├─ FCF = BEL + RA   =      797,279.41
    ├─ CSM = max(0,-FCF)=            0.00
    └─ loss_component   =      797,279.41
```

아홉 섹션이 한 화면에 다 들어옵니다. 검증 관점에서 가장 자주 보는 것:

- **Basis / Coverages** — "엔진이 내가 의도한 테이블을 잡았나?"
  `MORTALITY_STD` / `LAPSE_FC` 가 매칭. 만약 워크북에 `LAPSE_GA` 만 있는
  segment 인데 여기 `LAPSE_FC` 가 잡혔다면 segment 라우팅 오류.
- **Rates** — 첫 행의 `axes` 가 `sex=0, issue_age=35` 같이 model point 의
  실제 축. 각 연도의 rate 값이 자기 손계산 테이블의 그 셀과 일치해야 함.
- **Cash flows** — 연도별 premium / claim 합계. 첫 해 premium 이
  `premium × 12 × in-force` 와 어림셈으로 일치하는지.
- **Final** — headline 4 개. 손실부담계약이면 `CSM = 0` 이고
  `loss_component = FCF > 0`.

## 결과 해석 — 트리의 핵심 섹션

### Basis 블록의 `_fcf_table_id`

`mortality_annual -> MORTALITY_STD` 형태로 표시된 부분은 rate 함수가
**어느 워크북 시트의 어느 table_id** 에서 왔는지 알려줍니다.
`read_basis()` 가 rate 함수에 `_fcf_table_id` 메타데이터를
붙여 두기 때문입니다 — 이 라벨이 보이면 자기 입력 → 엔진 의 경로가
끊기지 않은 신호입니다.

만약 자신의 rate 를 직접 lambda 로 작성해 넣으면 `<callable>` 로 표시
됩니다. 추적은 가능하지만 어떤 table 인지 라벨이 없으므로 검토 가능성
이 줄어듭니다. 가능하면 워크북 / `read_basis()` 경로로 통일하는
편이 검증에 유리합니다.

### Coverages 블록의 risk 와 is_diagnosis

각 coverage 행의 `risk` 는 RA (Risk Adjustment = 위험조정) 계산 시 어느
변동계수 (`mortality_cv` / `morbidity_cv`) 와 묶이는지 결정합니다.
`is_diagnosis=True` 인 coverage 는 진단 시 한 번만 지급되고 in-force 의
"미진단 풀" 이 줄어드는 형태입니다 (CANCER 행). False 면 재발 가능한
형태 (INPATIENT 행 — 입원 발생).

### Rates 블록의 axes 라인

`axes: sex=0, issue_age=35, issue_class=0, elapsed_at_issue=0m` —
이 model point 의 실제 축입니다. 그 아래 표가 **그 축으로 rate 함수를
호출했을 때의 결과** 입니다. 자기 워크북의 사망률 시트에서 `(sex=0,
age=35, year=0)` 셀을 찾아 일치하는지 확인할 수 있습니다.

### BEL roll-forward 블록의 식

```
BEL[t] = annuity[t] - premium[t]
       + (claim+morbidity+disability+expense+surrender)[t] * (1+i)^(-1/2)
       + BEL[t+1] * (1+i)^(-1)
```

IFRS 17 의 backward recursion 입니다. **seed (=시작값)** 가
`BEL[term] = maturity_benefit` 이고, 거기서부터 거꾸로 한 달씩 내려옵니다.

만기환급 없는 정기보험은 `BEL[term] = 0` 으로 시작합니다. 만기금이 큰
저축성 / 단기납 종신은 큰 양수로 시작합니다 — 어떤 모양인지 한 줄로
보입니다.

### CSM 블록의 onerous 판정

`FCF[0] = BEL[0] + RA[0] = 797,279.41` 이 양수이므로 이 계약은 **손실
부담계약 (onerous contract)** 입니다. IFRS 17 §47-48 에 따라
`CSM = 0`, `loss_component = FCF`. 가입 시점에 즉시 손실 인식.

샘플 워크북의 정기보험 가격은 의도적으로 loss-making 으로 설정돼 있어
샘플 트레이스의 CSM 트랙이 거의 비어 있게 보입니다. 회사 데이터로
바꾸면 보통은 `CSM > 0` 가 정상 흐름입니다.

## 자주 쓰는 변형

### 월별 BEL 식 전개 — gmm.trace_bel_step

`gmm.trace` 가 BEL의 **궤적** 을 anchor 월에서 값으로 보여준다면,
`gmm.trace_bel_step` 은 한 달의 BEL 식을 **항별로** 풀어 보여줍니다.

```python
import fastcashflow as fcf

mp = fcf.samples.model_points()
basis = fcf.samples.basis()

fcf.gmm.trace_bel_step(0, mp, basis, months=[0, 12, 239, 240])
```

`months=` 인자로 풀어볼 월을 지정합니다. 기본은 `{0, 12, term//2,
term-1, term}` — 시작, 1년 끝, 중간, 마지막 step, seed.

출력 (`t=0` 부분):

```
├─ t=   0
│   ├─ i[t]                      = 0.002466
│   ├─ half = (1+i)^(-1/2)       = 0.998769
│   ├─ full = (1+i)^(-1)         = 0.997540
│   ├─ premium[t]                =       63,000.00
│   ├─ annuity[t]                =            0.00
│   ├─ claim[t]                  =        5,368.65
│   ├─ morbidity[t]              =            0.00
│   ├─ disability[t]             =            0.00
│   ├─ expense[t]                =       85,000.00
│   ├─ surrender[t]              =            0.00
│   ├─ mid-month sum             =       90,368.65
│   ├─ mid-month piece (×half)   =       90,257.42
│   ├─ BEL[t+1]                  =      698,635.90
│   ├─ tail piece (BEL[t+1]×full)=      696,917.12
│   ├─ recomputed BEL[t]         =      724,174.53
│   └─ engine BEL[t]             =      724,174.53  (residual +0.0000e+00)
```

**residual** (잔차 = recomputed - engine) 이 핵심입니다. 모든 step
에서 +0.0000e+00 (float64 정밀도) 이면 출력된 식과 엔진이 정확히
일치한다는 뜻 — "엔진은 이 식을 따른다" 의 증거.

손계산을 같은 월에 만들고 위 각 항과 비교하면 어느 항에서 어긋났는지
한눈에 잡힙니다. 잘 보면 좋은 항:

- `i[t]` — 월 할인율. 연 할인율 3% 면 `(1.03)^(1/12) - 1 = 0.002466`
- `premium[t]` — `premium × in-force` 와 어림셈으로 일치해야
- `claim[t]` — `coverage_amount × in-force × mortality_monthly` 정도

`t = term` (시드) 행은 `maturity_benefit` 만 표시하고 recursion 식은
없습니다 (그 아래 월이 없기 때문).

### 월별 CSM 식 전개 — gmm.trace_csm_step

CSM은 BEL의 반대 방향 — **forward recursion** 입니다. 시작값
`csm[0] = max(0, -(BEL+RA))` 에서 출발해 매월 이자부리 + coverage_unit
환입.

```python
fcf.gmm.trace_csm_step(0, mp, basis, months=[1, 60, 120, 240])
```

샘플의 정기보험은 손실부담이라 모든 step 에서 `csm = 0`. 의미는
"floor 가 작동" — 차라리 가입 시 CSM이 음수가 될 것을 0 으로 막은 것.
출력의 `Seed (t = 0)` 블록이 명시적으로 알려줍니다:

```
├─ Seed (t = 0)
│   ├─ BEL[0]               =      724,174.53
│   ├─ RA[0]                =       73,104.88
│   ├─ FCF[0] = BEL + RA    =      797,279.41
│   ├─ csm[0] = max(0,-FCF) =            0.00
│   └─ onerous contract -- csm = 0 throughout; release/accretion are 0 by construction.
```

수익성 있는 계약 (BEL < 0) 으로 바꿔 보면 매월 recursion 이 의미를
가집니다:

```python
import numpy as np
from fastcashflow.basis import Basis
from fastcashflow.modelpoints import ModelPoints

# 사망률 함수 -- 연 0.05% 의 평탄 사망률 (보험금 대비 매우 낮은 율)
def death_fn(s, ia, d, ic, em):
    return np.full(d.shape, 0.0005)

# 해지율 함수 -- 연 2% 의 평탄 해지율
def lapse_fn(s, ia, d, ic, em):
    return np.full(d.shape, 0.02)

# 산출기초 -- 이익이 나는 (CSM > 0) 시나리오
profitable = Basis(
    mortality_annual = death_fn,                                  # 보유계약 감쇠용 사망률 (연 0.05%)
    lapse_annual     = lapse_fn,                                  # 해지율 (연 2%)
    discount_annual  = 0.03,                                      # 연 할인율 3%
    ra_confidence    = 0.75,                                      # 위험조정 신뢰수준 75%
    mortality_cv     = 0.05,                                      # 사망률 변동계수 5%
    coverages        = (fcf.CoverageRate("DEATH", death_fn),),    # 사망 보장 1 종 (청구 rate = death_fn)
)

# 모델 포인트 -- 보험금 1 억, 월납 보험료 20 만, 5 년 만기 한 계약
mp_one = ModelPoints(
    issue_age        = np.array([40.0]),                           # 가입연령 40 세
    premium    = np.array([200_000.0]),                      # 월납 보험료 20 만
    term_months      = np.array([60]),                             # 보험기간 60 개월 (5 년)
    benefits         = {0: np.array([100_000_000.0])},             # 0 번 보장 (= DEATH) 의 보험금 1 억
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},  # 코드 → 산출방식 매핑
)

fcf.gmm.trace_csm_step(0, mp_one, profitable, months=[1, 30, 60])
```

마지막 step (`t = 60`) 의 행:

```
└─ t=  60
    ├─ csm[t-1]                  =      190,495.34
    ├─ accretion[t-1] = csm*i    =          469.81
    ├─ accreted = csm + acc      =      190,965.16
    ├─ coverage_units[t-1]       = 0.903220
    ├─ cu_tail[t-1] = sum(cu[t-1:]) = 0.903220
    ├─ release fraction          = ... = 1.000000
    ├─ release[t-1] = accreted * frac =      190,965.16
    ├─ recomputed csm[t]         =            0.00
    └─ engine csm[t]             =            0.00  (residual +0.0000e+00)
```

마지막 월에서 `release fraction = 1.0` (잔존 coverage_unit 전부 환입)
이라 `csm[60] = 0`. 보장 단위의 경계조건이 출력에 그대로 보입니다.

### 가정 변경의 전파 추적 — gmm.trace_diff

"사망률 +10% 면 BEL이 얼마나 움직이나" 는 시나리오 / 민감도의 가장
기본적인 질문입니다. 결과값만 보지 말고 **어디서 어떻게** 전파됐는지
한 화면에 봅니다.

```python
import fastcashflow as fcf
from dataclasses import replace

mp = fcf.samples.model_points()
basis = fcf.samples.basis()
baseline = basis[('TERM_LIFE_A', 'FC')]

# mortality x 1.10 shock — rate 함수를 wrap
def shock(rate_fn, factor):
    def wrapped(sex, issue_age, duration, issue_class, elapsed):
        return rate_fn(sex, issue_age, duration, issue_class, elapsed) * factor
    wrapped._fcf_table_id = getattr(rate_fn, '_fcf_table_id', None)
    wrapped._fcf_modifiers = getattr(rate_fn, '_fcf_modifiers', ()) + (f'x{factor}',)
    return wrapped

# 사망률 테이블 +10% -- 같은 MORTALITY_STD 가 in-force 감쇠(mortality_annual)
# 와 사망보장 claim(DEATH 담보 rate) 양쪽을 굴리므로 두 자리를 함께 shock
new_coverages = tuple(
    replace(c, rate=shock(c.rate, 1.10)) if c.code == "DEATH" else c
    for c in baseline.coverages
)
shocked = replace(baseline,
                  mortality_annual = shock(baseline.mortality_annual, 1.10),
                  coverages        = new_coverages)

fcf.gmm.trace_diff(0, mp, baseline, shocked,
                    label_a='baseline', label_b='mort+10%')
```

출력 (Final 블록):

```
└─ Final (headline change, per policy)
    ├─ BEL                  724,174.53  ->      827,176.05   (  +103,001.52,   +14.22%)
    ├─ RA                    73,104.88  ->       80,292.70   (    +7,187.83,    +9.83%)
    ├─ FCF = BEL+RA         797,279.41  ->      907,468.75   (  +110,189.34,   +13.82%)
    ├─ CSM = max(0,-FCF)          0.00  ->            0.00   (        +0.00,        --)
    └─ loss_component       797,279.41  ->      907,468.75   (  +110,189.34,   +13.82%)
```

BEL이 +14.22% 움직였습니다. 이 14% 가 어디서 왔는지 위쪽 섹션이
설명해 줍니다:

- **Rate deltas** — 매년 mortality(annual) 와 DEATH 담보 rate 가 (둘 다
  같은 `MORTALITY_STD`) 정확히 +10.00%. lapse / waiver / 나머지 담보는
  변화 없음 (출력에서 자동 숨김).
- **Cash flow deltas** — claim 이 매년 +10% 근처. 동시에 premium 이
  소폭 감소 (-0.05% 정도) — 사망률이 올라가니 in-force 가 빨리 줄어
  미래 premium 이 적어지는 자연스러운 전파.
- **BEL deltas (key months)** — `BEL[0]` 의 +14.22% 가 `BEL[240]` 의
  -0.46% (만기금이 줄어든 효과) 와 합쳐져 만들어진 결과.

`+14.22%` 라는 숫자 하나가 아니라 **각 단계가 어떻게 얽혔는지** 가
보이는 게 핵심입니다. 사내 검토 시 "왜 BEL이 14% 움직였는지" 의
질문에 일직선으로 답할 수 있습니다.

## 함정 — 검증 시 자주 마주치는 것

### 함정 1 — `residual` 이 0 인데 손계산과 안 맞음

`gmm.trace_bel_step` 의 잔차는 **엔진의 식 vs 엔진 자신** 의 비교입니다.
0 이라는 것은 엔진이 출력된 식을 정확히 따른다는 뜻일 뿐, **엔진과
사용자 손계산** 이 일치한다는 뜻은 아닙니다.

손계산과 엔진이 어긋나면 보통 다음 셋 중 하나입니다:

1. **사용자의 손계산이 다른 식** — 예를 들어 IFRS 17 의 mid-month
   할인 (`(1+i)^(-1/2)`) 대신 month-start 할인을 가정 ([§B71](https://www.ifrs.org)
   의 한 해석). 식이 다르면 결과도 다름. step 행의 `half = ...` /
   `full = ...` 자리에 자기 손계산의 할인을 대입해 보면 어느 쪽 정의를
   썼는지 분명해집니다.
2. **rate 환산 차이** — 자기 손계산은 월 사망률 `q_m = q_a / 12`, 엔진은
   `1 - (1 - q_a)^(1/12)` (constant-force = 사력 일정 가정). 두 값은
   작은 q_a 에서는 근사 같지만 q_a 가 크면 (10% 이상) 차이 명확.
3. **A/E factor / improvement / age_shift 누락** — 엔진은 적용했는데
   손계산은 raw rate 만 썼을 가능성. `gmm.trace` 의 Basis 행에
   `<callable -> MORTALITY_STD (+improvement, +ae)>` 같은 modifier 가
   붙어 있으면 wrap 이 걸린 것.

### 함정 2 — 자기 손계산 단순화의 영향 잊기

검증을 단순화하려고 `lapse = 0`, `expense = 0`, `discount = 0`, `RA cv
= 0` 으로 두면 손계산이 손쉽지만, **그러면 엔진도 같은 단순화로 돌려야**
공정한 비교입니다. 챕터 1 의 1.6 검증 절이 정확히 그 패턴 — 2개월,
사망률 1%, 그 외 전부 0.

`gmm.trace` 의 Basis 블록은 검증에 필수: 자신이 단순화한 항목이
실제로 0 인지 (예: `expense_items` 가 비어 있는지) 확인합니다.

### 함정 3 — shock 이 의도와 다르게 전파됨

`gmm.trace_diff` 에서 mortality 만 올렸는데 lapse / expense / surrender
가 미세하게 움직이는 것은 **버그가 아닙니다**. mortality 가 in-force
trajectory 를 바꾸고, 그 in-force 가 lapse_flow / expense / surrender
의 베이스를 바꾸는 자연스러운 전파입니다. 변화 폭이 mortality 의 변화
폭과 격이 다르면 (mortality +10% 인데 expense +50%) 그때 의심.

### 함정 4 — `gmm.trace*` 가 무거울 거라 생각해 안 씀

이 도구들 모두 **단일 행 subset 후 measure()** 만 호출합니다. 1M 계약
포트폴리오에서 `gmm.trace(0, mp, basis)` 를 호출해도 단일 계약 측정
비용만 발생합니다. 검토 회의 중에 화면 공유로 즉석 호출해도 부담 없는
수준입니다.

### 함정 5 — `mp_id` 가 문자열일 거라고 기대

도구의 첫 인자는 `model_points` 의 **0-based 정수 인덱스** 입니다.
워크북 / CSV 의 `mp_id` 컬럼 (P001, P002, ...) 과 다릅니다. 문자열 ID
로 찾고 싶다면 폴리어스 / 판다스로 인덱스를 미리 뽑아 둡니다:

```python
import fastcashflow as fcf
import polars as pl

# 샘플 파일 저장 (본인 파일 있으면 생략)
fcf.samples.export("samples", template="gmm")   # basis.xlsx + policies / coverages / calculation_methods (+ inforce)

# 만들어진 샘플 파일 읽어 들이기
basis = fcf.read_basis("samples/basis.xlsx")                  # 산출기초
mp = fcf.read_model_points(
    "samples/policies.csv",                                   # 계약 스펙
    coverages           = "samples/coverages.csv",            # 담보 가입금액
    calculation_methods = "samples/calculation_methods.csv",  # 담보별 산출방식
)

# mp_id (문자열) → 0-based 정수 인덱스
pol = pl.read_csv("samples/policies.csv")
idx = pol.with_row_index("idx").filter(pl.col("mp_id") == "P002")["idx"][0]
fcf.gmm.trace(int(idx), mp, basis)
```

## 인접 레시피

이 챕터를 읽고 자연스럽게 갈 다음 자리들:

- [8.1 시나리오 / 민감도 분석](sensitivity) — `gmm.trace_diff` 의 활용을
  단일 계약 검증 너머 포트폴리오 단위 민감도로 확장.
- [정기보험](../simple/term-life) 의 "함정 — 흔한 실수와 잡는 방법"
  절 — 2 개월 손계산 패턴. 이 챕터의 도구로 그 검증을 다시 추적하면
  식이 어떻게 매핑되는지 명확.
- [7.2 워크북 — 다중 segment / 다종 상품](../io/workbook-multi) — `gmm.trace` 의 segment
  라우팅이 맞는지 확인하는 자리.

기본 튜토리얼의 5장 (BEL 계산) / 7장 (CSM 계산) 이 `gmm.trace_bel_step` /
`gmm.trace_csm_step` 이 풀어 보여주는 식의 derivation (유도) 을 다룹니다.

```{admonition} 검증으로 무엇이 보장되고 무엇이 안 보장되나
:class: warning

이 챕터의 도구들은 **엔진이 출력된 식을 정확히 따른다** 는 것을
보여줍니다 (residual ≈ 0). 식 자체가 회사 상품 / 회사 가정 / IFRS 17
의 의도에 맞는지는 **사용자 판단** 입니다.

검증으로 잡히는 것: 입력 / 출력 / 식의 매핑이 끊기지 않았다는 사실.

검증으로 안 잡히는 것: 입력 가정이 옳은지, 식의 선택이 회사 상황에
적합한지, IFRS 17 의 §B71 mid-month 가정이 회사 회계정책과 정합한지.
이 판단은 도구 너머의 영역입니다.
```

# 6.1 비례 재보험 (quota share)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **비례 재보험 (quota share)** 보유계약 한 건을 fastcashflow 로 측정하는 법
- 원수계약과 재보험을 동시에 풀어, 재보험 BEL이 직접 부채의 "거울" 이
  되는 것을 손계산으로 확인 (둘 다 `-19.90`)
- 보유 재보험계약의 IFRS 17 두 가지 수정 — **전가위험 (RA)** 과
  **순원가 / 이익 (CSM)**, 그리고 손실요소가 없는 이유
- 출재비율 (cession) 을 바꿀 때 결과가 선형으로 움직이는 것
- 포트폴리오에 적용할 때 토이의 거울 관계가 깨지는 이유 (사업비는 출재 안 됨)

이 챕터만 봐도 비례 재보험 측정은 끝까지 갈 수 있도록 만들었습니다.
```

## 상품 소개 — 비례 재보험 (출재)

**재보험 (reinsurance)** 은 보험사 (원수사, cedant) 가 자신이 인수한
위험의 일부를 다른 보험사 (재보험사) 에게 넘기는 계약입니다. 위험을
넘기는 것을 **출재 (出再)** 라 합니다.

**비례 재보험 (proportional reinsurance)** 중 가장 단순한 형태가
**quota share (비례출재)** 입니다 — 원수사가 정한 **출재비율 (cession)**
만큼을 보험금과 보험료 양쪽에 똑같이 적용합니다. 예를 들어 출재비율이
50% 이면:

- 원수사는 보험금의 **50% 를 재보험사로부터 회수** (recovery) 하고,
- 그 대가로 받은 보험료의 **50% 를 재보험료로 재보험사에 지급** 합니다.

본 챕터는 한 원수 포트폴리오 위에 **단일 quota-share 출재** 를 얹은
가장 단순한 구조를 다룹니다. 출재수수료 (ceding commission) / 비비례
재보험 (초과손해액 XL, surplus) / 재보험사 부도위험 / 손실회수요소는
다루지 않습니다 (→ 비비례는 [확장 로드맵](../design/dynamic-lapse) 의
인접 주제).

## 모델링 매핑 — `reinsurance.measure` 와 `QuotaShare`

원수계약은 `fcf.gmm.measure` 로 측정했습니다. 보유 재보험계약은 전용
진입점 `fcf.reinsurance.measure` 를 씁니다 — **원수 포트폴리오 + 산출기초
+ treaty (재보험 약정)** 세 가지를 받습니다.

```{list-table}
:header-rows: 1
:widths: 34 66

* - 상품의 mechanic
  - fastcashflow 의 표현
* - 무엇을 어떻게 출재하는가
  - `treaty` — 예: `fcf.reinsurance.QuotaShare(cession=0.50)`
* - 출재된 보험금의 회수
  - `recovery` = 출재 사망 + 출재 morbidity (재보험사가 갚는 돈)
* - 지급하는 재보험료
  - `reinsurance_premium` = `cession` × 원수 보험료
* - 보유 재보험계약 측정
  - `fcf.reinsurance.measure(mp, basis, treaty)` → `ReinsuranceMeasurement`
```

IFRS 17 은 보유 재보험계약을 일반모형 (GMM) 으로 측정하되 **두 가지를
수정** 합니다 (Sec. 60-70):

- **RA 는 전가위험** (Sec. 64) — 원수계약의 RA가 원수사가 보유 한
  불확실성이라면, 재보험의 RA는 재보험사로 넘긴 위험의 마진입니다.
- **CSM 은 미실현이익이 아니라 커버의 순원가 / 이익** (Sec. 65) —
  재보험을 사는 것이 순비용이면 그만큼을 즉시 비용처리하지 않고
  **이연 (CSM이 음수)** 해 커버기간에 걸쳐 상각하고, 순이익이면
  CSM이 양수입니다. **손실요소 (loss component) 는 없습니다.**

부호 규약은 원수계약과 같습니다 (부채 관점, 유출 양수):

- **재보험료** — 원수사가 내는 돈 (유출), 재보험 부채를 **증가**
- **회수금** — 원수사가 받는 돈 (유입), 재보험 부채를 **감소**
- 따라서 `BEL = PV(재보험료) - PV(회수금)`. 양수면 순원가, 음수면 순이익.

## 최소 작동 예제 — 손계산과 엔진 한 번에

[2.1 정기보험](../simple/term-life) 의 두 달짜리 토이 계약을 그대로
재사용합니다 — 손계산이 그대로 잡히는 작은 예입니다. 차이는 **할인율을
0 으로** 둔 것뿐입니다 (할인을 빼면 재보험이 직접 부채의 거울이 되는
관계가 깔끔하게 보입니다).

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 2개월, 해지 없음
- 월 사망률 1%, 사망보험금 12,000, 월 보험료 100
- **할인율 0** (거울 관계를 깨끗하게 보기 위함)
- 출재비율 (cession) 50% — 보험금·보험료의 절반을 출재
```

먼저 원수계약의 손계산입니다. 월중 사망보험금, 월초 보험료:

| t | 보유계약 | 사망보험금 (월중) | 보험료 (월초) |
|---|---|---|---|
| 0 | 1.0000 | 1.00 × 1% × 12,000 = 120.00 | 100.00 |
| 1 | 0.9900 | 0.99 × 1% × 12,000 = 118.80 | 99.00 |

- 직접 BEL = (120.00 + 118.80) − (100.00 + 99.00) = 238.80 − 199.00 = **39.80**

이제 50% 출재 — 보험금·보험료 양쪽에 0.5 를 곱합니다:

| t | 회수금 (월중, = 0.5 × 보험금) | 재보험료 (월초, = 0.5 × 보험료) |
|---|---|---|
| 0 | 0.5 × 120.00 = 60.00 | 0.5 × 100.00 = 50.00 |
| 1 | 0.5 × 118.80 = 59.40 | 0.5 × 99.00 = 49.50 |

- PV(재보험료) = 50.00 + 49.50 = 99.50
- PV(회수금) = 60.00 + 59.40 = 119.40
- **BEL = 99.50 − 119.40 = −19.90** (= −0.5 × 39.80, 직접 부채의 거울)
- **RA** = z(0.75) × 사망률CV × PV(출재 사망) = 0.67449 × 0.10 × 119.40 = **8.05**
- **CSM** = −(BEL − RA) = −(−19.90 − 8.05) = **27.95** (순이익)

코드:

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_rate = 1 - (1 - 0.01) ** 12

# 해지율 함수 -- 해지 없음
lapse_rate = 0.0

# 원수 모델 포인트 (계약 하나)
mp = fcf.ModelPoints.single(
    issue_age   = 40,           # 가입연령 40세
    sex         = 0,            # 성별 (0=남, 1=여)
    benefits    = {0: 12_000},  # 0번 보장 (= DEATH) 의 보험금 12,000
    premium     = 100,          # 월납 보험료 100
    term_months = 2,            # 보험기간 2개월
)

# 산출기초
basis = fcf.Basis(
    mortality_annual = death_rate,  # 보유계약 사망률 (위 death_rate)
    lapse_annual     = lapse_rate,  # 해지율 (해지 없음)
    discount_annual  = 0.0,         # 할인율 0 (거울 관계를 깨끗하게)
    ra_confidence    = 0.75,        # 위험조정 신뢰수준 75%
    mortality_cv     = 0.10,        # 사망률 변동계수 10%
    coverages        = (
        fcf.CoverageRate("DEATH", death_rate),  # 사망 보장 1종 (청구 rate = death_rate)
    ),
)

# 재보험 약정 -- 50% quota share
treaty = fcf.reinsurance.QuotaShare(cession=0.50)  # 출재비율 50%

# 보유 재보험계약 측정
r = fcf.reinsurance.measure(mp, basis, treaty)
print(f"BEL = {r.bel[0]:.2f}")               # PV(재보험료) - PV(회수금)
print(f"RA  = {r.ra[0]:.2f}")                # 전가위험
print(f"CSM = {r.csm[0]:.2f}")               # 순원가(-) / 순이익(+)
print(f"recovery            = {np.round(r.recovery[0], 2)}")             # 월별 회수금
print(f"reinsurance_premium = {np.round(r.reinsurance_premium[0], 2)}")  # 월별 재보험료
```

출력:

```
BEL = -19.90
RA  = 8.05
CSM = 27.95
recovery            = [60.  59.4]
reinsurance_premium = [50.  49.5]
```

손계산 −19.90 과 엔진 −19.90 이 정확히 일치합니다. `recovery` 와
`reinsurance_premium` 의 월별 값도 손계산 표 그대로입니다.

## 결과 읽기 — BEL / RA / CSM

### BEL = −19.90 — 순이익 커버

재보험 BEL이 **음수** 라는 것은 `PV(회수금) > PV(재보험료)` — 즉
재보험사로부터 받을 것이 낼 것보다 크다는 뜻입니다. 원수계약이 손실
계약 (onerous) 이라 사망보험금이 보험료보다 컸고, 그 절반을 출재했으니
재보험은 원수사 입장에서 **순자산** 이 됩니다.

거울 관계가 핵심입니다 — 할인 0, 사업비 0 인 이 토이에서는
**재보험 BEL = −cession × 직접 BEL** 이 정확히 성립합니다
(−0.5 × 39.80 = −19.90). "절반을 출재하면 직접 순부채의 절반이
거울처럼 반대편에 선다" 는 직관.

### RA = 8.05 — 전가위험 (Sec. 64)

재보험 RA는 **재보험사로 넘긴 위험의 마진** 입니다. 원수 RA가 출재
부분만큼 재보험사로 이전된 것 — 이 토이에서는 `재보험 RA =
cession × 직접 RA` (0.5 × 16.11 = 8.05) 가 성립합니다. 즉 직접계약의
불확실성 중 출재비율만큼이 그대로 전가위험이 됩니다.

### CSM = +27.95, 손실요소 없음 (Sec. 65)

재보험 CSM은 **커버를 사는 것의 순원가 또는 순이익** 입니다 —
미실현이익 (원수 CSM의 의미) 이 아닙니다.

- `CSM = -(BEL - RA)`. BEL이 음수 (순이익 커버) → CSM 양수
- BEL이 양수 (순원가 커버) → **CSM 음수** — 순비용을 즉시 비용처리하지
  않고 이연해 커버기간에 상각
- 어느 쪽이든 **손실요소 (loss component) 가 없습니다** — 원수계약과
  달리 재보험에는 onerous → 즉시손실 인식 메커니즘이 없습니다.

이 토이는 순이익 커버라 `CSM = +27.95`. 음수 CSM은 아래 포트폴리오
예제에서 나타납니다.

## 자주 쓰는 변형

### 출재비율 (cession) 바꾸기 — 선형

quota share 는 보험금·보험료에 같은 비율을 곱하므로 BEL / RA / CSM이
모두 cession 에 **선형** 입니다:

```python
for c in (0.25, 0.50, 0.75, 1.00):
    rr = fcf.reinsurance.measure(mp, basis, fcf.reinsurance.QuotaShare(cession=c))
    print(f"cession={c:.2f}  BEL={rr.bel[0]:>8.4f}  RA={rr.ra[0]:>7.4f}  CSM={rr.csm[0]:>8.4f}")
```

출력:

```
cession=0.25  BEL= -9.9500  RA= 4.0267  CSM= 13.9767
cession=0.50  BEL=-19.9000  RA= 8.0534  CSM= 27.9534
cession=0.75  BEL=-29.8500  RA=12.0801  CSM= 41.9301
cession=1.00  BEL=-39.8000  RA=16.1068  CSM= 55.9068
```

`cession=1.00` (전부 출재) 이면 BEL = −39.80, RA = 16.11 로 직접계약의
거울이 정확히 됩니다. `cession=0` 이면 출재가 없으니 세 값 모두 0.

### 포트폴리오에 적용 — 그리고 거울이 깨지는 이유

샘플 포트폴리오의 한 segment 에 50% 출재를 얹어 봅니다. 원수 측정과
나란히:

```python
import numpy as np
import tempfile
from pathlib import Path
import fastcashflow as fcf

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    fcf.samples.export(tmp, template="gmm", quiet=True)        # basis.xlsx + 데이터 파일들

    port_basis = fcf.read_basis(tmp / "basis.xlsx")                                         # segment 가정 사전
    port_mp = fcf.read_model_points(tmp / "policies.csv", coverages=tmp / "coverages.csv",
                                    calculation_methods=tmp / "calculation_methods.csv")

    b      = port_basis.resolve(("TERM_LIFE_A", "GA"))                                              # 한 segment 의 가정
    direct = fcf.gmm.measure(port_mp, b, full=False)                                        # 원수 측정 (headline)
    reins  = fcf.reinsurance.measure(port_mp, b, fcf.reinsurance.QuotaShare(cession=0.50))

    print(f"direct  BEL={direct.bel.sum():>14,.0f}  RA={direct.ra.sum():>9,.0f}  CSM={direct.csm.sum():>14,.0f}")
    print(f"reins   BEL={reins.bel.sum():>14,.0f}  RA={reins.ra.sum():>9,.0f}  CSM={reins.csm.sum():>14,.0f}")
```

출력:

```
direct  BEL=   -26,441,301  RA=1,210,877  CSM=    27,593,495
reins   BEL=    29,205,831  RA=  605,439  CSM=   -28,600,392
```

여기서는 토이의 거울 관계가 **깨집니다** — 직접 BEL은 음수
(-26.4M, 이익계약) 인데 재보험 BEL은 양수 (+29.2M, 순원가) 이고 CSM은
음수입니다. 이유는 두 가지:

- **사업비는 출재되지 않습니다.** 직접 측정은 사업비까지 보지만, 재보험은
  사망보험금과 보험료만 출재합니다. 이 segment 는 보험료가 사망보험금보다
  커서, 출재하면 **재보험료 (지급) > 회수금** — 즉 순원가 커버 (BEL 양수,
  CSM 음수). 원수계약이 이익이라고 해서 재보험이 그 부호를 따라가지 않습니다.
- **할인 timing.** 재보험료는 월초 (`discount_bom`), 회수금은 월중
  (`discount_mid`) 으로 할인되어 토이의 할인 0 가정과 달라집니다.

즉 **재보험 측정은 출재된 보험금·보험료만 보지, 원수계약의 사업비나
onerous 여부를 그대로 따르지 않습니다.** 거울 관계는 할인 0 · 사업비 0
인 토이의 산물입니다.

```{admonition} BasisRouter 라우팅은 measure 쪽만
:class: note

`fcf.gmm.measure(mp, basis_dict, full=False)` 처럼 세그먼트 BasisRouter 를 통째로
넘기는 라우팅은 원수 측정 (`gmm.measure`) 의 기능입니다.
`reinsurance.measure` 는 단일 `Basis` 를 받으므로 위 예제처럼
`basis.resolve(("TERM_LIFE_A", "GA"))` 로 한 segment 를 골라 넘깁니다.
```

## 함정 — 흔한 실수와 잡는 방법

### 함정 1 — `cession` 범위

`cession` 은 `[0, 1]` 의 비율입니다. 범위를 벗어나거나 비유한 (NaN /
inf) 값이면 `QuotaShare(...)` 생성 시점에 바로 `ValueError` 로
막힙니다 — `measure` 깊은 곳에서 cryptic 에러로 터지지 않습니다.

```python
# ✗ 아래는 생성 즉시 ValueError (실행하지 마세요):
#     fcf.reinsurance.QuotaShare(cession=1.5)    # [0, 1] 벗어남
#     fcf.reinsurance.QuotaShare(cession=float("nan"))
```

### 함정 2 — 거울 항등식으로 검산

할인 0 · 사업비 0 인 토이라면 **재보험 BEL = −cession × 직접 BEL**,
**재보험 RA = cession × 직접 RA** 가 정확히 성립해야 합니다. 한 줄로
확인:

```python
d = fcf.gmm.measure(mp, basis)
r = fcf.reinsurance.measure(mp, basis, fcf.reinsurance.QuotaShare(cession=0.50))
assert abs(r.bel[0] - (-0.50 * d.bel[0])) < 1e-9    # BEL 거울
assert abs(r.ra[0]  - ( 0.50 * d.ra[0])) < 1e-9     # RA 비례
print("거울 항등식 OK")
```

출력:

```
거울 항등식 OK
```

이 항등식이 깨지면 (할인이나 사업비를 끄지 않았거나, rate 를 한쪽만
바꿨거나) 어디서 어긋났는지 추적할 신호입니다.

### 함정 3 — 음수 CSM을 손실로 오해

재보험 CSM이 음수인 것은 **정상** 입니다 — 커버의 순원가를 이연한
것이지 손실이 아닙니다. 재보험에는 손실요소가 없으므로
`ReinsuranceMeasurement` 에 `loss_component` 필드도 없습니다. 원수계약의
음수 BEL = 이익 계약이듯, 재보험의 부호는 원수와 별개로 읽어야 합니다.

| 재보험 BEL | 재보험 CSM | 의미 |
|---|---|---|
| 음수 | 양수 | 회수금 > 재보험료 — 순이익 커버 |
| 양수 | 음수 | 재보험료 > 회수금 — 순원가 커버 (이연·상각) |

## 인접 레시피

- [2.1 정기보험](../simple/term-life) — 본 챕터가 출재한 **원수계약** 의
  측정. 같은 토이를 `gmm.measure` 로 직접 측정. 재보험을 읽기 전 출발점.
- [9.1 결산 / 보유계약 평가](../workflow/settlement) — 보유 재보험계약은
  분기말 결산에서 원수계약과 **함께** 측정됩니다. 결산 워크플로의 자리.
- [1.4 담보별 산출방법](../basics/coverage-mechanics) — 출재 대상이
  되는 사망 (claim) / morbidity 현금흐름이 엔진 안에서 어떻게 만들어지는지.
- [확장 로드맵](../design/dynamic-lapse) 의 인접 주제 — 비비례 재보험
  (초과손해액 XL / surplus), 출재수수료, 재보험사 부도위험은 v1 범위 밖.

전체 챕터 라인업은 [쿡북 인덱스](../index) 참조.

```{admonition} 가정의 정확성과 결과의 의미
:class: warning

본 챕터의 BEL / RA / CSM 숫자는 **토이·샘플 가정 그대로** 의 결과입니다.
실제 재보험 평가에서는 출재비율 · 사망률CV · 할인율을 회사 약정과
산출기초에 맞춰야 의미 있는 숫자가 나옵니다. fastcashflow 의 결과는
"입력 가정에 충실한 산출" 이지, 회사 포트폴리오의 진실값은 아닙니다.
```

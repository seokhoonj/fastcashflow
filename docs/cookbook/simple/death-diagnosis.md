# 2.2 사망 + 단순 진단 일시금

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 정기보험 (2.1) 에 **진단 일시금 담보** 를 하나 더 얹는 가장 작은 확장
- 담보가 둘이 될 때 바뀌는 세 자리 — `benefits` / `coverages` /
  `calculation_methods`
- 사망 (`DEATH`) 과 진단 (`DIAGNOSIS`) 이 같은 계약 안에서 *다른 방식* 으로
  계산되는 이유 (1.4 메커니즘의 실전 적용)
- 진단율은 `mortality_annual` 에 **넣지 않는다** — 진단이 보유계약을
  줄이지 않는 이유

면책 / 감액 기간 없는 *단순* 결합만 다룹니다. 90 일 면책 같은 룰은
3.2 챕터에서.
```

## 상품 소개 — 사망 + 진단 결합

한국에서 가장 흔한 결합 중 하나는 **사망 보장에 진단 일시금을 얹은**
구조입니다. 피보험자가 사망하면 사망보험금을, 살아있는 동안 암 등으로
진단받으면 진단 일시금을 따로 지급합니다. 두 사건은 서로 독립적으로
일어나고 (진단받아도 계약은 유지, 보험료도 계속 납입), 각자 별도의
보험금을 가집니다.

2.1 의 정기보험이 사망 한 가지 담보만 가졌다면, 본 챕터는 거기에
**암 진단 담보** 하나를 추가합니다. 담보가 둘이 되면서 바뀌는 자리를
짚는 것이 핵심입니다.

## 모델링 매핑 — 두 번째 담보를 얹는다

담보가 하나에서 둘로 늘면 세 자리가 함께 바뀝니다:

```{list-table}
:header-rows: 1
:widths: 30 70

* - 자리
  - 무엇이 바뀌나
* - `Assumptions.coverages`
  - `CoverageRate` 가 둘 — `("DEATH", death_fn)` 에 `("CANCER", cancer_fn)` 추가
* - `ModelPoints.benefits`
  - `{0: 사망보험금, 1: 진단금}` — 정수 키가 `coverages` 의 순서 (0 = 첫째, 1 = 둘째)
* - `ModelPoints.calculation_methods`
  - `{"DEATH": DEATH, "CANCER": DIAGNOSIS}` — 각 담보가 어느 계산방식인지
```

**진단율 `cancer_fn` 은 `mortality_annual` 에 넣지 않습니다.** 사망은
보유계약을 줄이지만 (죽으면 더 이상 보장 진행 X), 암 진단은 사람을
보유계약에서 빼지 않습니다 — 진단받아도 살아있고 보험료도 계속 냅니다.
그래서 `mortality_annual = death_fn` 한 가지뿐이고, `cancer_fn` 은
`coverages` 의 CANCER 한 자리에만 들어갑니다 (1.3 의 두 역할 참조).

## 한 계약 — 손계산과 엔진

2.1 과 같은 두 달짜리 toy 계약에 암 진단 담보를 추가합니다.

```{admonition} 예제 설정
:class: note

- 가입연령 40세, 보험기간 2개월
- 월 사망률 1%, 월 암 진단율 0.5%, 해지 없음
- 사망보험금 12,000, 암 진단금 20,000, 월 보험료 100
- 월 할인율 0.5%
```

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 1% 의 연 환산 (보유계약 감쇠 + 사망 보장 청구 양쪽)
death_fn  = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.01) ** 12)
# 암 진단율 함수 -- 월 0.5% 의 연 환산 (암 보장 청구 전용)
cancer_fn = lambda s, a, d: np.full(a.shape, 1 - (1 - 0.005) ** 12)
# 해지율 함수 -- 해지 없음
lapse_fn  = lambda s, a, d: np.full(d.shape, 0.0)

# 계리적 가정
asmp = fcf.Assumptions(
    mortality_annual = death_fn,                              # 보유계약 감쇠용 사망률 (death_fn 만)
    lapse_annual     = lapse_fn,                              # 해지율 (해지 없음)
    discount_annual  = 1.005 ** 12 - 1,                       # 연 할인율 (월 0.5% 의 연 환산)
    ra_confidence    = 0.75,                                  # 위험조정 신뢰수준 75%
    mortality_cv     = 0.10,                                  # 사망률 변동계수 10%
    morbidity_cv     = 0.12,                                  # 진단율 변동계수 12%
    coverages        = (fcf.CoverageRate("DEATH",  death_fn),   # 0 번 담보 — 사망 청구율
                        fcf.CoverageRate("CANCER", cancer_fn)), # 1 번 담보 — 암 진단율
)

# 모델 포인트 (계약 하나, 담보 둘)
mp = fcf.ModelPoints.single(
    issue_age     = 40,                                       # 가입연령 40 세
    benefits      = {0: 12_000, 1: 20_000},                   # 0 = 사망보험금 12,000, 1 = 암 진단금 20,000
    level_premium = 100,                                      # 월납 보험료 100
    term_months   = 2,                                        # 보험기간 2 개월
    calculation_methods = {"DEATH":  fcf.CalculationMethod.DEATH,       # 사망 = 사망형 계산
                           "CANCER": fcf.CalculationMethod.DIAGNOSIS},  # 암 = 진단형 (미진단 풀)
)

m = fcf.measure(mp, asmp)
print(f"BEL  = {m.bel[0, 0]:.2f}")          # 최선추정부채
print(f"RA   = {m.ra[0, 0]:.2f}")           # 위험조정
print(f"CSM  = {m.csm[0, 0]:.2f}")          # 보험계약마진
print(f"Loss = {m.loss_component[0]:.2f}")  # 손실요소
```

출력:

```
BEL  = 236.63
RA   = 32.01
CSM  = 0.00
Loss = 268.64
```

손계산으로 BEL 을 두 담보로 나눠 확인합니다 (월 할인 0.5%, 보험료는
월초 · 청구는 월중):

| t | 보유계약 | 사망 청구 (월중) | 미진단 풀 | 암 진단 청구 (월중) | 보험료 (월초) |
|---|---|---|---|---|---|
| 0 | 1.0000 | 1.00 × 1% × 12,000 = 120.00 | 1.0000 | 1.00 × 0.5% × 20,000 = 100.00 | 100.00 |
| 1 | 0.9900 | 0.99 × 1% × 12,000 = 118.80 | 0.9851 | 0.9851 × 0.5% × 20,000 = 98.51 | 99.00 |

- PV(사망) = 120 × 0.99751 + 118.80 × 0.99256 = **237.62**
- PV(암 진단) = 100 × 0.99751 + 98.51 × 0.99256 = **197.52**
- PV(보험료) = 100 × 1.0 + 99 × 0.99502 = **198.51**
- **BEL = 237.62 + 197.52 − 198.51 = 236.63**

엔진 236.63 과 손계산이 일치합니다.

```{admonition} 미진단 풀이 두 번 줄어드는 것에 주목
:class: note

t=1 의 암 진단 청구는 *미진단 풀* 0.9851 을 씁니다 — 단순히 진단율로만
줄어든 0.995 가 아닙니다. 풀은 **(아직 살아있고) ∧ (아직 진단 안 받은)**
사람이라, 보유계약 감쇠 (0.99) 와 자기 진단 감쇠 (0.995) 를 *둘 다*
받습니다: `0.99 × 0.995 = 0.98505`. 사망 보장의 `in_force` 와 진단 보장의
`undiagnosed` 풀이 어떻게 다른지는 [보장 청구 메커니즘](../basics/coverage-mechanics).
```

## 결과 읽기 — 담보 추가가 BEL·RA 에 미치는 영향

2.1 의 사망 단독 계약은 BEL 39.11 / RA 16.03 이었습니다. 암 진단 담보를
더하니:

- **BEL 39.11 → 236.63** — 암 진단금 (PV 197.52) 이 미래 유출로 더해져
  부채가 커짐. 진단 담보는 보험료를 거의 안 늘리고 보장만 추가한
  toy 설정이라 BEL 이 크게 증가 (실제 상품은 진단 담보 몫의 보험료가
  따로 붙습니다).
- **RA 16.03 → 32.01** — `morbidity_cv = 0.12` 로 암 진단율의 불확실성이
  RA 에 기여. 사망 위험만 있던 2.1 보다 위험조정이 커짐.
- **CSM 0 / Loss 268.64** — FCF = BEL + RA = 236.63 + 32.01 > 0 이라
  손실부담계약. `Loss = FCF`.

`morbidity_cv` 를 빼면 (또는 0 으로 두면) RA 는 16.03 으로 돌아갑니다 —
진단 담보가 RA 에 기여하려면 자기 변동계수가 필요합니다.

```{admonition} show_trace 로 두 담보 확인
:class: tip

`fcf.show_trace(0, mp, asmp)` 의 Coverages 노드가 두 담보를 나란히
보여줍니다 — `'DEATH' pattern=DEATH`, `'CANCER' pattern=DIAGNOSIS  is_diagnosis=True`.
`is_diagnosis=True` 인 CANCER 만 별도 `undiagnosed` 풀 노드가 붙습니다.
```

## 자주 쓰는 변형

### 진단금만 키우기

진단 담보의 보장금액은 `benefits` 의 1 번 키:

```python
mp = fcf.ModelPoints.single(
    issue_age     = 40,                                       # 가입연령 40 세
    benefits      = {0: 100_000_000, 1: 30_000_000},          # 사망 1 억, 암 진단 3,000 만
    level_premium = 80_000,                                   # 월납 보험료 8 만
    term_months   = 240,                                      # 보험기간 20 년
    calculation_methods = {"DEATH":  fcf.CalculationMethod.DEATH,
                           "CANCER": fcf.CalculationMethod.DIAGNOSIS},
)
```

### 진단 담보를 더 추가 — 뇌혈관 / 심혈관

암 외에 뇌혈관 · 심혈관 진단을 더하려면 `coverages` 에 `CoverageRate`
를, `benefits` 에 키를, `calculation_methods` 에 매핑을 각각 한 줄씩
늘립니다 (모두 `DIAGNOSIS`). 발생률 함수는 담보마다 별도:

```python
coverages = (
    fcf.CoverageRate("DEATH",  death_fn),                     # 0 — 사망
    fcf.CoverageRate("CANCER", cancer_fn),                    # 1 — 암 진단
    fcf.CoverageRate("CEREBRAL", cerebral_fn),                # 2 — 뇌혈관 진단
    fcf.CoverageRate("CARDIAC",  cardiac_fn),                 # 3 — 심혈관 진단
)
```

각 진단 담보는 *자기만의* `undiagnosed` 풀을 가집니다 — 암 진단을 받아도
뇌혈관 풀은 줄지 않습니다 (서로 독립).

## 함정 — 진단율을 `mortality_annual` 에 넣지 말 것

가장 흔한 실수는 진단율을 보유계약 감쇠에 섞는 것입니다:

```python
# 잘못된 예 — 진단율을 decrement 에 더함
mortality_annual = lambda s, a, d: death_fn(s,a,d) + cancer_fn(s,a,d)   # ✗
```

이러면 암 진단받은 사람이 보유계약에서 빠져나가, 그 이후의 사망보험금 ·
보험료 · 만기금이 모두 과소평가됩니다. 암 진단은 보유계약을 줄이지
않습니다 — 진단받은 사람도 살아있고 보험료를 계속 냅니다.

`mortality_annual` 에는 **보유계약을 실제로 줄이는 율** (사망 · 해지) 만
들어갑니다. 진단 · 입원 같은 담보의 발생률은 `coverages` 의 자기 자리에만.
(decrement 와 보장 청구의 분리 — [1.3 사망률의 두 역할](../basics/mortality-roles).)

## 인접 레시피

- [1.4 보장 청구 메커니즘](../basics/coverage-mechanics) — `DEATH` 의
  공유 `in_force` 와 `DIAGNOSIS` 의 per-coverage `undiagnosed` 풀이
  엔진 안에서 어떻게 다르게 도는지.
- [2.1 정기보험 평가](term-life) — 사망 단독, 본 챕터의 출발점.
- 보험료 납입면제 (waiver) (작성 예정) — 상태 추적이 들어가는 첫 챕터.
- 다종 진단 + 면책 / 감액 (작성 예정) — 90 일 면책 / 감액기간 같은
  보장 룰 추가.
- [검증 패턴](../workflow/validation) — `show_trace` 로 두 담보의
  cash flow 를 한 줄씩 확인.
```

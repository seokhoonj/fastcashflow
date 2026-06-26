# 2.4 갱신형 보험과 계약의 경계

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **IFRS 17 계약의 경계 (문단 34)** — 어디까지가 **현재 계약**의 현금흐름인지
- 갱신형 보험의 두 해석 — **차기갱신** (갱신 후는 신계약) vs **최종만기**
  (갱신을 같은 계약의 연장으로)
- `ModelPoints.contract_boundary_months` — 측정 범위를 차기갱신에서 **끊기**
- 경계 밖으로 빠지는 것 — 갱신 후 보험료 · 보험금 · 만기급부
:::

## 상품 소개 — 갱신형 보험

**갱신형 보험** (갱신형 암보험, 1년 갱신 실손 등) 은 일정 주기마다 보장이
갱신되고, 갱신 시점에 보험사가 **보험료를 재산정** 합니다. 갱신암은 보통 10~20년
주기로 갱신가가 크게 오르고, 실손은 1년마다 갱신·재산정됩니다.

IFRS 17 의 질문은: **"갱신 후의 보험료 · 보험금을 지금 계약의 현금흐름에 넣어도
되는가?"** 입니다. 답은 **계약의 경계 (contract boundary, 문단 34)** 가 정합니다 —
보험사가 갱신 시점에 위험을 다시 반영해 가격을 **충분히 조정할 실질 권리** 가
있으면, 그 갱신 이후는 현재 계약의 경계 **밖** 이고, 갱신 후 현금흐름은 지금
계약의 BEL 에 넣지 않습니다 (별도 신계약으로 인식).

:::{admonition} 경계는 판단의 결과이지 엔진이 정하는 값이 아님
:class: warning

문단 B62 의 "substantive right" 판단은 약관 · 재가격권 · 감독해석에 따라
**상품별로** 갈립니다. fastcashflow 는 이 회계 판단을 **자동으로 하지 않습니다**
— 실무자가 산정한 경계를 `contract_boundary_months` **입력으로 받아** 그 월
이후를 잘라낼 뿐입니다.
:::

## 모델링 매핑 — 보장기간 vs 경계

:::{list-table}
:header-rows: 1
:widths: 38 62

* - 필드
  - 의미
* - `ModelPoints.term_months`
  - **보장기간 (최종만기)** — 계약의 사실. 예: 80세까지 480개월.
* - `ModelPoints.contract_boundary_months`
  - **문단 34 계약의 경계** — 측정이 멈추는 월. 미설정 시 `term_months` (경계
    없음, 기존 동작). 차기갱신형이면 다음 갱신월.
:::

경계를 넘는 보험료 · 보험금 · 사업비 · 해약환급금은 측정에서 빠지고, **만기급부는
경계가 보장만기에 닿을 때만** 지급됩니다 (경계가 짧으면 만기는 경계 밖).

## 최소 작동 예제 — 차기갱신 vs 최종만기

40세 가입, 보장 80세 (480개월), 갱신주기 10년 (= 차기갱신 120개월) 인 갱신형
암보험을 두 경계로 측정해 비교합니다:

```python
import numpy as np
import fastcashflow as fcf

# 계리적 가정 (평탄 toy -- 실무는 경험률표)
mort   = 0.003  # 사망 decrement
cancer = 0.005  # 암 진단율
lapse  = 0.05   # 해지율

basis = fcf.Basis(
    mortality_annual=mort,                            # 사망 decrement
    lapse_annual=lapse,                               # 해지
    discount_annual=0.03,                             # 할인율
    ra_confidence=0.75,                               # 위험조정 신뢰수준
    mortality_cv=0.10,                                # 사망 변동계수
    morbidity_cv=0.15,                                # 발생 변동계수
    coverages=(fcf.CoverageRate("CANCER", cancer),),  # 암 진단 담보
)

def renewable(boundary):
    return fcf.ModelPoints(
        issue_age          = np.array([40], dtype=np.int64),        # 40세 가입
        benefits           = {"CANCER": np.array([30_000_000.0])},  # 진단금 3,000만
        premium            = np.array([25_000.0]),                  # 월 2.5만
        term_months        = np.array([480], dtype=np.int64),       # 보장 80세 (40년)
        contract_boundary_months=(None if boundary is None
                                  else np.array([boundary], dtype=np.int64)),
        calculation_methods= {"CANCER": fcf.CalculationMethod.MORBIDITY})

final = fcf.gmm.measure(renewable(None), basis, full=False)  # 경계 = 보장만기 480
bdy   = fcf.gmm.measure(renewable(120),  basis, full=False)  # 경계 = 차기갱신 120

print(f"final maturity (480)  BEL {final.bel[0]:>12,.0f}  CSM {final.csm[0]:>11,.0f}")
print(f"next renewal (120)    BEL {bdy.bel[0]:>12,.0f}  CSM {bdy.csm[0]:>11,.0f}")
```

```text
final maturity (480)  BEL   -1,730,470  CSM   1,555,019
next renewal (120)    BEL   -1,017,896  CSM     914,693
```

## 결과 읽기 — 경계가 측정 범위를 끊는다

- **최종만기 (480)** 는 40년 전체를 한 계약으로 봅니다 — 갱신을 같은 계약의
  연장으로 (갱신 후 보험료 · 보험금을 모두 projection). BEL · CSM 이 더 큽니다.
- **차기갱신 (120)** 은 첫 10년만 측정합니다 — 차기갱신 이후는 경계 밖이라
  현금흐름에서 빠지고, **CSM 이 작아집니다** (이익도 첫 경계 기간분만 인식).
  차기갱신 시점에 새로 인수하는 계약은 그때 **신계약** 으로 따로 인식됩니다.

어느 쪽이 맞는지는 상품의 재가격권이 정합니다 — 갱신 시 위험을 충분히 반영해
재산정할 수 있으면 **차기갱신** (문단 34), 못 하면 (갱신가가 약관표로 고정 등)
**최종만기** 쪽입니다.

## 1년 갱신 실손 — 경계 = 12개월

실손은 보통 1년마다 갱신·재산정합니다. 보장기간이 길어도 (재가입 전까지) 경계는
**12개월** 이 됩니다. 단기 경계라 PAA 적격이기도 합니다:

```python
health = fcf.ModelPoints(
    issue_age          = np.array([40], dtype=np.int64),
    benefits           = {"CANCER": np.array([1_000_000.0])},
    premium            = np.array([12_000.0]),
    term_months        = np.array([600], dtype=np.int64),       # 명목 보장기간
    contract_boundary_months = np.array([12], dtype=np.int64),  # 1년 갱신 = 경계
    calculation_methods= {"CANCER": fcf.CalculationMethod.MORBIDITY})
m = fcf.gmm.measure(health, basis, full=False)
print(f"medical 1yr renewal (boundary 12)  BEL {m.bel[0]:>10,.0f}  CSM {m.csm[0]:>10,.0f}")
```

```text
medical 1yr renewal (boundary 12)  BEL   -133,793  CSM    133,305
```

명목 보장기간이 600개월이어도, 경계가 12개월이라 **첫 1년치 현금흐름만** 측정에
들어갑니다 — 매년 갱신이 신계약으로 재인식되는 1년-경계 상품의 IFRS 17 측정입니다.

## 함정 / 검증

### 함정 1 — 경계는 보장기간을 넘을 수 없음

`contract_boundary_months > term_months` 는 생성자가 거부합니다 (경계가 보장
밖으로 늘어날 수 없음). 경계 미설정 시 `term_months` 로 기본 — 경계 없는 기존
동작과 **bit-identical** 입니다.

### 함정 2 — 만기급부는 경계가 만기에 닿을 때만

만기환급형이라도, 경계가 보장만기보다 짧으면 **만기급부는 지급되지 않습니다**
(경계 밖 현금흐름). 차기갱신 경계로 끊으면 원래 만기의 환급금은 현재 계약에서
빠집니다.

### 함정 3 — 갱신 후 보험료를 "예측"해 넣지 말 것

갱신형 (재산정형) 의 갱신 후 보험료는 갱신 시점 경험률 · 재산정으로 정해져
**예측 불가** 합니다. 그래서 문단 34 가 차기갱신에서 경계를 끊는 것이고, 갱신
후 기간을 projection 에 넣으면 안 됩니다. (보험료가 약관표로 **확정** 된
체증형 · 비재산정형은 다른 얘기 — 그건 경계가 늘어나고 확정 보험료 스케줄을
넣습니다.)

## 인접 레시피

- [2.1 정기보험](term-life) — 경계 = 보장만기인 단순 (비갱신) 상품.
- [9.1 결산 / 보유계약 평가](../workflow/settlement) — 차기갱신마다 신계약으로
  재인식하는 워크플로의 출발점.

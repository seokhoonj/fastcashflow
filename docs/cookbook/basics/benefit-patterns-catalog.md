# BenefitPattern 결정 가이드

```{admonition} 이 챕터에서 배우는 것
:class: tip

- `benefit_patterns.csv` 가 무엇인지, 왜 별도 파일인지 — 회사 카탈로그 (taxonomy)
  / 결산 basis / portfolio 의 3-파일 분리
- 다섯 가지 `BenefitPattern` (DEATH / MORBIDITY / DIAGNOSIS / ANNUITY /
  MATURITY) 의 의미와 한국 상품 매핑
- 사망 종류 (일반 / 상해 / 질병 / 재해 / ADB) 가 모두 DEATH 패턴인 이유
- 카탈로그 작성 절차와 등록 누락 시 어디서 에러가 나는가
```

## 왜 `benefit_patterns.csv` 인가 — 라이프사이클 분리

회사가 평가 엔진에 넣는 입력은 갱신 주기가 서로 다릅니다.

| 단위 | 파일 | 갱신 빈도 | 무엇 |
|---|---|---|---|
| 회사 카탈로그 | `benefit_patterns.csv` | 연 1 회 미만 | 어떤 담보가 어떤 패턴인가 |
| 결산 basis | `assumptions.xlsx` | 분기 / 연 | 위험률 / 할인율 / 사업비 / 위험조정 |
| Portfolio | `policies.csv` + `coverages.csv` | 일 / 월 | 어떤 계약이 어떤 담보를 얼마에 |

세 파일이 각자 자기 책임만 가지면 한 갱신이 다른 파일에 닿지 않습니다.
**담보 패턴은 회사 차원의 결정**이고, **위험률은 분기마다 다시 칼리브레이션**되며,
**보유계약은 매일 바뀝니다**.

```python
import fastcashflow as fcf

# 패키지 샘플로 시연 -- 자기 데이터를 쓸 때는 이 네 줄을 빼고
# 그 자리에 자기 파일이 있다고 보면 됩니다.
fcf.save_sample_assumptions("assumptions.xlsx")
fcf.save_sample_policies("policies.csv")
fcf.save_sample_coverages("coverages.csv")
fcf.save_sample_benefit_patterns("benefit_patterns.csv")

basis = fcf.read_assumptions("assumptions.xlsx")
mp    = fcf.read_model_points(
    "policies.csv",
    basis[("TERM_LIFE_A", "GA")],                 # 한 segment 의 Assumptions
    coverages="coverages.csv",
    benefit_patterns="benefit_patterns.csv",      # ← 회사 카탈로그
)
```

## 다섯 가지 `BenefitPattern`

```python
from fastcashflow import BenefitPattern

BenefitPattern.DEATH        # 사망 — rate 가 mortality 류, in-force 줄이지 않음
BenefitPattern.MORBIDITY    # 입원 / 수술 / 통원 — 반복 발생 (in-force 줄지 않음)
BenefitPattern.DIAGNOSIS    # 진단 / 생활비 lump — 1 회 지급, depleting pool
BenefitPattern.ANNUITY      # 생존 연금 — 월 정액 지급 (scalar field)
BenefitPattern.MATURITY     # 만기환급 — 만기 시 1 회 지급 (scalar field)
```

| Pattern | 지급 mechanic | rate 출처 | engine 자리 |
|---|---|---|---|
| **DEATH** | 사망 사건 발생 시 amount 지급 | 자체 rate_table (사망률 / 상해사망률 / ADB 등) | rate-driven coverage 슬롯 |
| **MORBIDITY** | 사건 발생 시 amount 지급, 다음 달에도 발생 가능 | 자체 incidence rate_table | rate-driven coverage 슬롯 |
| **DIAGNOSIS** | 첫 발생 시 1 회 지급, 그 이후 미지급 | 자체 incidence rate_table | rate-driven coverage 슬롯 (depleting pool) |
| **ANNUITY** | 생존 시 매월 / 매 N개월 정액 지급 | rate 없음 (생존자에게 지급) | `ModelPoints.annuity_payment` |
| **MATURITY** | 만기까지 생존 시 1 회 지급 | rate 없음 | `ModelPoints.maturity_benefit` |

```{note}
주계약 (生保 의 주계약 / 損保 의 보통약관) 은 **상품마다 다른 패턴**입니다.
정기 / 종신은 DEATH, 암보험은 DIAGNOSIS, 건강 / 실손은 MORBIDITY, 연금은
ANNUITY. 엔진은 "주계약" 을 모릅니다 — 카탈로그에 등록된 담보들 중 어느
것이 주계약인지는 회사 / product 단위 결정입니다.
```

## 한국 상품 → 패턴 매핑 표

```{list-table}
:header-rows: 1
:widths: 25 25 50

* - 상품 / 담보 종류
  - BenefitPattern
  - 비고
* - 일반사망 (정기 / 종신 주계약)
  - DEATH
  - rate_table = mortality 류
* - 상해사망 / 재해사망 / ADB / 80% 후유장해
  - DEATH
  - rate_table = 자체 incidence
* - 질병사망 특약
  - DEATH
  - rate_table = 질병 사망률
* - 입원 일당 / 수술비 / 통원 / 골절
  - MORBIDITY
  - 반복 발생, in-force 안 줄임
* - 암 / 뇌혈관 / 심혈관 진단비
  - DIAGNOSIS
  - 1 회 지급, depleting pool. 유사암 등 합성 위험률은 사용자 ETL 에서 미리 합성
* - 암 진단 생활비 (N 개월 정액)
  - (현 v1 미지원)
  - 진단 + sojourn-bounded — semi-Markov 영역. 별도 phase
* - 연금 (즉시) / 종신연금
  - ANNUITY
  - rate 없음. 생존자에게 지급
* - 만기환급금 / 단기납 종신의 환급
  - MATURITY
  - 생존 시 1 회. rate 없음
```

```{warning}
**모든 사망 종류는 DEATH 패턴**입니다 — 한국 시장의 사망 보장 분화 (일반
/ 상해 / 질병 / 재해 / 80% 후유장해 의제사망 / ADB) 는 **같은 mechanic**
입니다 (사건 발생 시 amount 지급). 차이는 *rate_table* 일 뿐. 카탈로그에
각자 별도 `coverage_code` 로 등록하되 패턴은 모두 `DEATH`.
```

## 카탈로그 작성 — `benefit_patterns.csv`

세 컬럼:

```
coverage_code,coverage_name,benefit_pattern
DEATH,일반사망 (주계약),DEATH
ADB,재해사망 특약,DEATH
DISEASE_DEATH,질병사망 특약,DEATH
CANCER,암 진단 특약,DIAGNOSIS
INPATIENT,입원 일당,MORBIDITY
ANNUITY,종신 연금,ANNUITY
MATURITY,만기환급,MATURITY
```

* `coverage_code` — 사용자 시스템의 코드 (cross-file join key).
  `coverages.csv` 와 `assumptions.xlsx` 의 `coverages` 시트가 같은 값을 씁니다.
* `coverage_name` — 사람용 라벨. 엔진은 무시; show_trace 표시에만 쓰입니다.
* `benefit_pattern` — 위 다섯 enum 값 중 하나. 다른 값이면
  `ValueError` 가 어느 행에서 났는지 알려줍니다.

```{note}
엔진에는 reserved 코드가 없습니다. 사망 보장은 다른 담보와 똑같이
`coverages.xlsx` 의 rate-driven 자리에 등록하고, `rate_table` 에
mortality_tables (또는 incidence_rate_tables) 의 항목을 가리키게
합니다 — 일반사망 / 상해사망 / 재해사망 / ADB / 질병사망 모두 같은
방식입니다. 엔진의 `mortality_annual` 입력은 **계약 종료 (decrement)**
용도로만 쓰입니다 — 사람이 죽으면 in-force 가 종료되는 것은 모든
상품에 항상 일어나는 사건이라 별도 입력으로 두지만, 사망 보장금의
지급 rate 는 카탈로그에 등록된 DEATH 담보의 자체 `rate_table` 이
결정합니다.
```

## 변형 — 회사 카탈로그를 어떻게 짜는가

회사가 사용하는 모든 담보를 카탈로그에 한 번 정리:

1. 사망 보장 — DEATH 로 등록. rate_table 별로 다른 coverage_code.
2. 입원 / 수술 / 통원 — MORBIDITY.
3. 암 / 뇌 / 심 진단비 — DIAGNOSIS.
4. 연금 / 만기환급 — ANNUITY / MATURITY (이 두 종류는 rate 없음 ⇒
   `coverages` 시트에 rate_table 등록 X).
5. 생활비 / 재진단 / DI 같은 sojourn-bounded 패턴은 v1 미지원 — semi-Markov
   영역. (별도 phase)

회사 카탈로그는 **연 1 회 미만** 으로 바뀝니다. 분기 결산 때 갱신되는
건 `assumptions.xlsx` 입니다 — 카탈로그가 분리되어 있어 결산 워크플로가
카탈로그를 건드리지 않습니다.

## 함정 / 검증 — 등록 누락 시 어디서 에러

네 단계 검증이 카탈로그 미스를 catch 합니다:

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - 단계
  - 어디서
  - 무엇
* - V1
  - `_parse_benefit_patterns` (read 시점)
  - `benefit_pattern` 값이 다섯 enum 중 하나가 아닌 행
* - V2
  - 같음
  - `coverage_code` 가 중복인 행
* - V3
  - `_long_model_points`
  - `assumptions.xlsx` 의 rate-driven 담보가 카탈로그에 누락
* - V4
  - `measure()` / `value()` 진입
  - 카탈로그에 있는 rate-driven 코드의 rate_table 이 basis 에 없음
```

```python
# V1 예시 — 알 수 없는 패턴
# benefit_patterns.csv 의 한 행:
#   ADB,재해사망,DEAATH    # 오타: DEATH → DEAATH
# 결과:
#   ValueError: benefit_patterns row 'ADB': benefit_pattern='DEAATH'
#               is not one of {DEATH, MORBIDITY, DIAGNOSIS, ANNUITY, MATURITY}

# V3 예시 — 카탈로그 누락
# assumptions.xlsx coverages 시트:
#   CA_DIAG | CANCER_STD       ← rate_table 등록
# 그런데 benefit_patterns.csv 에는 CA_DIAG 행 없음.
# 결과:
#   ValueError: rate-driven coverage code(s) ['CA_DIAG'] from the
#               assumptions workbook are not registered in benefit_patterns
#               -- add each code (with its BenefitPattern) to
#               benefit_patterns.csv
```

`show_trace` 의 Coverages 섹션이 한 계약의 카탈로그 매핑을 그대로
보여줍니다:

```
├─ Coverages (rate-driven, n=3)
│   ├─ 'INPATIENT'    pattern=MORBIDITY  risk=1  is_diagnosis=False  rate -> INPATIENT_STD
│   ├─ 'CANCER'       pattern=DIAGNOSIS  risk=1  is_diagnosis=True   rate -> CANCER_STD
│   └─ 'ADB'          pattern=DEATH      risk=0  is_diagnosis=False  rate -> ADB_STD
```

`pattern` 칸이 카탈로그에서 가져온 결정, `risk` / `is_diagnosis` 는
엔진이 패턴에서 derive 한 값입니다.

## 인접 레시피

- [보장 청구 메커니즘](coverage-mechanics) — 각 BenefitPattern 이
  엔진 안에서 어떤 알고리즘으로 처리되는지 (이 챕터는 *카탈로그* 결정,
  메커니즘 챕터는 *실행 알고리즘*).
- [검증 패턴 — show_trace](../workflow/validation) — 카탈로그 변경이
  어느 계산 단계에 어떻게 들어가는지 추적.
- [정기보험 평가](../simple/term-life) — DEATH 만 사용하는 가장 단순한 사례.
- 사망 + 진단 일시금 (작성 예정) — DIAGNOSIS 추가, depleting pool mechanic.

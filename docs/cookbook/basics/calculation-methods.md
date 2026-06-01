# 1.2 담보와 산출방식 매칭

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 엔진이 *왜* 각 담보의 산출방식을 미리 알아야 하는가
- 산출방식에 따라 어떤 계산 분기로 들어가는지 — 다섯 가지로 단순화
- 사용자가 지정해주는 자리 (`calculation_methods.csv`) 가 왜 별도 파일인가
- 사망 종류 (일반사망 / 질병사망 / 재해사망) 가 모두 같은 산출방식인 이유
- 한국 상품의 매핑 표, 담보별 산출방식 작성, 등록 누락 시 잡히는 자리
```

## 왜 사용자가 산출방식을 지정해야 하나

엔진이 청구 (cash flow 발생) 를 계산하는 방식은 담보마다 다릅니다. **사망
보장** 은 사람이 한 번 죽으면 더 발생하지 않고, **입원 보장** 은 같은
사람이 매월 다시 발생할 수 있고, **진단 보장** 은 한 번 진단 받으면
다음 달부터 발생하지 않습니다. 같은 수학식 한 줄로 세 경우를 다 풀 수
없어, 엔진이 담보별로 *다른 계산 알고리즘* 을 골라 적용해야 합니다.

문제는 엔진이 담보의 이름 (`'CANCER'`, `'INPATIENT'`, `'ADB'` 등) 만 보고
어떤 알고리즘을 적용할지 자동 추론할 수 없다는 점 — 한국 시장의 담보
이름은 회사마다, 상품마다 자유 형식입니다. 그래서 사용자가 **"이 담보는
이런 식으로 지급된다"** 를 미리 알려줘야 합니다. 그 매핑이 *담보별 산출방식*
(`calculation_method`) 이고, 새로운 담보가 생길 때 추가로 매칭해주면 됩니다.

다섯 가지 분기로 단순화됩니다:

```{list-table}
:header-rows: 1
:widths: 18 35 47

* - 산출방식
  - 어떤 계산
  - 한국 상품 예
* - **DEATH**
  - 사망률 × 보유계약 × 보험금. 보유계약은 별도 사망률로 따로 감쇠 (한 번 죽으면 끝).
  - 일반사망 / 질병사망 / 재해사망 등 *모든 사망형 담보*
* - **MORBIDITY**
  - 발생률 × 보유계약 × 보험금. 매월 반복 (같은 사람도 다시 발생 가능).
  - 입원 / 수술 / 통원
* - **DIAGNOSIS**
  - 진단율 × 미진단풀 × 보험금. 풀이 매월 감쇠 (한 번 진단 받으면 끝).
  - 암 / 뇌혈관 / 심혈관 진단보험금
* - **ANNUITY**
  - 생존계약 × 정액 (매월 / 매 N 개월). 위험률 없음.
  - 생존연금
* - **MATURITY**
  - 만기 생존계약 × 정액. 만기 시 1 회.
  - 만기환급금
```

세 가지 위험률 기반 산출방식 (DEATH / MORBIDITY / DIAGNOSIS) 의 차이는 본질
적으로 *어떤 풀에서 차감되는가* 의 차이입니다 — 자세한 메커니즘은
[보장 청구 메커니즘](coverage-mechanics) 챕터.

```{note}
주계약은 **상품마다 다른 산출방식** 입니다. 정기 / 종신은 DEATH, 암보험은
DIAGNOSIS, 건강 / 실손은 MORBIDITY, 연금은 ANNUITY. 엔진은 "주계약"
자체를 모릅니다 — 담보별 산출방식에 등록된 담보들 중 어느 것이 주계약인지는
회사 / product 단위 결정.
```

## 왜 별도 파일인가 — 책임 분리

회사가 평가 엔진에 넣는 입력은 각자 다른 *책임* 을 갖습니다.

| 단위 | 파일 | 무엇 |
|---|---|---|
| 결산 basis | `assumptions.xlsx` | 위험률 / 할인율 / 사업비 / 위험조정 |
| Portfolio | `policies.csv` + `coverages.csv` | 어떤 계약이 어떤 담보를 얼마에 |
| 담보별 산출방식 | `calculation_methods.csv` | 어떤 담보가 어떤 산출방식인가 |

세 파일이 각자 자기 책임만 가지면 한 갱신이 다른 파일에 닿지 않습니다.
담보별 산출방식은 *신담보 추가* 작업의 자리, basis 는 *위험률 calibration*
의 자리, portfolio 는 *정책관리* 의 자리 — 그래서 한 작업이 다른 자리를
건드리지 않습니다.

```python
import fastcashflow as fcf

# 패키지 샘플로 시연 -- 자기 데이터를 쓸 때는 이 네 줄을 빼고
# 그 자리에 자기 파일이 있다고 보면 됩니다.
fcf.save_sample_basis("assumptions.xlsx")             # .xlsx 만 (multi-sheet 워크북)
fcf.save_sample_policies("policies.csv")                    # .csv / .xlsx / .parquet / .feather
fcf.save_sample_coverages("coverages.csv")                  # .csv / .xlsx / .parquet / .feather
fcf.save_sample_calculation_methods("calculation_methods.csv")    # .csv / .xlsx / .parquet / .feather

mp = fcf.read_model_points(
    "policies.csv",                                 # 계약 spec 파일
    coverages="coverages.csv",                      # 담보 가입금액 파일
    calculation_methods="calculation_methods.csv",  # ← 담보별 산출방식
)
```

## 한국 상품 → 산출방식 매핑 표

```{list-table}
:header-rows: 1
:widths: 25 25 50

* - 상품 / 담보 종류
  - CalculationMethod
  - 비고
* - 일반사망 (정기 / 종신 주계약)
  - DEATH
  - rate_table = mortality 류 (all-cause)
* - 재해사망 (80% 후유장해 의제사망 포함)
  - DEATH
  - rate_table = 자체 incidence (재해 사고 사망률)
* - 질병사망
  - DEATH
  - rate_table = 질병 사망률 (재해 제외)
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
**모든 사망 종류는 DEATH 산출방식**입니다 — 한국 시장의 사망 보장 분화
(일반사망 / 질병사망 / 재해사망) 는 **같은 mechanic** (사건 발생 시
amount 지급). 차이는 *rate_table* 일 뿐. 담보별 산출방식에 각자 별도
`coverage_code` 로 등록하되 산출방식은 모두 `DEATH`.
```

## 담보별 산출방식 작성 — `calculation_methods.csv`

세 컬럼:

```
coverage_code,coverage_name,calculation_method
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
* `coverage_name` — 사람용 라벨. 엔진은 무시; gmm.trace 표시에만 쓰입니다.
* `calculation_method` — 위 다섯 enum 값 중 하나. 다른 값이면
  `ValueError` 가 어느 행에서 났는지 알려줍니다.

```{note}
엔진에는 reserved 코드가 없습니다. 사망 보장은 다른 담보와 똑같이
`coverages.xlsx` 의 rate-driven 자리에 등록하고, `rate_table` 에
mortality_tables (또는 incidence_rate_tables) 의 항목을 가리키게
합니다 — 일반사망 / 질병사망 / 재해사망 모두 같은
방식입니다. 엔진의 `mortality_annual` 입력은 **계약 종료 (decrement)**
용도로만 쓰입니다 — 사람이 죽으면 in-force 가 종료되는 것은 모든
상품에 항상 일어나는 사건이라 별도 입력으로 두지만, 사망 보장금의
지급 rate 는 담보별 산출방식에 등록된 DEATH 담보의 자체 `rate_table` 이
결정합니다.
```

## 변형 — 담보별 산출방식을 어떻게 짜는가

회사가 사용하는 모든 담보를 한 번 정리:

1. 사망 보장 — DEATH 로 등록. rate_table 별로 다른 coverage_code.
2. 입원 / 수술 / 통원 — MORBIDITY.
3. 암 / 뇌 / 심 진단비 — DIAGNOSIS.
4. 연금 / 만기환급 — ANNUITY / MATURITY (이 두 종류는 rate 없음 ⇒
   `coverages` 시트에 rate_table 등록 X).
5. 생활비 / 재진단 / DI 같은 sojourn-bounded 산출방식은 v1 미지원 — semi-Markov
   영역. (별도 phase)

담보별 산출방식은 **신담보가 추가될 때만** 한 줄을 더해줍니다. 분기 결산
때 갱신되는 건 `assumptions.xlsx` 입니다 — 담보별 산출방식이 분리되어 있어
결산 워크플로가 이 파일을 건드리지 않습니다.

## 함정 / 검증 — 등록 누락 시 어디서 에러

네 단계 검증이 담보별 산출방식 미스를 catch 합니다:

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - 단계
  - 어디서
  - 무엇
* - V1
  - `_parse_calculation_methods` (read 시점)
  - `calculation_method` 값이 다섯 enum 중 하나가 아닌 행
* - V2
  - 같음
  - `coverage_code` 가 중복인 행
* - V3
  - `_long_model_points`
  - `assumptions.xlsx` 의 rate-driven 담보가 담보별 산출방식에 누락
* - V4
  - `measure()` 진입
  - 담보별 산출방식에 있는 rate-driven 코드의 rate_table 이 basis 에 없음
```

```python
# V1 예시 — 알 수 없는 산출방식
# calculation_methods.csv 의 한 행:
#   ADB,재해사망,DEAATH    # 오타: DEATH → DEAATH
# 결과:
#   ValueError: calculation_methods row 'ADB': calculation_method='DEAATH'
#               is not one of {DEATH, MORBIDITY, DIAGNOSIS, ANNUITY, MATURITY}

# V3 예시 — 담보별 산출방식 누락
# assumptions.xlsx coverages 시트:
#   CA_DIAG | CANCER_STD       ← rate_table 등록
# 그런데 calculation_methods.csv 에는 CA_DIAG 행 없음.
# 결과:
#   ValueError: rate-driven coverage code(s) ['CA_DIAG'] from the
#               assumptions workbook are not registered in calculation_methods
#               -- add each code (with its CalculationMethod) to
#               calculation_methods.csv
```

`gmm.trace` 의 Coverages 섹션이 한 계약의 담보별 산출방식 매핑을 그대로
보여줍니다:

```
├─ Coverages (rate-driven, n=3)
│   ├─ 'INPATIENT'       method=MORBIDITY  risk=1  is_diagnosis=False  rate -> INPATIENT_STD
│   ├─ 'CANCER'          method=DIAGNOSIS  risk=1  is_diagnosis=True   rate -> CANCER_STD
│   └─ 'ADB'             method=DEATH      risk=0  is_diagnosis=False  rate -> ADB_STD
```

`method` 칸이 담보별 산출방식에서 가져온 값, `risk` / `is_diagnosis` 는
엔진이 산출방식에서 derive 한 값입니다.

## 인접 레시피

- [보장 청구 메커니즘](coverage-mechanics) — 각 CalculationMethod 이
  엔진 안에서 어떤 알고리즘으로 처리되는지 (이 챕터는 *담보별 산출방식 선택*,
  메커니즘 챕터는 *실행 알고리즘*).
- [검증 패턴 — gmm.trace](../workflow/validation) — 담보별 산출방식 변경이
  어느 계산 단계에 어떻게 들어가는지 추적.
- [정기보험](../simple/term-life) — DEATH 만 사용하는 가장 단순한 사례.
- 사망 + 진단 일시금 (작성 예정) — DIAGNOSIS 추가, depleting pool mechanic.

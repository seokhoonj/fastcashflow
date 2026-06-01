# Design decisions

가정 워크북 구조와 명명 규칙에 대해 결정된 내용과 그 이유. 각 항목은
나중에 다시 같은 토론을 반복하지 않도록 **근거**까지 기록합니다.

## 1. 단일 워크북 (basis.xlsx)

**결정**: rate tables + segments + coverages 등 여러 시트를 한 워크북에
담는다. 이전 v1 reader가 썼던 두 파일 분리 (rate-table 워크북 + 별도 매핑
워크북) 는 폐기.

**근거**:
- 두 파일 분리를 정당화하는 시나리오 — (1) 한 tables를 여러 basis가 공유,
  (2) tables는 분기 갱신/basis는 월 갱신, (3) 거대 워크북 회피 — 가 현
  프로젝트 규모에서 모두 약함.
- 현실: 한 portfolio = 한 valuation 입력 묶음 → 한 파일에 같이 두는 게
  자연.
- 사용자 인터페이스 단순: `read_basis(path)` 단일 경로.

**대안**:
- 외부 table 라이브러리 (대형 valuation 플랫폼들이 갖는 inventory 패턴)
  는 fastcashflow 규모를 넘는다. yagni.

---

## 2. 시트명 `segments` (이전엔 `basis`)

**결정**: (product_code, channel_code) 매핑 시트의 이름은 `segments`. `basis`는
쓰지 않음.

**근거**:
- 보험계리에서 "basis" (산출기초) 는 mortality·lapse·expense·discount
  까지 묶은 **valuation 입력 전체**를 가리키는 용어. 한 시트의 이름으로
  들어올리면 의미 충돌.
- 그 시트의 실제 역할은 "(product_code, channel_code) → 어느 table을 쓸지 + 스칼라
  파라미터 배정" 즉 **매핑 / configuration**. 기초율 아님.
- 후보들:
  - `mapping_tables`: `_tables` 접미사가 "여러 named table의 registry"
    의미로 일관되게 쓰여 (mortality_tables 등) 단일 매핑 시트엔 의미 부정확
  - `mapping_table` (단수): 워크북 내 유일한 단수형 — 패턴 일관성 깨짐
  - `segments`: `coverages`와 같은 패턴 (복수형 일반명사, 행의 정체를 직시).
    "각 행 = 한 segment". 채택.

**관련 코드**: 리더 내부에서도 `segments`라는 변수명을 이미 사용 중.

---

## 3. 파일명 `basis.xlsx` (이전엔 `assumptions.xlsx`)

**결정**: 워크북 파일명은 `basis.xlsx`. Python 클래스 `Basis`, 모듈
`basis.py`, 함수 `read_basis()` 와 한 단어로 통일. 번들 샘플은
`sample_basis.xlsx` (VFA 샘플 `sample_vfa_basis.xlsx` 와 같은 패턴).

**근거**:
- #2 의 정의대로 "basis" (산출기초) 는 mortality·lapse·expense·discount
  를 묶은 **valuation 입력 전체** 를 가리키는 넓은 용어 — 워크북이 담는
  것이 정확히 그것. 파일 = 직렬화된 basis, `read_basis()` 가 그것을
  `Basis` 개체로 적재 (`config.yaml` → `Config` 패턴).
- 한때 `basis.xlsx` 를 꺼린 이유였던 자기참조 ("basis.xlsx 안의 basis
  시트") 는 더 이상 성립하지 않음 — 그 매핑 시트는 #2 에서 `segments` 로
  명명됐고, 워크북에 `basis` 라는 시트는 없음.
- 클래스 / 모듈 / 함수 / 파일이 한 단어 (`basis`) 로 일관.

**이력**: 초기에는 자기참조 우려로 `assumptions.xlsx` 를 골랐으나, 매핑
시트가 `segments` 로 확정되어 우려가 사라졌고 `Basis` / `read_basis` 와의
단일 어휘 통일을 위해 2026-06 `basis.xlsx` 로 전환.

---

## 4. 컬럼 값 명명 규칙

**결정**:
| 컬럼 | 규칙 | 예 |
|---|---|---|
| `product` | SCREAMING_SNAKE_CASE | `TERM_A`, `WHOLE_LIFE_A` |
| `channel` | ALL UPPERCASE 약어 | `GA`, `FC`, `BANCA` |
| `table_id` | SCREAMING_SNAKE_CASE | `MORTALITY_STD`, `LAPSE_GA` |
| `coverage_code` | SCREAMING_SNAKE_CASE | `DEATH`, `INPATIENT` |
| `calculation_method` | SCREAMING_SNAKE_CASE | `DEATH`, `MORBIDITY` |
| 컬럼 헤더 | snake_case 소문자 | `alpha_flat`, `mortality_cv` |

**근거**:
- channel은 업계 관용 약어 (General Agency, Financial Consultant)
  대문자 표기 보존이 정보 손실 적음.
- table_id 와 product 는 외부 식별자 (코드 상수에 가까운 named reference)
  이라 SCREAMING_SNAKE_CASE 로 데이터 값과 시각적으로 구분됨.
  product / channel / table_id 가 모두 대문자 family 로 통일되어
  컬럼명(소문자) 과 값(대문자) 의 시각적 구분도 강해짐.
- coverage_code 는 model points / 코드 enum 과 1:1 매핑되므로 snake_case
  소문자.

---

## 5. Column semantics (`rate` / `amount` / `factor`)

**결정**: 컬럼 이름이 값의 의미를 표시.

| 컬럼명 | 의미 | 단위 | 시트 |
|---|---|---|---|
| `rate` | 확률 / 발생률 / 환산률 | 무차원 (0~1) | mortality, incidence_rate, waiver, lapse, discount, inflation |
| `amount` | 화폐 금액 | 원 (또는 portfolio 통화) | maintenance |
| `factor` | 곱셈자 (multiplier) | 무차원 (보통 ~1.0) | **현재 없음**; 향후 A/E layer 도입 시 (Task #8) |

**근거**:
- 의미가 다른 값에 같은 컬럼명을 쓰면 reader가 단위 처리를 잘못할
  소지. `rate` 가정 → 0~1 검증 / `amount` 가정 → 통화 처리.
- 기존 `maintenance_tables.rate` (값이 60,000 같은 원 단위)는 명칭
  오류. `amount`로 rename 확정.

---

## 6. 위험률 vs 발생률 가정 (best estimate) — IFRS 17 정합

**결정**: 워크북의 rate table들은 **best-estimate 발생률 가정**으로
간주한다. "위험률" (pricing 기초율) 이라는 표현은 docs/naming에서 제거.

**근거**:
- IFRS 17 Sec. 33, B37: BEL은 편의 없는 (unbiased) 최선추정으로 측정.
  pricing 기초율 (보수적 마진 포함) 을 그대로 쓰면 BEL 과대 / CSM 부풀림.
- 한국 산업 실무: 가정관리시스템에서 `위험률 × A/E ratio = best estimate`
  를 외부에서 계산한 결과가 valuation 엔진의 입력. fastcashflow는 그
  결과를 받는다.
- pricing 위험률 마진은 `level_premium` 안에 이미 녹아 있어서, `premium_cf
  > E(claim_cf)` 가 자연스럽게 발생 — 이 차이가 CSM의 원천.

---

## 7. A/E factor 레이어 — 패턴 1 vs 2

**결정**: 워크북에 A/E factor 시트를 **선택적 레이어로 추가** (Task #8,
미구현). 기본은 없음 (factor 1.0). 사용자가 (a) 외부에서 위험률 × A/E를
미리 곱해 base rate에 넣거나 (b) base + ae_factors 시트 둘 다 채워서
엔진 런타임 곱셈 — 둘 다 지원.

**근거**:
- 두 워크플로우:
  - **패턴 1 (외부 calibration)**: 가정관리시스템에서 사전 곱셈, 엔진엔
    final best-estimate만 입력. 한국 자체개발 시스템에 많은 형태.
  - **패턴 2 (런타임 곱셈)**: 엔진이 base + multiplier 둘 다 받음.
    상용 valuation 플랫폼들의 기본 동작.
- 둘 다 산업에 존재 → 한쪽에 치우치지 않게 양쪽 다 지원.
- 단, scalar A/E는 (age × duration × channel) 패턴을 못 잡으므로 ae_factors
  시트도 axis-flex 설계로 (Task #7과 같은 schema-detection).

**선행 토론 / 폐기 안**:
- 처음엔 "fastcashflow의 (product × channel × age) per-segment table 설계가
  이미 A/E를 함의" 라고 했으나, 산업 정합 / 운영 편의 (sensitivity 한 칸
  수정) 측면에서 별도 레이어 지원이 가치 있다는 결론.

---

## 8. Maximize 설계 원칙 — Layered, optional

**결정**: 엔진의 callable은 최대 차원 (sex, issue_age, duration,
calendar_year)을 받는다. 워크북 입력층은 base table (필수) + A/E factor
(optional) + age_shift (optional) + improvement (optional) 의 **4 레이어
조합**으로 표현. 사용자가 채운 레이어만 활성, 안 채우면 no-op (1.0 / 0).

**근거**:
- "엔진은 풍부하게, 입력은 단순~복잡까지 사용자 선택" 원칙.
- 각 레이어 의미:
  - **base** (Task #7): 발생률의 main shape. axis-flex로 1차원 (scalar)
    부터 4차원 ((sex, issue_age, duration)) 까지.
  - **A/E factor** (Task #8): 런타임 multiplier. base와 같은 axis-flex.
  - **age_shift** (Task #9): 정수 shift. table 재사용 시 cohort 보정.
  - **improvement** (Task #10): 연도별 mortality improvement scale (산업 표준 패턴).
- 4개 layer 각각이 실무에서 흔히 보이는 워크어라운드 (위험률 × A/E를
  외부 Excel에서 미리 곱셈, cohort마다 테이블 복사, improvement scale
  사전 곱셈, select-and-ultimate 별도 시트 분리) 를 시스템 내부 1급
  표현으로 끌어들임. 입력 안 채우면 단순 base 모델과 동일하게 동작.

---

## 9. Sample identifier — generic placeholder

**결정**: 샘플 워크북의 table_id는 generic placeholder.
- `MORTALITY_STD` (기존 `KMT_STD`)
- `DISCOUNT_STD` (기존 `RFR_2025`)
- `WAIVER_STD`, `INFLATION_STD`, `LAPSE_GA`, `LAPSE_FC`,
  `INPATIENT_STD`, `CANCER_STD`, `ADB_STD` 등은 이미 generic

**근거**:
- `KMT_STD` (Korean Mortality Table Standard 패턴) 는 placeholder인데
  실제 한국 산업 표준 약어 (예: KIDI 경험생명표 9회) 와 혼동 가능. "이게
  KELT-9인가" 오해 회피.
- `RFR_2025` 의 "2025"는 vintage 표시지만 sample에 박혀 있으면 시간
  지나서 오래된 데이터로 보임. `DISCOUNT_STD` 로 단순화.
- 실 사용 시엔 회사 경험분석 / 감독원 발표 자료의 정확한 식별자로 교체
  (docs에 명시).

---

## 10. 도입 결정 (확정) vs 향후 검토 (보류)

**확정**:
- 1~9 위 항목 전부
- 워크북 단일화, `segments` 시트명, snake_case 규칙, column semantics,
  best-estimate 정의, layered optional 설계 골격

**보류 (Task로 등록, 별도 구현)**:
- Task #7: base table axis-flex 리더 (schema-detecting)
- Task #8: A/E factor 레이어
- Task #9: age_shift 컬럼
- Task #10: improvement 레이어
- Task #1: discount / inflation / maintenance curve화
- Task #5: curves.py / numerics.py 레이어 분리 (완료)
- Task #2: ModelPoints product/channel + measure helper

**검토 안 함 (yagni)**:
- 워크북 두 파일 분리 (#1 결정)
- "위험률" 그대로 입력 (#6 결정 — best-estimate이 표준)
- 손해율 테이블 입력 (별도 토론, GMM은 발생률 × 금액 직접 계산)
- VFA 파라미터 (`fund_fee`, `guaranteed_credit_rate`) Basis 잔류
  ([[vfa-param-relocation]] 메모 — 별도 refactor)

# 계리 가정 입력 포맷 (assumptions input)

fastcashflow 엔진에 들어가는 **계리 가정**을 정의하는 입력 포맷입니다. 현업
담당자가 이 스펙에 맞춰 요율·파라미터를 채워 넣습니다.

> **상태**: 이 문서는 *목표 포맷*입니다. 현재 엔진(`read_assumptions`)은 구
> 단일 워크북 포맷을 읽습니다 — 이 레지스트리 기반 포맷으로 reader를 개편하는
> 작업은 별도 진행됩니다.

---

## 1. 개요 — 두 워크북

| 파일 | 역할 |
|---|---|
| `tables.xlsx` | **레지스트리** — 이름 붙은 요율 표들. 한 번 등록하고 여러 곳에서 참조. |
| `assumptions.xlsx` | **basis** — (상품 × 채널) 세그먼트마다 *어느 표를 쓰는지* + 스칼라 값. |

가정은 **세그먼트**(상품 × 채널) 단위로 다릅니다. 한국 시장에서 유지율은
채널(GA / 전속 등)별로 크게 다르고, 사망률·사업비도 상품별로 갈립니다.
레지스트리에 표를 한 번만 등록하고, basis가 세그먼트별로 그 표를 참조합니다.

엔진은 세그먼트별로 평가한 뒤 IFRS 17 그룹으로 합산합니다.

---

## 2. 공통 규약

- **모든 요율은 연(annual)** 단위입니다. 엔진이 월 요율로 변환합니다
  (constant-force — 12회 적용 시 연 요율이 정확히 복원되는 변환).
- `sex`: `0` = 남, `1` = 여.
- 모든 시트의 **1행은 헤더**, 2행부터 데이터입니다. 컬럼은 헤더 **이름**으로
  읽습니다 (순서 무관).
- 표는 **조회 범위 밖이면 끝값을 유지**합니다. 따라서 균일한 가정은 **한 줄만**
  넣으면 전 구간에 적용됩니다 (flat = 1행).
- 두 가지 시간축을 구분합니다:
  - **경과연수(`duration`)** — 계약 발행 후 경과 연수, `0`부터. lapse·
    maintenance에 사용. (mortality는 *도달연령* = 가입연령 + 경과연수.)
  - **투영연도(`year`)** — 평가일로부터의 연수, `0`부터. discount·inflation·
    investment에 사용.

---

## 3. `tables.xlsx` — 레지스트리

각 시트는 여러 표를 `table_id`로 구분해 담습니다. 한 시트 안에서 `table_id`가
다르면 다른 표입니다.

### 3.1 `mortality_tables` · `waiver_tables` · `rider_rate_tables`

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `KMT2019`, `CANCER_DX_v3`) |
| `sex` | 0 남 / 1 여 |
| `age` | 도달연령 (정수, 연속) |
| `rate` | 연 요율 |

- `mortality_tables` — 기본 사망률 (주계약 사망 + in-force 감소).
- `waiver_tables` — 납입면제 / 장해 **개시율** (active 상태에서 면제·장해
  상태로의 전이율).
- `rider_rate_tables` — 요율 기반 특약(사망형·질병형·진단형)의 위험률.
- 두 성별은 같은 연령 범위를 가져야 하며, 연령은 연속 정수여야 합니다.

### 3.2 `lapse_tables` · `maintenance_tables`

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `LAPSE_GA_TERM`, `MAINT_STD`) |
| `duration` | 경과연수 (0부터 연속) |
| `rate` | `lapse`: 연 해지율 · `maintenance`: 계약당 연 유지비(실질 단가) |

- `lapse_tables` — 해지율. 보통 상품 × 채널별로 다릅니다.
- `maintenance_tables` — 유지비 단가. 경과연수별로 다르면 여러 줄, 단일
  단가면 한 줄. **실질 단가**이며 인플레이션은 `inflation` 표가 따로 키웁니다.

### 3.3 `discount_tables` · `inflation_tables` · `investment_tables`

| 컬럼 | 의미 |
|---|---|
| `table_id` | 표 식별자 (예: `RFR_2025Q1`) |
| `year` | 투영연도 (평가일로부터, 0부터) |
| `rate` | `discount`: 연 할인율 · `inflation`: 연 비용 인플레이션 · `investment`: 연 투자수익률 |

- 경제 가정입니다 — 시장 기준, **평가일**별로 다르며 상품·채널과 무관합니다.
- `discount_tables` — 할인율 기간구조(커브). 평면 할인율이면 한 줄.
- `investment_tables` — 변액(VFA) 상품 전용.

---

## 4. `assumptions.xlsx` — basis

### 4.1 `basis` 시트

`defaults` 행 하나 + 세그먼트(상품 × 채널)별 행들. **빈 칸은 `defaults` 행의
값을 상속**하고, 채우면 그 행에서 override합니다.

**키 컬럼**: `product`, `channel` (선택: `valuation_date` — 여러 평가시점을
한 시트에 보관할 때).

**표 참조 컬럼** (값 = 레지스트리의 `table_id`):
`mortality_table`, `lapse_table`, `maintenance_table`, `waiver_table`,
`discount_table`, `inflation_table`, `investment_table`.
- `waiver_table` — 납입면제·장해가 없는 상품이면 비웁니다.
- `investment_table` — 변액(VFA) 상품에만.

**스칼라 컬럼** (값 = 숫자/문자):
`expense_acquisition`, `ra_confidence`, `mortality_cv`, `morbidity_cv`,
`longevity_cv`, `disability_cv`, `expense_cv`, `cost_of_capital_rate`,
`ra_method`.

예시:

```
product channel mortality_table lapse_table   discount_table expense_acquisition ra_confidence mortality_cv
defaults        KMT2019                       RFR_2025Q1                         0.75          0.10
정기A   GA                      LAPSE_GA_TERM                150000
정기A   전속                    LAPSE_FC_TERM                 80000
변액B   GA      KMT2019_SEL     LAPSE_GA_SAV                 200000              0.80          0.18
```

- `정기A/GA` — `mortality_table`·`discount_table`·`ra_confidence`·
  `mortality_cv`는 빈 칸이라 `defaults`에서 상속(`KMT2019`, `RFR_2025Q1`,
  0.75, 0.10). `lapse_table`·`expense_acquisition`만 행에서 지정.
- `변액B/GA` — `mortality_table`(`KMT2019_SEL`)·`ra_confidence`(0.80)·
  `mortality_cv`(0.18)를 override.

전사 고정값(`ra_confidence` 등)은 `defaults`에 한 번만 적습니다 — 바꿀 때
한 칸만 고치면 됩니다.

### 4.2 `riders` 시트

각 상품이 어떤 특약을 갖는지, 특약별 유형과 요율표를 정의합니다.

| 컬럼 | 의미 |
|---|---|
| `product` | 상품 |
| `rider_code` | 특약 코드 (모델포인트 파일이 담보를 이 코드로 지칭) |
| `rider_name` | 특약명 (사람용 메모 — 엔진은 안 읽음) |
| `type` | `death_main` / `death` / `morbidity` / `diagnosis` / `annuity` / `maturity` |
| `rate_table` | 요율 기반(`death`·`morbidity`·`diagnosis`)이면 `rider_rate_tables`의 `table_id`; 나머지는 빈칸 |

- `death_main` — 주계약 사망. 요율은 basis의 `mortality_table`이 구동하므로
  `rate_table`은 빈칸.
- `annuity` / `maturity` — 생존급부. 요율 없음.

---

## 5. reader 해소 순서

reader는 두 워크북을 읽어 세그먼트별 `Assumptions`를 만듭니다:

1. `tables.xlsx`의 모든 표를 `table_id`로 적재.
2. `basis` 시트의 각 세그먼트 행에서 빈 칸을 `defaults`로 채움.
3. 표 참조 컬럼의 `table_id`를 실제 표로 해소.
4. `riders` 시트에서 그 상품의 특약 목록을 붙임.
5. → `(상품, 채널)`별 `Assumptions` 하나 완성. 엔진이 세그먼트별로 평가.

---

## 6. 범위 밖 — 참고

- **상품 구조**(상태기계 — 납입면제·장해 상태 전이, 어떤 특약을 갖는지의
  구조)는 가정이 아니라 *상품 정의*입니다. 이 워크북은 **요율과 파라미터**만
  담습니다.
- 변액 계약조건(`fund_fee`, 최저보증이율)은 계리 가정이 아니라 계약조건이라
  이 워크북에 들어가지 않습니다 — 상품 정의 / 모델포인트 쪽입니다.
- 계약별 정보(가입연령·보험금·보험료·계좌가치 등)는 가정이 아니라
  **모델포인트** 파일입니다.

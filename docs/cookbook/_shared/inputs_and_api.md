### 입력 파일과 입력 개체

엔진은 두 클래스의 *개체* 만 받습니다 — `Assumptions` 와 `ModelPoints`.
**`measure(mp, asmp)`** 호출의 두 인자가 바로 이 개체들. 사용자가 다루는
*입력 파일들* 은 reader 함수를 거쳐 이 두 개체로 모입니다:

- **`Assumptions` 클래스** — `basis = fcf.read_assumptions(...)` 가 한
  입력 파일 (`assumptions.xlsx`, multi-sheet 워크북) 을 읽어 가정 개체를
  만듭니다.
- **`ModelPoints` 클래스** — `mp = fcf.read_model_points(...)` 가 세 입력
  파일 (`policies.csv` / `coverages.csv` / `calculation_methods.csv`) 을 읽어
  한 개체로 합칩니다.

#### Assumptions 클래스 — 한 입력 파일에서 만드는 개체

```
Assumptions (basis = fcf.read_assumptions(...))
└── assumptions.xlsx          ── 계리적 가정 (multi-sheet workbook)
    ├── segments              · (product_code, channel_code) → 어느 테이블 쓸지
    ├── mortality_tables      · table_id × sex × age → 사망률
    ├── lapse_tables          · table_id × duration → 해지율
    ├── discount_tables       · table_id × year → 할인율
    ├── expense_tables        · table_id → 사업비 행 (acquisition / maintenance / ...)
    └── coverages             · 담보 코드 → 어느 위험률 테이블을 쓸지
```

#### ModelPoints 클래스 — 세 입력 파일에서 만드는 개체

```
ModelPoints (mp = fcf.read_model_points(...))
├── policies.csv              ── 보유 계약 (한 줄 = 한 계약, 가입 시점 영구 spec)
│   ├── mp_id                 · 계약 식별자 (다른 파일과 join 키)
│   ├── product_code          · 어느 segment 가정을 쓸지 (assumptions 의 segments 와 맞물림)
│   ├── channel_code          · 채널
│   ├── issue_age             · 가입연령
│   ├── sex                   · 0 = 남, 1 = 여
│   ├── term_months           · 보험기간 (개월)
│   ├── premium_term_months   · 보험료 납입기간 (개월)
│   └── count                 · 이 줄이 대표하는 계약 수 (없으면 1)
│
├── coverages.csv             ── 담보 가입금액 (long-form, 한 줄 = 한 (계약, 담보))
│   ├── mp_id                 · 어느 계약의 담보인지
│   ├── coverage_code         · 담보 코드 (calculation_methods 의 코드와 맞물림)
│   ├── amount                · 가입금액 (보험금)
│   ├── premium               · 월 보험료 (선택)
│   ├── waiting               · 면책기간 개월수 (선택)
│   ├── reduction_end         · 감액기간 종료 개월수 (선택)
│   └── reduction_factor      · 감액기간 중 지급률 (선택, 0..1)
│
└── calculation_methods.csv      ── 담보 계산방식 (담보 코드 → 지급 패턴)
    ├── coverage_code         · 담보 코드 (DEATH, CANCER, INPATIENT ...)
    ├── coverage_name         · 사람 친화 라벨 (선택)
    └── calculation_method       · DEATH / MORBIDITY / DIAGNOSIS / ANNUITY / MATURITY
```

결산 모드 (보유계약 평가) 에서는 `policies.csv` 가 *분기말 상태 컬럼*
네 개를 더 갖는 `inforce_2026Q1.csv` 같은 한 파일로 들어옵니다 —
`elapsed_months` / `count` (잔존) / `prior_csm` (직전 분기 CSM) /
`lock_in_rate` (가입 시점의 할인율). reader 도
`mp, state = fcf.read_inforce_policies(...)` 로 바뀌어 두 개체를 돌려줍니다.

### 사용자 함수

```
fastcashflow 사용자 API
├── 샘플 파일 폴더에 생성 (한 번만, 자기 파일이 있으면 생략)
│   ├── fcf.save_sample_assumptions(path)           ── assumptions.xlsx
│   ├── fcf.save_sample_policies(path)              ── policies.csv
│   ├── fcf.save_sample_coverages(path)             ── coverages.csv
│   ├── fcf.save_sample_calculation_methods(path)   ── calculation_methods.csv
│   └── fcf.save_sample_inforce_policies(path)      ── 결산 1-파일 (spec + state)
│
├── 파일 읽어 들이기
│   ├── fcf.read_assumptions(path)                  ── basis dict 반환
│   ├── fcf.read_model_points(path, coverages=, ...) ── 신계약 평가용
│   └── fcf.read_inforce_policies(path, coverages=, ...) ── 결산 1-파일 reader
│
├── 평가
│   ├── fcf.measure(mp, asmp)                       ── 신계약, 시간 trajectory 전체
│   ├── fcf.value(mp, asmp)                         ── 신계약, 시점 0 의 4 숫자만 (빠름)
│   ├── fcf.value_segmented(mp, basis)              ── (product, channel) 자동 라우팅
│   ├── fcf.measure_in_force(mp, asmp, ...)         ── 결산, trajectory
│   └── fcf.value_in_force(mp, asmp, ...)           ── 결산, 시점 0
│
├── 결과 저장
│   ├── fcf.write_valuation(val, path)              ── 결과 한 파일에 저장
│   └── fcf.value_file(parquet, out_dir, asmp)      ── 메모리 초과 portfolio 스트리밍
│
├── 변동분해 (분기간 비교)
│   ├── fcf.roll_forward(m, period_months=...)      ── 분기 사이 변동 분해
│   └── fcf.reconcile(movements)                    ── 분해 결과를 항별로 합산
│
└── 검증 / 시각화
    ├── fcf.show_trace(mp_id, mp, basis)            ── 한 계약의 BEL 계산 ASCII 트리
    ├── fcf.show_bel_step(mp_id, mp, basis, ...)    ── 월별 BEL 식 전개
    ├── fcf.show_csm_step(mp_id, mp, basis, ...)    ── 월별 CSM 식 전개
    └── fcf.plot_liability(m) / plot_cashflows(m) / plot_csm_runoff(m) ...
```

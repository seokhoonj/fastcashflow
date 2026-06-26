# 7.1 워크북 — 단일 segment

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 엔진은 `Basis` 와 `ModelPoints` 두 **개체** 만 받고, **입력 파일은
  reader 함수가 그 두 개체로 모은다**
- 네 입력 파일 — `policies` / `coverages` / `calculation_methods` /
  `basis.xlsx`
- **`basis.xlsx` 의 매 시트 / 매 컬럼** — 가정을 회사 워크북으로 주는 자리
- `_DEFAULTS` 행으로 공통값을 한 번만 적고 segment 가 덮어쓰는 패턴
- rate 테이블의 **축 자동 감지** (sex / age / issue_age+duration / ...)
- `samples.export` 로 견본을 만들고 `read_*` 로 읽어 평가까지 돌리는 라운드트립
:::

지금까지의 챕터는 가정을 Python 코드 (`fcf.Basis(...)`) 로 직접
지었습니다. 실무에서는 가정이 **회사 워크북** 에 있고, 그 워크북을 엔진이 읽는
형식으로 맞추는 것이 진입점입니다. 이 챕터는 그 워크북 — `basis.xlsx` —
의 구조를 시트 단위로 풉니다.

## 엔진이 받는 것은 파일이 아니라 개체

`measure(mp, basis)` 의 두 인자는 **개체** (`ModelPoints`, `Basis`) 이지
파일이 아닙니다. 파일은 reader 가 개체로 바꿔 줍니다:

- **`Basis`** — `basis = fcf.read_basis("basis.xlsx")` 가 한
  워크북을 읽어 **`(product, channel) → Basis` 사전** 을
  돌려줍니다 (segment 별 가정 한 벌씩).
- **`ModelPoints`** — `mp = fcf.read_model_points("policies.csv",
  coverages=..., calculation_methods=...)` 가 세 파일을 한 개체로 합칩니다.

네 입력 파일의 전체 트리와 사용자 함수 지도는 [1.1 한눈에 보기](../basics/overview)
에 있습니다. 이 챕터는 그중 **`basis.xlsx`** 에 집중합니다.

## basis.xlsx — 시트 구성

워크북은 한 시트가 한 가지 역할을 맡는 multi-sheet 파일입니다. 견본
(`samples.export` 가 떨구는 `basis.xlsx`) 의 시트는 다음과 같습니다:

:::{list-table}
:header-rows: 1
:widths: 26 14 60

* - 시트
  - 필수
  - 역할
* - `segments`
  - 필수
  - `(product, channel)` 마다 **어느 테이블을 쓸지** 와 스칼라 가정
* - `coverages`
  - 필수
  - 담보 코드 → 어느 위험률 테이블 (`rate_table`) 을 쓸지
* - `mortality_tables`
  - 필수
  - 사망률. `table_id` 별 grid
* - `incidence_rate_tables`
  - 선택
  - 진단 / 입원 등 rate 기반 담보의 발생률
* - `waiver_tables`
  - 선택
  - 납입면제 / 장해 발생률 (active → waiver 전이)
* - `lapse_tables`
  - 필수
  - 해지율
* - `discount_tables`
  - 필수
  - 할인율 (`table_id` × `year`)
* - `surrender_value_tables`
  - 선택
  - 해약환급금 곡선 (`duration_month` × `factor` 또는 `amount`; segments 의
    `surrender_value_basis` 가 해석 결정)
* - `expense_tables`
  - 선택
  - 사업비 항목 ledger (`category` × `base` × `value`)
* - `inflation_tables`
  - 선택
  - 사업비 인플레이션 (`table_id` × `year`)
:::

(reader 는 `ae_factors` / `improvement_tables` 시트도 선택적으로 읽습니다 —
A/E 보정과 사망률 개선. 견본에는 없습니다.)

### `segments` 시트 — 어느 테이블을 쓸지

한 행이 한 segment `(product, channel)` 입니다. 컬럼은 세 부류:

- **식별 키** — `product` / `channel`. 이 쌍이 segment 를
  식별하고 모델포인트를 라우팅하는 **계산용** 키입니다. 옆에 둘 수 있는
  `product_name` / `channel_name` 은 **보고서용** 표시 라벨로, reader 가
  읽되 매칭엔 쓰지 않습니다 (담보의 `coverage` / `coverage_name` 과
  같은 code/name 관례 — code 는 계산, name 은 사람이 읽는 라벨).
- **테이블 참조** — 값이 rate 시트의 `table_id` 를 가리킵니다:
  `mortality_table` / `lapse_table` / `discount_table` (필수),
  `waiver_table` / `inflation_table` / `surrender_value_table` /
  `expense_table` / `mortality_improvement_table` (선택).
- **스칼라 가정** — `ra_confidence` / `mortality_cv` (필수),
  `morbidity_cv` / `disability_cv` / `longevity_cv` / `expense_cv` /
  `cost_of_capital_rate` / `state_model` / `investment_return` /
  `fund_fee` 등 (선택), `mortality_age_shift` 등 연령 shift (선택).

:::{admonition} `_DEFAULTS` 행 — 공통값을 한 번만
:class: note

`product` 가 `_DEFAULTS` 인 첫 행은 **다른 행의 빈 칸을 채우는 기본값**
입니다. 견본은 `_DEFAULTS` 에 `MORTALITY_STD` / `DISCOUNT_STD` /
`state_model=WAIVER` / `ra_confidence=0.75` 등을 두고, 각 segment 행은
**다른 부분만** 덮어씁니다 — 예컨대 `lapse_table` 과 `expense_table` 은
상품×채널별로 (`LAPSE_TERM_FC` / `EXPENSE_TERM_FC` 등). 같은 값을 모든 행에
반복하지 않아도 됩니다.
:::

### rate 시트의 축 자동 감지

`mortality_tables` / `incidence_rate_tables` / `waiver_tables` /
`lapse_tables` 는 모두 `table_id` + `rate` 를 필수로 갖고, **있는 축 컬럼을
reader 가 자동 감지** 합니다:

- `sex` (0=남, 1=여)
- `age` (도달연령) — 또는 `issue_age` + `duration` (select-and-ultimate)
- `issue_class` (인수등급), `elapsed` (Semi-Markov 경과) — 선택

한 `table_id` 는 가진 축 위에서 **빈틈 없는 grid** 를 이뤄야 합니다 (reader 가
검사). 없는 축은 그 값으로 평탄하게 broadcast 됩니다. `discount_tables` /
`inflation_tables` 는 `table_id` × `year` × `rate`,
`surrender_value_tables` 는 `table_id` × `duration_month` × `factor` (또는
`amount`; segments 의 `surrender_value_basis` 가 해석),
`expense_tables` 는 `table_id` × `category` × `base` × `value` 입니다.

## 작동 예제 — 견본을 만들고 읽어 평가

자기 워크북이 아직 없으면 `samples.export` 로 견본 파일을 만들어 형식을
눈으로 확인할 수 있습니다. 아래는 견본을 임시 폴더에 떨어뜨리고, 읽어
들이고, 한 segment 의 가정 개체를 들여다본 뒤 평가까지 가는 전체 흐름입니다.

```python
import tempfile
from pathlib import Path
import fastcashflow as fcf

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # 1) 견본 네 파일을 폴더에 생성 (자기 파일이 있으면 이 블록은 생략)
    fcf.samples.export(tmp, template="gmm", quiet=True)  # basis.xlsx + policies/coverages/calculation_methods

    # 2) 워크북을 읽으면 (product, channel) → Basis 사전
    basis = fcf.read_basis(tmp / "basis.xlsx")
    print("segments =", sorted(basis.segments))

    # 3) 한 segment 의 가정 개체를 꺼내 본다
    asmp = basis.resolve(("TERM_LIFE_A", "FC"))
    print("ra_confidence   =", asmp.ra_confidence)
    print("mortality_cv    =", asmp.mortality_cv)
    print("discount_annual =", asmp.discount_annual[:4].round(5), "...",
          f"(len {len(asmp.discount_annual)})")

    # 4) 모델포인트 = 세 파일을 한 개체로
    mp = fcf.read_model_points(tmp / "policies.csv", coverages=tmp / "coverages.csv",
                               calculation_methods=tmp / "calculation_methods.csv")
    print("n model points  =", mp.issue_age.shape[0])

    # 5) 평가 -- 각 계약을 자기 (product, channel) 가정으로 라우팅 (6.2 에서 자세히)
    val = fcf.gmm.measure(mp, basis, full=False)
    print(f"BEL sum = {val.bel.sum():>12,.0f}")
    print(f"CSM sum = {val.csm.sum():>12,.0f}")
```

출력:

```
segments = [('HEALTH_A', 'FC'), ('HEALTH_A', 'GA'), ('HEALTH_A', 'TM'), ('TERM_LIFE_A', 'FC'), ('TERM_LIFE_A', 'GA'), ('WHOLE_LIFE_A', 'FC'), ('WHOLE_LIFE_A', 'GA')]
ra_confidence   = 0.75
mortality_cv    = 0.1
discount_annual = [0.03103 0.03103 0.03999 0.03947] ... (len 101)
n model points  = 11
BEL sum =  -10,182,300
CSM sum =   10,280,704
```

- `read_basis` 는 **사전** 을 돌려줍니다 — 견본은 7 개 segment. 단일
  segment 워크북이면 행이 하나뿐이고 사전 키도 하나입니다.
- `basis.resolve(("TERM_LIFE_A", "FC"))` 가 그 segment 의 `Basis` 개체입니다.
  `ra_confidence` 0.75 / `state_model` = WAIVER 는 `_DEFAULTS` 행에서,
  `lapse_table` = `LAPSE_TERM_FC` 는 segment 행에서 온 값입니다.
- `discount_annual` 이 길이 101 배열인 것은 견본 `discount_tables` 가 국고채
  현물금리에서 만든 **연도별 할인 곡선** (관찰 + 보간 + LTFR 수렴) 을 담고
  있기 때문입니다 — `year` 별 한 행씩, 그 **전체 연도별 곡선** 이 그대로
  들어옵니다 (한 행만 있으면 그 값이 평탄 적용).

:::{admonition} 단일 가정 적용 vs segment 별 라우팅
:class: note

`fcf.gmm.measure(mp, basis)` 에 **단일 `Basis`** 를 주면 그 한 가정을 모든
모델포인트에 적용합니다 — 모델포인트가 동질한 한 segment 일 때 맞습니다.
견본처럼 여러 segment 가 섞인 portfolio 는 **BasisRouter**
(`{(product, channel): Basis}`) 를 주면 각 계약을 자기 segment 가정으로
라우팅합니다: `fcf.gmm.measure(mp, basis, full=False)` (BasisRouter 는
`full=False` headline 도, `full=True` 궤적도 둘 다 라우팅합니다). 라우팅
메커니즘은 [7.2](workbook-multi).
:::

## 함정

### 함정 1 — `basis` 는 반드시 `.xlsx`

`basis` 는 multi-sheet 워크북이라 `.csv` 로 줄 수 없습니다. 반면
`policies` / `coverages` / `calculation_methods` 는 단일 표라 `.csv` /
`.parquet` / `.feather` / `.xlsx` 어느 형식이든 됩니다 (reader 가 확장자로
감지). 대형 portfolio 는 `.parquet` 가 빠릅니다.

### 함정 2 — `segments` 의 `table_id` 가 rate 시트와 안 맞음

`segments.mortality_table` 의 값은 `mortality_tables` 시트의 `table_id` 와
**정확히** 일치해야 합니다. 오타 / 대소문자 불일치면 그 테이블을 못 찾아
에러가 납니다. `_DEFAULTS` 가 채우는 칸도 마찬가지입니다.

### 함정 3 — rate 테이블에 grid 빈틈

한 `table_id` 가 `sex` × `age` 축을 가지면 모든 (sex, age) 조합 행이 있어야
합니다. 한 칸이라도 비면 reader 가 grid 불완전으로 거부합니다 — 빠진 조합을
채우거나, 그 축을 빼서 평탄 broadcast 로 두세요.

## 인접 레시피

- [1.1 한눈에 보기](../basics/overview) — 네 입력 파일과 사용자 API 의 전체 트리.
- [1.2 담보와 산출방법 매칭](../basics/calculation-methods) —
  `calculation_methods.csv` 의 5 종 산출방법.
- [7.2 워크북 — 다중 segment / 다종 상품](workbook-multi) — `measure`
  라우팅과 segment 별 다른 StateModel / lapse.
- [튜토리얼 11장](../../tutorial/11-in-practice) — 결산 워크플로와 보유계약
  입력 (`read_inforce_policies`).

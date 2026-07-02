# 9.5 샘플 워크북으로 결산하기 -- 입력 시트에서 결산파일까지

다른 결산 챕터([9.1 정산](settlement), [9.4 결산팩](close-pack)) 가 코드 흐름을
설명한다면, 이 챕터는 **번들 샘플 엑셀을 실제로 열어** 시트·컬럼을 하나하나
짚으며 그것이 **결산파일(.xlsx)** 까지 어떻게 흘러가는지 끝까지 추적합니다.
샘플은 합성 데이터라 마음껏 열어봐도 안전합니다.

:::{admonition} 그림은 샘플 데이터에서 자동 렌더됩니다
:class: note

아래 그림은 **번들 샘플을 그대로 옮긴 렌더** 입니다 (placeholder 가 아님) --
값·`_DEFAULTS` 행·열머리글까지 실제 `samples/` 산출물 그대로입니다. 샘플
데이터나 결산팩 레이아웃이 바뀌면 한 번에 다시 그립니다:

    .venv/bin/python docs/generate_sheet_images.py
:::

## 0. 샘플 파일 떨구기

`samples.export` 가 한 세트를 디스크에 떨굽니다 -- 가정 워크북 `basis.xlsx`
한 권과, 계약 파일 `policies` / `coverages` / `calculation_methods` /
`inforce_state` (그리고 spec + 결산상태를 합친 1-파일 `inforce_policies`).

```python
import fastcashflow as fcf
import numpy as np

fcf.samples.export("samples", template="gmm", quiet=True)
```

## 1. `basis.xlsx` -- `segments` 시트: 어느 테이블을 쓸지

:::{figure} ../../images/sample-basis-segments.png
:width: 95%
:alt: basis.xlsx segments 시트

`segments` -- `(product, channel)` 한 행이 한 보험계약집합의 가정을 묶는다.
:::

`segments` 는 **어느 rate 테이블과 스칼라 가정을 쓸지**를 `(product, channel)`
별로 가리킵니다. 컬럼을 하나씩 보면:

- `product` / `channel` -- 라우팅 키. `measure` / `settle` 가 모델포인트의
  `(product, channel)` 로 이 행을 찾습니다.
- `mortality_table` / `lapse_table` / `waiver_table` / `discount_table` /
  `inflation_table` / `surrender_value_table` / `expense_table` -- 각 rate 시트의
  `table_id` 참조 (실제 숫자는 그 시트에).
- `ra_confidence` / `mortality_cv` / `morbidity_cv` -- 위험조정 스칼라.
- `state_machine` -- 상태기계 (`Model.from_preset` 키, 없으면 단일상태).
- **`_DEFAULTS` 행** -- 모든 세그먼트의 공통 기본값. 각 세그먼트 행은 다른
  칸을 비워 두고 (`None`) **자기에게 다른 것만 override** 합니다 -- 예:
  `TERM_LIFE_A / FC` 는 `lapse_table` / `surrender_value_table` / `expense_table`
  만 채우고 사망률·할인·RA 는 `_DEFAULTS` 를 물려받습니다.

→ 흐름: 이 한 행이 `read_basis` 에서 **한 `Basis` 로 resolve** 되어, 그 세그먼트의
모든 모델포인트 측정에 쓰입니다.

## 2. `basis.xlsx` -- `mortality_tables` (위험률 시트의 예)

:::{figure} ../../images/sample-basis-mortality.png
:width: 95%
:alt: basis.xlsx mortality_tables 시트

`mortality_tables` -- `table_id` 별 성별·연령 grid.
:::

rate 시트는 `table_id` 로 묶인 grid 입니다. 사망률은 `(table_id, sex, age) ->
rate`: `MORTALITY_STD`, `sex=0`(남), `age=20` 의 연 사망률이 `0.00038` 식.
`segments` 의 `mortality_table = MORTALITY_STD` 가 이 묶음을 가리킵니다.
(`lapse_tables` / `discount_tables` 등 다른 시트도 같은 `table_id` x 축 구조 --
형식 상세는 [7.1 워크북](../io/workbook-single).)

## 3. `basis.xlsx` -- `coverages` 시트: 담보 -> 위험률 매핑

:::{figure} ../../images/sample-basis-coverages.png
:width: 95%
:alt: basis.xlsx coverages 시트

`coverages` -- 담보 코드가 어느 발생률 테이블을 쓰는지.
:::

`coverage -> rate_table`: `DEATH -> MORTALITY_STD`, `INPATIENT -> INPATIENT_STD`,
`CANCER -> CANCER_STD`. 담보별로 어느 발생률을 적용할지 정합니다 (담보가 어느
**산출방법**(DEATH/MORBIDITY/DIAGNOSIS...)인지는 `calculation_methods` 가 따로
정함 -- [1.2](../basics/calculation-methods)).

## 4. 계약 파일 -- `policies` / `coverages` / `inforce_state`

:::{figure} ../../images/sample-policies.png
:width: 95%
:alt: policies.csv

`policies` -- 계약의 영구 spec (한 행 = 한 모델포인트).
:::

`policies` 는 계약 spec: `mp_id` / `product` / `channel` / `issue_date` /
`issue_age` / `sex` / `term_months` / `premium_term_months` / `count`.
`product` / `channel` 이 `segments` 와 맞물려 가정이 라우팅됩니다.

:::{figure} ../../images/sample-coverages.png
:width: 95%
:alt: coverages.csv

`coverages` -- 모델포인트마다 어떤 담보를, 얼마(`amount`)에, 보험료(`premium`) 얼마로.
:::

`coverages` 는 `mp_id` 별 담보 줄: `P001` 은 `DEATH` 8,000만 (보험료 28,216) +
`MATURITY` 1,000만 (11,286). 한 계약이 여러 담보를 가집니다.

:::{figure} ../../images/sample-inforce-state.png
:width: 95%
:alt: inforce_state.csv

`inforce_state` -- 결산일의 상태 (경과월수 · 잔존 · 직전 CSM · lock-in).
:::

`inforce_state` 가 **신계약과 결산을 가르는 입력**입니다: `elapsed_months`
(가입 후 경과), `count` (결산일 잔존), `prior_csm` (직전 분기 종가 CSM),
`lock_in_rate` (가입 시 할인율), `prior_count` (기초 잔존). 정책관리 시스템이
매 분기말 떨어뜨리는 **보유계약 마감파일**의 상태 컬럼이 이것입니다.

## 5. 흐름 실행 -- 입력에서 결산파일까지

이제 위 파일들을 읽어 **보험계약집합 단위로 정산**하고 결산팩으로 모읍니다.
`settle_group_of_contracts` 한 번이 세그먼트 라우팅과 (상품 x 연도코호트 x
수익성) 그룹핑·정산을 모두 처리합니다 -- per-세그먼트 루프를 직접 돌 필요가
없습니다.

```python
basis = fcf.read_basis("samples/basis.xlsx")                       # segments -> BasisRouter
model_points, state = fcf.read_inforce_policies(                   # spec + 결산상태
    "samples/inforce_policies.csv",
    coverages="samples/coverages.csv",
    calculation_methods="samples/calculation_methods.csv",
)

# profitability: 보험계약집합의 수익성 축 (문단 16, inception 동결 문단 24).
# 샘플엔 없으니 inception onerous 검사 (loss_component > 0) 로 도출 --
# 엔진의 group_of_contracts 기본 분류와 같다.
profitability = np.where(
    fcf.gmm.measure(model_points, basis).loss_component > 0.0,
    "onerous", "remaining")

# settle_group_of_contracts 한 번 = 세그먼트 라우팅 + (상품 x 연도코호트 x
# 수익성) 그룹핑 + 보험계약집합별 정산 (문단 44, 그룹 내 floor 상계 포함).
goc = fcf.settle_group_of_contracts(
    model_points, state, basis,
    period_months=12,        # 문단 44 기초 -> 기말
    coverage_units="count",  # CSM 배분 단위 (보장수량)
    profitability=profitability,
)
pack = fcf.close([fcf.reconcile(goc)])                             # 공시 명세서 조립
print(pack)
```

출력:

```
IFRS 17 close pack -- 12-month period
  Net statement of financial position
    LRC excluding loss component           1,575,567       8,888,116      10,463,683
    Loss component                                 0       2,764,609       2,764,609
    Liability for incurred claims                  0               0               0
    Total                                  1,575,567      11,652,725      13,228,292
```

세 컬럼은 **기초 | 변동 | 기말** 입니다 (`기초 + 변동 = 기말`). 9개 보험계약집합
(상품 x 연도코호트 x 수익성) 은 합산돼 **회사 전체 SoFP 한 장**으로 나오고, 집합별
상세는 그 옆에 함께 떨어지는 **별도 parquet 파일**에 남습니다. 그룹 내 floor 상계
(profitable 이 onerous 손실을 흡수) 덕분에 손실요소가 per-세그먼트 합산보다 작게
잡힙니다.

## 6. 산출 -- 결산팩 엑셀

```python
fcf.write_close_pack(pack, "samples/close_pack_2026Q1.xlsx", movements=[goc])
```

`close_pack_2026Q1.xlsx` 한 권은 **회사 전체로 집계된 명세서 시트들**과, 그 옆에
함께 떨어지는 **보험계약집합별 상세 parquet 파일**로 이뤄집니다. 시트를 하나씩
보면:

:::{figure} ../../images/close-pack-00-index.png
:width: 70%
:alt: 결산팩 00_Index 시트

`00_Index` -- 표지: 보고기간, 담긴 모델 / 그룹, 시트 목록, per-MP 상세 파일 참조.
:::

:::{figure} ../../images/close-pack-01-sofp.png
:width: 95%
:alt: 결산팩 01_SoFP 시트

`01_SoFP` -- 재무상태표의 보험계약부채: LRC (잔여보장, 손실요소 분리) + LIC
(발생사고). 세 컬럼은 **기초 | 변동 | 기말**. 9개 보험계약집합이 회사 전체 한 장으로
합산됩니다.
:::

:::{figure} ../../images/close-pack-03-finance.png
:width: 80%
:alt: 결산팩 03_Finance 시트

`03_Finance` -- 보험금융손익 (문단 B72): BEL / RA / CSM / LIC 이자부리 +
finance wedge, 발행계약 · 보유재보험 · 순액 세 묶음.
:::

:::{figure} ../../images/close-pack-04-reconciliation.png
:width: 95%
:alt: 결산팩 04_Reconciliation 시트 (상위 행)

`04_Reconciliation` -- 변동분석표의 tidy 상세 (감사조인용). 한 행 = 한 라인, 각
라인이 `line_code` 와 **IFRS 17 문단 앵커** (`100(a)`, `B72(a)`, `B123` ...) 를
달고 나옵니다. (상위 행만 -- 전체는 35행.)
:::

:::{figure} ../../images/close-pack-sidecar.png
:width: 95%
:alt: per-MP parquet 상세 파일 (선택 컬럼)

`close_pack_2026Q1_per_mp_0.parquet` -- 보험계약집합 (9개) 단위 상세. 그룹별
BEL / RA / CSM 의 개시·종가가 그대로 남습니다 (38컬럼 중 일부만 표시).
:::

입력이 어디로 갔는지 역추적하면:

- `segments` 행 -> `Basis` -> `settle_group_of_contracts` 의 가정
- `policies` / `coverages` -> 측정되는 계약과 현금흐름
- `inforce_state.prior_csm` / `lock_in_rate` -> CSM roll-forward 의 기초
- 보험계약집합 정산 -> `reconcile` -> `close` -> **01_SoFP / 03_Finance / 04_Reconciliation** 시트
- 보험계약집합 (9개) 단위 상세 -> 별도 parquet 파일

## 인접 레시피

- [7.1 워크북 -- 단일 segment](../io/workbook-single) -- `basis.xlsx` 의 매 시트 / 매 컬럼 형식 상세.
- [9.4 결산팩 -- 공시 명세서 조립](close-pack) -- `close` / `write_close_pack` / `line_metadata` 의 상세.
- [9.1 결산 / 보유계약 평가](settlement) -- `gmm.settle` 의 뼈대와 분기 체이닝.

# 9.5 샘플 워크북으로 결산하기 -- 입력 시트에서 결산파일까지

다른 결산 챕터([9.1 정산](settlement), [9.4 결산팩](close-pack)) 가 코드 흐름을
설명한다면, 이 챕터는 **번들 샘플 엑셀을 실제로 열어** 시트·컬럼을 하나하나
짚으며 그것이 **결산파일(.xlsx)** 까지 어떻게 흘러가는지 끝까지 추적합니다.
샘플은 합성 데이터라 마음껏 열어봐도 안전합니다.

```{admonition} 이미지는 캡쳐 자리입니다
:class: note

아래 그림은 **placeholder** 입니다 -- `samples/` 에 떨어진 실제 엑셀 파일을 열어
스크린샷으로 교체하세요. `docs/images/` 의 파일명을 그대로 두면 문서에 자동
반영됩니다 (예: `sample-basis-segments.png`).
```

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

```{figure} ../../images/sample-basis-segments.png
:width: 95%
:alt: basis.xlsx segments 시트

`segments` -- `(product, channel)` 한 행이 한 보험계약집합의 가정을 묶는다.
```

`segments` 는 **어느 rate 테이블과 스칼라 가정을 쓸지**를 `(product, channel)`
별로 가리킵니다. 컬럼을 하나씩 보면:

- `product` / `channel` -- 라우팅 키. `measure` / `settle` 가 모델포인트의
  `(product, channel)` 로 이 행을 찾습니다.
- `mortality_table` / `lapse_table` / `waiver_table` / `discount_table` /
  `inflation_table` / `surrender_value_table` / `expense_table` -- 각 rate 시트의
  `table_id` 참조 (실제 숫자는 그 시트에).
- `ra_confidence` / `mortality_cv` / `morbidity_cv` -- 위험조정 스칼라.
- `state_model` -- 상태기계 (`STATE_MODELS` 키, 없으면 단일상태).
- **`_DEFAULTS` 행** -- 모든 세그먼트의 공통 기본값. 각 세그먼트 행은 다른
  칸을 비워 두고 (`None`) **자기에게 다른 것만 override** 합니다 -- 예:
  `TERM_LIFE_A / FC` 는 `lapse_table` / `surrender_value_table` / `expense_table`
  만 채우고 사망률·할인·RA 는 `_DEFAULTS` 를 물려받습니다.

→ 흐름: 이 한 행이 `read_basis` 에서 **한 `Basis` 로 resolve** 되어, 그 세그먼트의
모든 모델포인트 측정에 쓰입니다.

## 2. `basis.xlsx` -- `mortality_tables` (위험률 시트의 예)

```{figure} ../../images/sample-basis-mortality.png
:width: 95%
:alt: basis.xlsx mortality_tables 시트

`mortality_tables` -- `table_id` 별 성별·연령 grid.
```

rate 시트는 `table_id` 로 묶인 grid 입니다. 사망률은 `(table_id, sex, age) ->
rate`: `MORTALITY_STD`, `sex=0`(남), `age=20` 의 연 사망률이 `0.00038` 식.
`segments` 의 `mortality_table = MORTALITY_STD` 가 이 묶음을 가리킵니다.
(`lapse_tables` / `discount_tables` 등 다른 시트도 같은 `table_id` x 축 구조 --
형식 상세는 [7.1 워크북](../io/workbook-single).)

## 3. `basis.xlsx` -- `coverages` 시트: 담보 -> 위험률 매핑

```{figure} ../../images/sample-basis-coverages.png
:width: 95%
:alt: basis.xlsx coverages 시트

`coverages` -- 담보 코드가 어느 발생률 테이블을 쓰는지.
```

`coverage -> rate_table`: `DEATH -> MORTALITY_STD`, `INPATIENT -> INPATIENT_STD`,
`CANCER -> CANCER_STD`. 담보별로 어느 발생률을 적용할지 정합니다 (담보가 어느
**산출방법**(DEATH/MORBIDITY/DIAGNOSIS...)인지는 `calculation_methods` 가 따로
정함 -- [1.2](../basics/calculation-methods)).

## 4. 계약 파일 -- `policies` / `coverages` / `inforce_state`

```{figure} ../../images/sample-policies.png
:width: 95%
:alt: policies.csv

`policies` -- 계약의 영구 spec (한 행 = 한 모델포인트).
```

`policies` 는 계약 spec: `mp_id` / `product` / `channel` / `issue_date` /
`issue_age` / `sex` / `term_months` / `premium_term_months` / `count`.
`product` / `channel` 이 `segments` 와 맞물려 가정이 라우팅됩니다.

```{figure} ../../images/sample-coverages.png
:width: 95%
:alt: coverages.csv

`coverages` -- 모델포인트마다 어떤 담보를, 얼마(`amount`)에, 보험료(`premium`) 얼마로.
```

`coverages` 는 `mp_id` 별 담보 줄: `P001` 은 `DEATH` 8,000만 (보험료 28,216) +
`MATURITY` 1,000만 (11,286). 한 계약이 여러 담보를 가집니다.

```{figure} ../../images/sample-inforce-state.png
:width: 95%
:alt: inforce_state.csv

`inforce_state` -- 결산일의 상태 (경과월수 · 잔존 · 직전 CSM · lock-in).
```

`inforce_state` 가 **신계약과 결산을 가르는 입력**입니다: `elapsed_months`
(가입 후 경과), `count` (결산일 잔존), `prior_csm` (직전 분기 종가 CSM),
`lock_in_rate` (가입 시 할인율), `prior_count` (기초 잔존). 정책관리 시스템이
매 분기말 떨어뜨리는 **보유계약 마감파일**의 상태 컬럼이 이것입니다.

## 5. 흐름 실행 -- 입력에서 결산파일까지

이제 위 파일들을 읽어 세그먼트별로 정산하고 결산팩으로 모읍니다.

```python
basis = fcf.read_basis("samples/basis.xlsx")                       # segments -> BasisRouter
model_points, state = fcf.read_inforce_policies(                   # spec + 결산상태
    "samples/inforce_policies.csv",
    coverages="samples/coverages.csv",
    calculation_methods="samples/calculation_methods.csv",
)

movements, recons, group_labels = [], [], []
for key, seg_basis in basis.segments.items():                      # settle 은 세그먼트 단위
    idx = np.where((np.asarray(model_points.product) == key[0]) &
                   (np.asarray(model_points.channel) == key[1]))[0]
    if len(idx) == 0:
        continue
    mv = fcf.gmm.settle(model_points.subset(idx), state.subset(idx),
                        seg_basis, period_months=12)               # Sec. 44 기초 -> 기말
    movements.append(mv)
    recons.append(fcf.reconcile([mv])[0])                          # 보험계약집합 정산표
    group_labels.append("/".join(key))

pack = fcf.close(recons, group_ids=group_labels)                   # 공시 명세서 조립
print(pack)
```

출력:

```
IFRS 17 close pack -- 12-month period
  Net statement of financial position
    LRC excluding loss component           1,575,567       8,871,199      10,446,766
    Loss component                                 0       2,879,823       2,879,823
    Liability for incurred claims                  0               0               0
    Total                                  1,575,567      11,751,022      13,326,589
```

## 6. 산출 -- 결산팩 엑셀

```python
fcf.write_close_pack(pack, "samples/close_pack_2026Q1.xlsx", movements=movements)
```

```{figure} ../../images/close-pack-output.png
:width: 95%
:alt: 결산팩 출력 엑셀

`close_pack_2026Q1.xlsx` -- 00_Index · 01_SoFP · 03_Finance · 04_Reconciliation
시트 (+ 모델포인트 상세는 옆에 parquet 사이드카).
```

입력이 어디로 갔는지 역추적하면:

- `segments` 행 -> `Basis` -> `gmm.settle` 의 가정
- `policies` / `coverages` -> 측정되는 계약과 현금흐름
- `inforce_state.prior_csm` / `lock_in_rate` -> CSM roll-forward 의 기초
- 세그먼트별 정산표 -> `close` -> **01_SoFP / 03_Finance / 04_Reconciliation** 시트
- 모델포인트 단위 movement -> parquet 사이드카

## 인접 레시피

- [7.1 워크북 -- 단일 segment](../io/workbook-single) -- `basis.xlsx` 의 매 시트 / 매 컬럼 형식 상세.
- [9.4 결산팩 -- 공시 명세서 조립](close-pack) -- `close` / `write_close_pack` / `line_metadata` 의 상세.
- [9.1 결산 / 보유계약 평가](settlement) -- `gmm.settle` 의 뼈대와 분기 체이닝.

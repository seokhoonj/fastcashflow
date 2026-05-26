# 11장. 실무에서의 활용 (1)

```{admonition} 이 장에서 배우는 것
:class: tip

- 모델포인트 파일과 가정 파일의 구조
- 파일을 읽어 평가하고 결과를 저장하기
- 메모리를 넘는 대규모 포트폴리오
```

8장에서 엔진으로 일반모형을 측정하고, 9~10장에서 PAA와 VFA까지
봤습니다. 모두 입력을 코드로 만들거나 샘플 데이터로 불러와 돌린
것이었죠. 실제 업무에서는 자사의 계약과 가정을 다뤄야 하고,
그것은 보통 엑셀·CSV 파일로 들어옵니다. 이 장은 그 파일들의 구조와,
파일을 읽어 평가하는 흐름을 다룹니다.

## 11.1 입력 파일의 구조

입력은 두 갈래 — 모델포인트와 계리적 가정입니다. 파일을 보기 전에,
담보가 어떻게 구성되는지부터 짚습니다.

### 담보 구조

계약 하나는 **주계약과 특약 여러 개**로 이뤄집니다. 엔진은 이 전부를
**담보**의 목록으로 다룹니다 — 주계약 사망도, 특약도 모두 담보
하나씩입니다.

담보마다 **`coverage_code`**(특약코드)가 붙습니다. 이 코드가 가정 파일의
`coverages`·`rates` 시트와 맞물려 그 담보의 유형과 위험률을 끌어옵니다.
유형(`type`)은 엔진이 그 담보를 어떻게 계산할지를 정합니다.

```{list-table}
:header-rows: 1
:widths: 22 78

* - `benefit_pattern`
  - 엔진의 동작
* - `DEATH`
  - 사망형 담보 (일반사망 / 상해사망 / 재해사망 / ADB 등). 자기 위험률로 지급, 보유계약은 그대로 (in-force 감쇠는 별도 `mortality_annual` 입력이 담당)
* - `MORBIDITY`
  - 입원·수술 등 반복지급 담보. 매번 지급하고 계약은 유지
* - `DIAGNOSIS`
  - 진단 등 1회 지급 담보. 미진단 풀에서 차감
* - `ANNUITY`
  - 매월 지급하는 생존급부
* - `MATURITY`
  - 보험기간 끝에 지급하는 생존급부
```

실무에서는 담보코드와 위험률코드가 별개입니다 — 한 위험률을 여러
담보가 나눠 쓰기도 하니까요. fastcashflow의 `coverage_code`는 그 둘을
미리 맺어 둔 키입니다. 담보와 위험률을 잇는 작업은 입력 파일을 만들기
전에 끝내고, 엔진에는 담보마다 위험률이 정해진 상태로 들어옵니다.

### 모델포인트 — 계약 파일과 담보 파일

모델포인트는 **long-form**, 두 개의 파일로 나뉩니다 — 계약 자체의
속성을 담는 **계약 파일**과, 계약이 가진 담보를 담는 **담보
파일**입니다. 두 파일 모두 CSV이거나 parquet(파케이 — Hadoop
생태계에서 나온 컬럼 단위 데이터 파일 형식으로, CSV보다 빠르고
작습니다)입니다.

계약 파일은 한 줄이 한 계약입니다.

```{list-table}
:header-rows: 1
:widths: 30 70

* - 열 이름
  - 뜻
* - `mp_id`
  - 계약 식별자
* - `product`
  - 상품명
* - `issue_age`
  - 가입연령
* - `sex`
  - 성별 (0 남, 1 여)
* - `term_months`
  - 보험기간 (개월)
* - `count`
  - 이 줄이 대표하는 계약 수 (없으면 1)
```

담보 파일은 한 줄이 한 담보입니다. 주계약 사망도, 특약도 모두 한
줄씩이고 `mp_id`로 계약 파일과 묶입니다. 계약이 가진 담보 수만큼
줄이 생깁니다.

```{list-table}
:header-rows: 1
:widths: 30 70

* - 열 이름
  - 뜻
* - `mp_id`
  - 어느 계약의 담보인지
* - `coverage_code`
  - 특약코드 (가정의 `coverages` 시트와 맞물림)
* - `amount`
  - 가입금액
* - `premium`
  - 월 보험료
```

샘플 파일을 보면 — 계약 파일:

| mp_id | product | issue_age | sex | term_months | count |
|---|---|---|---|---|---|
| P001 | TERM_LIFE | 35 | 0 | 240 | 1 |
| P002 | HEALTH | 38 | 1 | 240 | 1 |

담보 파일:

| mp_id | coverage_code | amount | premium |
|---|---|---|---|
| P001 | DEATH_GENERAL | 80000000 | 45000 |
| P001 | MATURITY | 10000000 | 18000 |
| P002 | DEATH_GENERAL | 50000000 | 28000 |
| P002 | CANCER | 30000000 | 22000 |
| P002 | INPATIENT | 1000000 | 9000 |

P001은 담보 파일에 두 줄 — 주계약 사망(`DEATH_GENERAL`)과 생존특약(`MATURITY`).
P002는 세 줄입니다. 계약마다 담보 수가 다르니 줄 수도 다릅니다.

담보에 면책기간이나 감액기간이 있으면 담보 파일에 `waiting`(면책 개월
수)·`reduction_end`(감액 종료 시점)·`reduction_factor`(감액률) 열을
더합니다. 없는 담보는 비워 두면 됩니다.

계약 한 건이 한 줄로 끝나는 **wide** 형식도 받습니다. 담보를 열로
펼친 것이라 단일 상품의 작은 포트폴리오에 편하지만, 특약이 많은 이종
포트폴리오에는 long-form이 자연스럽습니다.
`ModelPoints.to_wide()`·`.to_long()`로 두 형식을 오갈 수 있습니다.

### 가정 파일 — 엑셀 워크북

가정 파일은 엑셀 워크북이고, 시트 다섯 개로 이뤄집니다. 각 시트가
실제로 어떻게 생겼는지 예시로 봅니다.

**`parameters`** — 할인율·사업비·위험조정 등 스칼라 가정을 이름과 값,
두 열로 적습니다.

| parameter | value |
|---|---|
| discount_annual | 0.03 |
| ra_confidence | 0.75 |
| mortality_cv | 0.10 |

**`mortality`** — 기저 사망률. 보유계약 감소와 주계약 사망에 쓰입니다.

| sex | age | rate |
|---|---|---|
| 0 | 40 | 0.0013 |
| 0 | 41 | 0.0014 |
| 1 | 40 | 0.0007 |

**`lapse`** — 경과연수별 연 해지율.

| duration | rate |
|---|---|
| 0 | 0.13 |
| 1 | 0.118 |
| 2 | 0.106 |

**`coverages`** — 담보 마스터. 담보 하나에 한 줄. `benefit_pattern`은 그 담보를
엔진이 어떻게 다룰지를 정하며, 값은 앞의 담보 구조 표를 따릅니다.

| product | coverage_code | coverage_name | benefit_pattern |
|---|---|---|---|
| - | DEATH_GENERAL | 일반사망 | DEATH |
| - | CANCER | 암진단특약 | DIAGNOSIS |
| - | INPATIENT | 입원특약 | MORBIDITY |

**`rates`** — 담보별 위험률. 위험률이 있는
담보(`DEATH`·`MORBIDITY`·`DIAGNOSIS`)만 들어갑니다. `ANNUITY`·`MATURITY`는
위험률이 없습니다.

| coverage_code | sex | age | rate |
|---|---|---|---|
| CANCER | 0 | 40 | 0.002 |
| CANCER | 1 | 40 | 0.0022 |
| INPATIENT | 0 | 40 | 0.08 |

8장에서 `lambda`로 적었던 사망률·위험률을, 여기서는 엑셀 표에 채워
넣습니다 — 사용자들에게 익숙한 방식이죠.

## 11.2 파일로 평가하기

파일이 준비됐으면 읽어서 평가하고, 결과를 저장합니다.

```python
import fastcashflow as fcf

basis        = fcf.read_assumptions("assumptions.xlsx")    # {(product, channel): Assumptions}
assumptions  = basis[("TERM_A", "GA")]                     # 한 세그먼트 선택
model_points = fcf.read_model_points("policies.csv", assumptions, coverages="coverages.csv")
val          = fcf.value(model_points, assumptions)
fcf.write_valuation(val, "results.csv")
```

- `read_assumptions` — 가정 엑셀을 읽어 `{(product, channel): Assumptions}`
  딕셔너리로 돌려줍니다. 한 워크북에 여러 세그먼트(상품 × 채널)를 함께
  관리하기 위함입니다. 한 세그먼트만 쓰려면 키로 골라냅니다.
- `read_model_points` — 계약 파일과 담보 파일을 읽어 모델포인트를
  만듭니다. 담보 파일의 특약코드를 가정에 등록된 특약과 맞춰야 하므로,
  가정(`assumptions`)을 함께 넘깁니다.
- `value` — 평가합니다.
- `write_valuation` — BEL·RA·CSM·손실요소를 모델포인트마다 한 줄씩
  파일로 저장합니다.

wide 한 파일이면 `coverages` 없이 `read_model_points("portfolio.csv",
assumptions)`로 읽습니다. 8장의 `load_sample_*`도 사실 이 `read_*`로 패키지
안의 샘플 파일을 읽는 것입니다. 자기 파일이 준비되면 경로만 바꾸면
됩니다.

## 11.3 메모리를 넘는 규모

포트폴리오가 너무 커서 메모리에 한꺼번에 올리기 어렵다면
`value_file()`을 씁니다. wide 형식의 parquet 파일을 조각조각 나눠
읽고, 평가하고, 결과를 쓰는 일을 흘려 가며 처리해, 메모리에는 한 번에
한 조각만 올립니다.

```python
fcf.value_file("portfolio.parquet", "results/", assumptions)
```

이 방식이면 포트폴리오 크기는 메모리가 아니라 디스크가 허락하는
만큼까지 늘어납니다.

## 11.4 다음 장

여기까지 입력 파일을 읽어 평가하고 저장하는 흐름을 봤습니다. 다음 장은
같은 측정 결과를 그림으로 보고, 기간별 변동을 분석하고, 손익 리포트로
정리하는 법을 다룹니다.

# 9장. 실제 업무에서의 활용

```{admonition} 이 장에서 배우는 것
:class: tip

- 모델포인트 파일과 가정 파일의 구조
- 파일을 읽어 평가하고 결과를 저장하기
- 메모리를 넘는 대규모 포트폴리오
- 튜토리얼을 마치며
```

8장에서는 입력을 코드로 만들거나 샘플 데이터로 불러와 엔진을
돌렸습니다. 실제 업무에서는 자기 회사의 계약과 가정을 다뤄야 하고,
그것은 보통 엑셀·CSV 파일로 들어옵니다. 이 장은 그 파일들의 구조와,
파일을 읽어 평가하는 흐름을 다룹니다.

## 9.1 입력 파일의 구조

입력은 두 갈래 — 모델포인트와 계리적 가정입니다.

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
* - `policy_id`
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
줄씩이고 `policy_id`로 계약 파일과 묶입니다. 계약이 가진 담보 수만큼
줄이 생깁니다.

```{list-table}
:header-rows: 1
:widths: 30 70

* - 열 이름
  - 뜻
* - `policy_id`
  - 어느 계약의 담보인지
* - `rider_code`
  - 특약코드 (가정의 `riders` 시트와 맞물림)
* - `amount`
  - 가입금액
* - `premium`
  - 월 보험료
```

샘플 파일을 보면 — 계약 파일:

```
policy_id,product,issue_age,sex,term_months,count
P001,term_life,35,0,240,1
P002,health,38,1,240,1
```

담보 파일:

```
policy_id,rider_code,amount,premium
P001,DTH_MAIN,80000000,45000
P001,MAT,10000000,18000
P002,DTH_MAIN,50000000,28000
P002,CANCER,30000000,22000
P002,HOSP,1000000,9000
```

P001은 담보 파일에 두 줄 — 주계약 사망(`DTH_MAIN`)과 생존특약(`MAT`).
P002는 세 줄입니다. 계약마다 담보 수가 다르니 줄 수도 다릅니다.

계약 한 건이 한 줄로 끝나는 **wide** 형식도 받습니다. 담보를 열로
펼친 것이라 단일 상품의 작은 포트폴리오에 편하지만, 특약이 많은 이종
포트폴리오에는 long-form이 자연스럽습니다.
`ModelPointSet.to_wide()`·`.to_long()`로 두 형식을 오갈 수 있습니다.

### 가정 파일 — 엑셀 워크북

가정 파일은 엑셀 워크북이고, 시트 다섯 개로 이뤄집니다.

- `parameters` — 이름과 값, 두 열. `discount_annual`, 사업비,
  위험조정 관련 값 등을 적습니다.
- `mortality` — 기저 사망률. `sex`, `age`(연령), `rate`(연 사망률)
  세 열의 long-form 표입니다. 보유계약 감소와 주계약 사망에 쓰입니다.
- `lapse` — 두 열. 경과연수와 연 해지율입니다.
- `riders` — 특약 마스터. `rider_code`, `rider_name`(특약명),
  `product`(상품), `type` 네 열이고 특약 하나에 한 줄씩입니다. `type`은
  엔진이 그 특약을 어떻게 다룰지를 정합니다 — `death_main`(주계약
  사망), `death`(사망형 특약), `morbidity`(입원·수술 등 반복지급),
  `diagnosis`(진단 1회 지급), `annuity`(월 생존급부), `maturity`(만기
  생존급부).
- `rates` — 특약별 위험률. `rider_code`, `sex`, `age`, `rate` 네 열의
  long-form 표로, 위험률이 있는 특약(`death`·`morbidity`·`diagnosis`)을
  담습니다. `annuity`·`maturity`는 위험률이 없어 여기 들어가지 않습니다.

8장에서 `lambda`로 적었던 사망률·위험률을, 여기서는 엑셀 표에 채워
넣습니다 — 계리사에게 익숙한 방식이죠.

## 9.2 파일로 평가하기

파일이 준비됐으면 읽어서 평가하고, 결과를 저장합니다.

```python
import fastcashflow as fcf

asmp = fcf.read_assumptions("basis.xlsx")
mps  = fcf.read_model_points("policies.csv", asmp, coverages="coverages.csv")
val  = fcf.value(mps, asmp)
fcf.write_valuation(val, "results.csv")
```

- `read_assumptions` — 가정 엑셀을 읽어 들입니다.
- `read_model_points` — 계약 파일과 담보 파일을 읽어 모델포인트를
  만듭니다. 담보 파일의 특약코드를 가정에 등록된 특약과 맞춰야 하므로,
  가정(`asmp`)을 함께 넘깁니다.
- `value` — 평가합니다.
- `write_valuation` — BEL·RA·CSM·손실요소를 모델포인트마다 한 줄씩
  파일로 저장합니다.

wide 한 파일이면 `coverages` 없이 `read_model_points("portfolio.csv",
asmp)`로 읽습니다. 8장의 `load_sample_*`도 사실 이 `read_*`로 패키지
안의 샘플 파일을 읽는 것입니다. 자기 파일이 준비되면 경로만 바꾸면
됩니다.

## 9.3 메모리를 넘는 규모

포트폴리오가 너무 커서 메모리에 한꺼번에 올리기 어렵다면
`value_file()`을 씁니다. wide 형식의 parquet 파일을 조각조각 나눠
읽고, 평가하고, 결과를 쓰는 일을 흘려 가며 처리해, 메모리에는 한 번에
한 조각만 올립니다.

```python
fcf.value_file("portfolio.parquet", "results/", asmp)
```

이 방식이면 포트폴리오 크기는 메모리가 아니라 디스크가 허락하는
만큼까지 늘어납니다.

## 9.4 마치며

여기까지가 튜토리얼입니다. 되짚어 보면:

- **1~2장** — IFRS 17이 무엇이고, 보험계약부채가 BEL·RA·CSM으로
  이뤄진다는 것
- **3~4장** — 모델포인트와 계리적 가정이라는 입력, 그리고 엔진이
  현금흐름을 만들어 내는 방식
- **5~7장** — BEL·RA·CSM을 차례로 계산하고 손으로 검증
- **8~9장** — 엔진으로 직접 측정하고, 실제 업무처럼 파일로 다루기

이제 `measure()` 한 줄 안에서 무엇이 어떤 순서로 일어나는지를 끝까지
펼쳐 본 셈입니다. 1.6절에서 한 약속을 지킨 거죠.

마지막으로 한 가지. 엔진이 내놓는 BEL·RA·CSM은 **넣어 준 가정만큼만**
정확합니다. 엔진은 모형을 충실하고 빠르게 계산하지만, 그 가정이
현실에 맞는지는 말해 주지 못합니다. 가정을 세우고 검증하는 일 —
계리사의 판단 — 이야말로 측정에서 가장 어렵고 중요한 몫입니다. 이
엔진은 그 판단을 숫자로 옮기는 빠르고 투명한 도구입니다.

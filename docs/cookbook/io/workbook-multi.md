# 6.2 워크북 — 다 segment / 다 상품

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 여러 `(product_code, channel_code)` 가 섞인 portfolio 를
  **`measure` 로 한 번에** 평가 — 각 계약을 자기 segment 가정으로 라우팅
- segment 마다 다른 테이블을 거는 법 — 견본의 채널별 `lapse_table`,
  상품×채널별 `expense_table`
- segment 마다 다른 **StateModel** 을 거는 `state_model` 열
- 라우팅 키 매칭 (`product_code` / `channel_code`) 과 흔한 키 불일치 함정
```

[6.1](workbook-single) 은 워크북 한 바퀴와 한 segment 를 들여다봤습니다. 실무
portfolio 는 여러 상품·채널이 섞여 있고, 각 계약은 자기 segment 의 가정으로
평가돼야 합니다. 이 챕터는 그 **라우팅** 을 다룹니다.

## measure — 행마다 맞는 가정으로

`fcf.gmm.measure(mp, basis)` 에 단일 `Basis` 를 주면 한 가정을 전체에
적용하는 것과 달리, **`fcf.gmm.measure(mp, basis, full=False)`** 는 `basis`
가 dict 일 때 각 모델포인트의 `(product_code, channel_code)` 를 보고 사전
에서 맞는 `Basis` 를 골라 적용합니다 (dict 라우팅은 headline 전용이라
`full=False`). `basis` 는 [6.1](workbook-single) 의 `read_basis` 가 돌려준
바로 그 `(product_code, channel_code) -> Basis` 사전입니다.

```python
import numpy as np
import tempfile
from pathlib import Path
import fastcashflow as fcf

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    asmp_path = fcf.save_sample_basis(tmp / "basis.xlsx")
    pol_path  = fcf.save_sample_policies(tmp / "policies.csv")
    cov_path  = fcf.save_sample_coverages(tmp / "coverages.csv")
    cm_path   = fcf.save_sample_calculation_methods(tmp / "calculation_methods.csv")

    basis = fcf.read_basis(asmp_path)             # 7 segment 가정 사전
    mp = fcf.read_model_points(pol_path, coverages=cov_path,
                               calculation_methods=cm_path)

    # 각 계약을 자기 (product_code, channel_code) 가정으로 라우팅
    val = fcf.gmm.measure(mp, basis, full=False)
    print("BEL  sum =", f"{val.bel.sum():,.0f}")
    print("RA   sum =", f"{val.ra.sum():,.0f}")
    print("CSM  sum =", f"{val.csm.sum():,.0f}")
    print("loss sum =", f"{val.loss_component.sum():,.0f}")

    # segment 별로 BEL 을 갈라 보면 라우팅이 동작한 게 보인다
    prod = np.array(mp.product_code)
    chan = np.array(mp.channel_code)
    for key in sorted(basis):
        msk = (prod == key[0]) & (chan == key[1])
        if msk.any():
            print(f"  {key}: n={int(msk.sum())} BEL={val.bel[msk].sum():,.0f}")
```

출력:

```
BEL  sum = 27,818,583
RA   sum = 1,387,416
CSM  sum = 632,252
loss sum = 29,838,251
  ('HEALTH_A', 'FC'): n=2 BEL=939,473
  ('HEALTH_A', 'GA'): n=2 BEL=5,674,178
  ('HEALTH_A', 'TM'): n=2 BEL=6,463,740
  ('TERM_LIFE_A', 'FC'): n=2 BEL=3,700,421
  ('TERM_LIFE_A', 'GA'): n=1 BEL=757,630
  ('WHOLE_LIFE_A', 'FC'): n=1 BEL=7,403,742
  ('WHOLE_LIFE_A', 'GA'): n=1 BEL=2,879,399
```

11 개 계약이 7 개 segment 로 갈라져, 각 묶음이 자기 segment 의 사망률 ·
해지율 · 사업비로 평가됐습니다. segment 별 BEL 합은 전체 BEL 과 일치합니다.

## segment 마다 다르게 — 테이블 참조

segment 별 차이는 `segments` 시트의 **테이블 참조 열** 로 표현합니다. 같은
값은 `defaults` 행에 두고, 다른 부분만 segment 행이 덮어씁니다 (6.1 의
`defaults` 참조). 견본 워크북의 `segments` 는 이렇게 짜여 있습니다:

```{list-table}
:header-rows: 1
:widths: 30 22 22 26

* - segment
  - lapse_table
  - expense_table
  - 나머지
* - `defaults`
  - —
  - —
  - MORTALITY_STD / DISCOUNT_STD / state_model=WAIVER / ra_confidence=0.75
* - (TERM_LIFE_A, FC)
  - LAPSE_FC
  - EXP_TE_FC
  - (defaults 상속)
* - (TERM_LIFE_A, GA)
  - LAPSE_GA
  - EXP_TE_GA
  - (defaults 상속)
* - (HEALTH_A, TM)
  - LAPSE_GA
  - EXP_HE_TM
  - (defaults 상속)
```

읽는 법:

- **해지율은 채널별** — FC 는 `LAPSE_FC`, GA·TM 은 `LAPSE_GA`. 대면 채널과
  비대면 채널의 해지 행태가 다르다는 실무 구조입니다.
- **사업비는 상품×채널별** — `EXP_TE_FC` (정기·FC) / `EXP_HE_TM` (건강·TM) 처럼
  상품과 채널 조합마다 다른 `expense_table` 을 가리킵니다.
- **사망률 · 할인율 · state_model 은 공통** — `defaults` 에서 상속해 모든
  segment 가 `MORTALITY_STD` / `DISCOUNT_STD` / `WAIVER` 를 씁니다.

이 한 줄씩의 차이가 위 출력의 segment 별 BEL 격차로 나타납니다.

## segment 마다 다른 StateModel

`segments` 시트의 **`state_model` 열** 은 그 segment 가 쓸 상태 모델을
`STATE_MODELS` 레지스트리 이름으로 고릅니다. 한 워크북 안에서 segment 마다
다른 토폴로지를 줄 수 있습니다 — 예컨대 한 상품군은 납입면제가 있어 `WAIVER`,
다른 상품군은 상태 전이가 없어 빈 칸 (단일 상태):

```{list-table}
:header-rows: 1
:widths: 36 64

* - `state_model` 값
  - 의미
* - 빈 칸
  - 상태 모델 없음 (단일 상태). 상태 전이가 없는 정액형
* - `WAIVER`
  - `STATE_MODELS["WAIVER"]` — active / waiver 2-state ([3.1](../markov/waiver))
* - `WAIVER_PAIDUP`
  - active / waiver / paidup 3-state ([3.2](../markov/paid-up))
```

레지스트리에 없는 이름을 적으면 reader 가 등록된 키 목록과 함께 에러를
냅니다. 레지스트리 밖의 토폴로지 (재진단암 / DI 같은 semi-Markov) 는
[4 장](../semi-markov/reincidence) 처럼 코드로 `StateModel` 을 직접 조립해야
하므로, 워크북 `state_model` 열로는 고를 수 없습니다.

## 라우팅 메커니즘과 함정

### 키 매칭 — product_code / channel_code

`measure` 는 각 모델포인트의 `(product_code, channel_code)` 로
`basis` 사전을 조회합니다. 두 쪽의 코드 문자열이 **정확히** 같아야 합니다 —
`policies.csv` 의 `product_code` 와 `segments` 시트의 `product_code` 가
한 글자도 다르면 그 계약은 갈 곳이 없습니다.

```{admonition} 한글 코드와 유니코드 정규화
:class: note

코드에 한글을 써도 됩니다 (`product_code = "종신_A"` 등). `measure`
는 양쪽 키를 **NFC 정규화** (같은 글자의 서로 다른 유니코드 분해형을 한 형태로
맞춤) 한 뒤 비교하므로, 파일 인코딩에 따라 분해형이 달라도 매칭됩니다.
```

### 함정 1 — product_code / channel_code 가 없음

dict basis 라우팅은 모델포인트에 두 코드가 모두 채워져 있어야 합니다. 둘
중 하나라도 비면 라우팅을 못 합니다. 단일 가정으로 전체를 볼 거라면 dict
대신 단일 `Basis` 를 넘기세요 — `measure(mp, basis)`.

### 함정 2 — basis 에 없는 segment

모델포인트에 `(WHOLE_LIFE_B, TM)` 이 있는데 워크북에 그 segment 행이 없으면
조회가 실패합니다. portfolio 의 모든 `(product_code, channel_code)` 조합이 `segments`
시트에 행으로 존재해야 합니다.

### 함정 3 — 채널 빈 문자열 vs 누락

단일 채널 상품은 `channel_code` 를 **빈 문자열** `""` 로 두고 `segments` 도
같은 빈 문자열 행을 둡니다 (`None` / 누락이 아니라). 키는 항상
`(product_code, channel_code)` 2-튜플이고 빈 문자열도 유효한 채널 값입니다.

## 인접 레시피

- [6.1 워크북 — 단일 segment](workbook-single) — 워크북 시트 구조와 단일
  segment 의 read.
- [1.1 한눈에 보기](../basics/overview) — 네 입력 파일과 사용자 API.
- [3.1 납입면제](../markov/waiver) / [3.2 paid-up](../markov/paid-up) —
  `state_model` 열이 고르는 토폴로지.
- [튜토리얼 11장](../../tutorial/11-in-practice) — 결산 워크플로, 보유계약
  입력과 변동분해.
```

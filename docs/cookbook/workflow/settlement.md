# 9.1 결산 / 보유계약 평가

```{admonition} 이 챕터에서 배우는 것
:class: chapter-brief

- **신계약 평가** 와 **결산(보유계약) 평가** 의 차이 — 같은 엔진, 다른 입력 / 함수
- 분기말 "보유계약 마감파일" 한 장을 그대로 읽기 — `read_inforce_policies`
- 직전 분기 CSM을 carry-forward 하는 결산 측정 — `gmm.measure_inforce`
- 산출기초 (가정) 가 세그먼트별로 다를 때 — `state.subset` 으로 세그먼트별 측정
- 결산일 시점 BEL / RA / CSM을 포트폴리오 합계로 읽기
```

## 결산 평가 — 왜 / 언제

지금까지의 챕터는 모두 **신계약 평가** 였습니다 — 갓 인수한 계약을
가입 시점 (t = 0) 에서 측정. 하지만 실무의 IFRS 17 평가는 대부분
**분기말 결산** 입니다. 이미 몇 분기 굴러간 **보유계약** 을, 결산일
시점에서 다시 측정하고, 직전 분기 대비 잔액을 맞춰야 합니다.

두 평가는 **같은 엔진** 이지만 입력과 함수가 다른 두 **모드** 입니다:

```{list-table}
:header-rows: 1
:widths: 18 41 41

* -
  - 신계약 평가
  - 결산 (보유계약) 평가
* - 대상
  - 갓 인수한 계약 (t = 0)
  - 가입 후 N 개월 지난 보유계약
* - 측정 시점
  - 가입 시점
  - 결산일 (`elapsed_months` 시점)
* - 직전 CSM
  - 없음 (이번에 최초 인식)
  - 직전 분기 종가를 carry-forward
* - 할인율
  - 현재 가정
  - 가입 시점 lock-in 율 (Sec. B72(b))
* - 입력 파일
  - `policies.csv`
  - 보유계약 마감파일 (spec + 결산 상태)
* - 함수
  - `gmm.measure`
  - `gmm.measure_inforce`
```

결산 평가가 추가로 받는 것은 **결산 상태** 네 가지입니다 — 경과월수
(`elapsed_months`), 잔존 계약수 (`count`), 직전 분기 CSM (`prior_csm`),
가입 시점 할인율 (`lock_in_rate`). 정책관리 시스템이 매 분기말 떨어뜨리는
**보유계약 마감파일** 에 이 네 컬럼이 계약의 영구 spec 과 함께 들어 있고,
fastcashflow 는 그 한 파일을 그대로 받습니다.

## 모델링 매핑 — 세 함수

```{list-table}
:header-rows: 1
:widths: 28 72

* - 함수
  - 역할
* - `read_inforce_policies`
  - 보유계약 마감파일 한 장을 읽어 **`(ModelPoints, InforceState)`** 튜플로
    돌려줌. spec 은 `ModelPoints`, 결산 상태 네 컬럼은 `InforceState` 로 분리.
* - `apply_inforce_state`
  - 영구 spec (`ModelPoints`) 과 결산 상태 (`InforceState`) 가 **두 파일** 로
    따로 들어올 때 둘을 합침. 마감파일 1-파일 path 에서는 불필요.
* - `gmm.measure_inforce`
  - 결산 측정. `state` 에서 `prior_csm` / `lock_in_rate` 을 꺼내, 결산일
    시점 BEL / RA / CSM을 내고 직전 분기 CSM을 carry-forward.
```

`gmm.measure_inforce` 의 시그니처는 `measure_inforce(model_points, state,
basis, *, period_months=None, full=True)` 입니다. `period_months` 는 이번
보고기간의 길이 (분기 결산이면 3) — 이번 기간에 release 될 부분을 그만큼
잘라냅니다. `full=True` 는 월별 궤적까지, `full=False` 는 headline 네 숫자만
빠르게 냅니다.

## 최소 작동 예제 — 마감파일 한 장

샘플 파일로 한 분기 결산을 돌립니다. `samples.export` 가 spec + 결산 상태가
결합된 1-파일 `inforce_policies.csv` 를 세트에 함께 떨굽니다.

```python
import fastcashflow as fcf
import numpy as np

# 입력 파일 생성 (한 번만 -- 자기 파일이 있으면 생략).
# basis.xlsx + policies / coverages / calculation_methods / inforce_state /
# inforce_policies(결합 마감파일) 를 samples 폴더에 떨굼.
fcf.samples.export("samples", template="gmm", quiet=True)

# 산출기초 + 마감파일 읽기
basis = fcf.read_basis("samples/basis.xlsx")                            # {(product, channel): Basis}

model_points, state = fcf.read_inforce_policies(
    "samples/inforce_policies.csv",                                     # 마감 1-파일 (spec + state)
    coverages="samples/coverages.csv",                                  # 담보 파일
    calculation_methods="samples/calculation_methods.csv",              # 담보별 산출방식
)

# 전체 포트폴리오 결산 — dict basis 를 그대로 넘기면 각 (product, channel)
# 을 자기 산출기초로 자동 라우팅합니다 (신계약 measure() 와 같은 방식).
val = fcf.gmm.measure_inforce(
    model_points, state, basis,                                 # basis = 전체 dict
    period_months=3,                                            # 이번 분기 (3 개월)
)
fcf.write_measurement(val, "samples/results_2026Q1.csv")               # 결과 파일
```

`read_inforce_policies` 가 마감파일의 결산 상태 네 컬럼
(`elapsed_months` / `count` / `prior_csm` / `lock_in_rate`) 을 떼어
`state` 로 돌려주고, spec 은 `model_points` 로 돌려줍니다. `measure_inforce`
는 `state` 에서 `prior_csm` / `lock_in_rate` 을 읽고, `model_points` 가
나르는 `elapsed_months` 시점에서 측정합니다.

## 산출기초가 세그먼트별로 다를 때

`read_basis` 는 `{(product, channel): Basis}` 딕셔너리를 돌려줍니다 —
한 워크북에 여러 세그먼트 (상품 x 채널) 의 가정을 함께 담기 때문입니다.
위처럼 그 **dict 를 `measure_inforce` 에 그대로 넘기면** 각 계약을 자기
세그먼트의 산출기초로 자동 라우팅해 **전체 포트폴리오를 한 번에** 결산합니다
(내부에서 세그먼트별로 잘라 측정한 뒤 다시 하나로 잇습니다 — 신계약
`measure()` 의 dict 라우팅과 같은 방식).

세그먼트를 **직접 통제**하고 싶을 때만 (한 세그먼트만 보거나, 세그먼트별로
다른 `period_months` 를 주거나) 손수 잘라 단일 `Basis` 로 넘깁니다.
`ModelPoints.subset` 으로 계약을, 짝이 되는 `InforceState.subset` 으로 결산
상태를 같은 인덱스로 잘라 넘기면 됩니다 — 단, 자르기 전에
`align_inforce_state` 로 결산 상태를 보유계약 행 순서에 **한 번 맞춰** 둬야
합니다 (정렬 안 하면 `state.subset(idx)` 가 다른 계약의 직전 CSM 을 끌어옵니다;
`measure_inforce` 자체는 mp_id 로 내부 재정렬하지만, 합계의 Opening CSM 처럼
`state` 를 직접 읽는 자리는 정렬된 상태가 필요합니다).

```python
import fastcashflow as fcf
import numpy as np

# 산출기초 (가정) + 보유계약 + 결산 상태
basis     = fcf.samples.basis()          # {(product, channel): Basis}
portfolio = fcf.samples.model_points()   # 보유계약 영구 spec
state     = fcf.samples.inforce_state()  # 결산 상태 (경과월수 / 잔존 / 직전 CSM / lock-in)

# 결산 상태를 spec 에 fold + 보유계약 행 순서에 정렬 (prior_csm 까지)
mp    = fcf.apply_inforce_state(portfolio, state)
state = fcf.align_inforce_state(portfolio, state)

# 세그먼트별 결산 측정 -- 합계
bel = ra = csm = csm_prior = 0.0
for key, segment_basis in basis.items():
    idx = np.where((np.asarray(mp.product) == key[0]) &
                   (np.asarray(mp.channel) == key[1]))[0]
    if len(idx) == 0:
        continue
    val = fcf.gmm.measure_inforce(
        mp.subset(idx),          # 이 세그먼트의 보유계약
        state.subset(idx),       # 이 세그먼트의 결산 상태
        segment_basis,           # 이 세그먼트의 산출기초
        period_months=3,         # 이번 분기 (3 개월)
    )
    bel       += float(np.sum(val.bel))
    ra        += float(np.sum(val.ra))
    csm       += float(np.sum(val.csm))
    csm_prior += float(np.sum(state.subset(idx).prior_csm))

print("=== 2026 Q1 결산 (보유계약 평가) ===")
print(f"BEL         = {bel:>16,.0f}   (최선추정부채)")
print(f"RA          = {ra:>16,.0f}   (위험조정)")
print(f"Opening CSM = {csm_prior:>16,.0f}   (기초 = 직전 분기 종가)")
print(f"Closing CSM = {csm:>16,.0f}   (기말, carry-forward 결과)")
```

출력:

```
=== 2026 Q1 결산 (보유계약 평가) ===
BEL         =       11,252,051   (최선추정부채)
RA          =        1,476,720   (위험조정)
Opening CSM =          562,000   (기초 = 직전 분기 종가)
Closing CSM =          548,921   (기말, carry-forward 결과)
```

```{admonition} state.subset 을 꼭 써야 하나
:class: warning

`prior_csm` 만 잘라 넘기면 안 됩니다 — `elapsed_months` / `count` 는 전체
길이 그대로라 **길이가 어긋난 (ragged) 상태** 가 되고, 엔진이 이를 명시적으로
거부합니다. `InforceState.subset(idx)` 은 네 개의 per-계약 필드를 **한꺼번에**
잘라 일관성을 지키고, scalar `lock_in_rate` 은 그대로 나릅니다.
```

## 결과 해석

- **BEL (최선추정부채)** — 결산일 시점의 미래 현금흐름 현재가치. 신계약
  평가와 달리 이미 경과한 기간의 보험료 / 보험금은 빠지고, **잔존 기간** 만
  남습니다.
- **CSM (보험계약마진)** — 결산의 핵심. 직전 분기 종가 (`prior_csm`) 를
  출발점으로, 가입 시점 lock-in 율로 이자부리되고 (Sec. B72(a)), 이번
  기간 제공한 보장만큼 환입됩니다. 위에서 기말 CSM (548,921) 이 직전 분기
  기초 (562,000) 보다 줄어든 것은, 이번 분기 이자부리보다 환입이 컸기
  때문입니다. **무엇이 얼마씩 움직였는지** 의 항별 분해는 다음 챕터
  ([변동분해](movement)) 에서 풉니다.
- **직전 분기 CSM이 결산에 들어오는 이유** — IFRS 17 Sec. 44 의
  carry-forward. CSM은 매 분기 재계산되는 값이 아니라, 직전 종가에서
  이번 분기 변동을 더해 굴러가는 **누적 잔액** 입니다. 마감파일이
  `prior_csm` 을 나르는 이유가 이것입니다.

## 변형 — 입력이 두 파일로 들어올 때

ETL 환경에 따라 영구 spec (`policies.csv`) 과 분기별 갱신
(`inforce_state.csv`) 이 **따로** 들어오기도 합니다. 그때는 둘을 따로
읽어 합칩니다 — 결과는 1-파일 path 와 동일합니다.

```python
# 영구 spec (3 파일) 을 읽고, 분기 결산 상태를 따로 읽어 mp_id 로 합칩니다
model_points = fcf.read_model_points(
    "samples/policies.csv",                                 # 계약 spec (영구)
    coverages="samples/coverages.csv",                      # 담보
    calculation_methods="samples/calculation_methods.csv",  # 산출방법 카탈로그
)
state        = fcf.read_inforce_state("samples/inforce_state.csv")  # 분기 결산 상태만 따로
mp           = fcf.apply_inforce_state(model_points, state)         # mp_id 로 join (행 순서 무관)
```

대형 portfolio 에서는 마감파일을 `.parquet` / `.feather` 로 두는 편이
좋습니다 (`.xlsx` 는 시트당 ~ 1M 행 한계). 더 큰 규모는
`gmm.measure_stream` 으로 조각조각 흘려 보냅니다.

## 함정 / 검증

- **`period_months` 를 잊지 말 것** — 분기 결산이면 `period_months=3`.
  생략하면 기본 12 (연 단위) 로 이번 기간 release 가 잘려, CSM 환입이
  과대평가됩니다.
- **`lock_in_rate` 은 가입 시점 율** — 현재 할인율이 아닙니다. 마감파일이
  나르는 값을 그대로 쓰세요. 신계약 평가의 현재 율 (`discount_annual`) 과
  혼동하면 CSM 이자부리가 틀립니다.
- **mp_id 로 join** — `apply_inforce_state` 는 spec 과 state 를 `mp_id` 로
  맞춥니다 — 두 파일의 행 순서가 달라도 알아서 재정렬하고, `mp_id` 집합이
  어긋나면 (한쪽에만 있는 계약) `ValueError` 로 거부합니다. 미리 정렬할
  필요는 없습니다. (모델포인트에 `mp_id` 가 없는 손-제작 세트에서만 행 순서
  그대로의 위치 매칭으로 떨어집니다.)
- **검증** — 한 계약의 결산 CSM 흐름을 손으로 확인하려면
  [검증 패턴](validation) 의 `gmm.trace_csm_step` 으로 직전 CSM →
  이자부리 → 환입 → 기말 CSM을 항별로 펼쳐 봅니다.

## 인접 레시피

- [변동분해](movement) — 결산 사이에 BEL / CSM이 움직인 이유를 가정변경 /
  경험 / 이자 / 상각으로 귀속.
- [검증 패턴](validation) — 한 계약의 결산 계산 경로 추적.
- 기본 튜토리얼 11 장 (`실무에서의 활용 (1)`) — 결산 워크플로의 개념 도입.

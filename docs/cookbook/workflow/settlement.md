# 9.1 결산 / 보유계약 평가

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **신계약 평가** 와 **결산(보유계약) 정산** 의 차이 — 같은 엔진, 다른 입력 / 함수
- 결산의 동사 — `gmm.settle`: IFRS 17 Sec. 44 의 기초 → 기말 정산
- 분기말 "보유계약 마감파일" 한 장을 그대로 읽기 — `read_inforce_policies`
- 변동분석표 — `reconcile` 의 블록 항등식과 `finance_wedge`
- 다음 분기로 잔액 넘기기 — `closing_inputs()` 체이닝
- 결산일 **진단** 뷰 — `gmm.measure_inforce` (carry) 는 어디에 쓰는가
```

## 결산 정산 — 왜 / 언제

지금까지의 챕터는 모두 **신계약 평가** 였습니다 — 갓 인수한 계약을
가입 시점 (t = 0) 에서 측정. 하지만 실무의 IFRS 17 평가는 대부분
**분기말 결산** 입니다. 직전 분기에 닫아둔 잔액 (CSM / 손실요소) 에서
출발해, 이번 분기에 일어난 일 (이자부리 / 경험조정 / 보장 제공분 환입)
을 행별로 쌓고, 기말 잔액을 닫는 **기초 → 기말 정산** — IFRS 17
Sec. 44 가 규정하는 보고기간의 CSM 후속측정이 그것입니다.

fastcashflow 에서 이 정산의 동사는 **`gmm.settle`** 입니다. 두 평가는
같은 엔진이지만 입력과 함수가 다른 두 **모드** 입니다:

```{list-table}
:header-rows: 1
:widths: 18 41 41

* - 구분
  - 신계약 평가
  - 결산 (보유계약) 정산
* - 대상
  - 갓 인수한 계약 (t = 0)
  - 가입 후 N 개월 지난 보유계약
* - 측정
  - 가입 시점의 단일 측정
  - 한 보고기간의 기초 → 기말 movement
* - 직전 잔액
  - 없음 (이번에 최초 인식)
  - 직전 분기 종가 CSM / 손실요소에서 출발
* - 할인율
  - 현재 가정
  - CSM 블록은 가입 시점 lock-in 율 (Sec. B72(b)(c)),
    BEL / RA는 현재 율 (Sec. B72(a))
* - 입력 파일
  - `policies.csv`
  - 보유계약 마감파일 (spec + 결산 상태)
* - 함수
  - `gmm.measure`
  - `gmm.settle`
```

결산 정산이 추가로 받는 것은 **결산 상태** 입니다 — 경과월수
(`elapsed_months`), 결산일의 잔존 계약수 (`count`), 직전 분기 종가 CSM
(`prior_csm`), **기초 시점의 잔존 계약수** (`prior_count` — 기대 경로의
스케일이자 Sec. B119 환입 분모), 가입 시점 할인율 (`lock_in_rate`).
직전 분기가 손실부담이었다면 `prior_loss_component` 도 함께 넘깁니다.
정책관리 시스템이 매 분기말 떨어뜨리는 **보유계약 마감파일** 에 이
컬럼들이 계약의 영구 spec 과 함께 들어 있고, fastcashflow 는 그 한
파일을 그대로 받습니다.

## 모델링 매핑 — 함수들

```{list-table}
:header-rows: 1
:widths: 28 72

* - 함수
  - 역할
* - `read_inforce_policies`
  - 보유계약 마감파일 한 장을 읽어 **`(ModelPoints, InforceState)`** 튜플로
    돌려줌. spec 은 `ModelPoints`, 결산 상태 컬럼은 `InforceState` 로 분리.
* - `gmm.settle`
  - **결산 정산.** 한 세그먼트 (단일 `Basis`) 의 보유계약을 한 보고기간
    정산해 `GMMSettlementMovement` (per-계약 행별 movement) 를 돌려줌.
* - `reconcile`
  - movement 리스트를 포트폴리오 합계의 **변동분석표** 로 집계.
    기초 + 모든 행 = 기말 이 블록마다 정확히 맞아떨어짐.
* - `closing_inputs()`
  - movement 의 메서드. 기말 잔액을 다음 분기의 `(ModelPoints,
    InforceState)` 시드로 돌려줌 — 분기 체이닝.
* - `gmm.measure_inforce`
  - **진단 / 런오프 뷰** (정산 아님). 결산일 한 시점의 BEL / RA 현행추정
    + carry-only CSM. 아래 "진단 뷰" 절 참조.
```

`gmm.settle` 의 시그니처는 `settle(model_points, state, basis, *,
period_months=None)` 입니다. `period_months` 는 이번 보고기간의 길이
(분기 결산이면 3, 생략하면 12) — 기초가 그만큼 앞 시점이 됩니다.

## 최소 작동 예제 — 마감파일 한 장

샘플 파일로 한 분기 결산을 돌립니다. `samples.export` 가 spec + 결산
상태가 결합된 1-파일 `inforce_policies.csv` 를 세트에 함께 떨굽니다.
`settle` 은 **세그먼트 (단일 산출기초) 단위** 입니다 — 신계약 `measure`
처럼 `BasisRouter` 를 통째로 받지 않으므로, 여러 세그먼트는 세그먼트별로
잘라 호출하고 movement 를 모읍니다.

```python
import fastcashflow as fcf
import numpy as np

# 입력 파일 생성 (한 번만 -- 자기 파일이 있으면 생략).
# basis.xlsx + policies / coverages / calculation_methods / inforce_state /
# inforce_policies(결합 마감파일) 를 samples 폴더에 떨굼.
fcf.samples.export("samples", template="gmm", quiet=True)

# 산출기초 + 마감파일 읽기
basis = fcf.read_basis("samples/basis.xlsx")                            # BasisRouter: {(product, channel): Basis}

model_points, state = fcf.read_inforce_policies(
    "samples/inforce_policies.csv",                                     # 마감 1-파일 (spec + state)
    coverages="samples/coverages.csv",                                  # 담보 파일
    calculation_methods="samples/calculation_methods.csv",              # 담보별 산출방법
)

# 세그먼트별 결산 정산 -- settle 은 세그먼트(단일 Basis) 단위
movements = []
for key, segment_basis in basis.segments.items():
    idx = np.where((np.asarray(model_points.product) == key[0]) &
                   (np.asarray(model_points.channel) == key[1]))[0]
    if len(idx) == 0:
        continue
    movements.append(fcf.gmm.settle(
        model_points.subset(idx),    # 이 세그먼트의 보유계약
        state.subset(idx),           # 결산 상태 (직전 CSM / 기초 잔존 / lock-in)
        segment_basis,               # 이 세그먼트의 산출기초
        period_months=3,             # 이번 분기 (3 개월)
    ))

# 포트폴리오 합계 -- 기말 잔액
csm_open  = sum(float(mv.csm_opening.sum())  for mv in movements)
csm_close = sum(float(mv.csm_closing.sum())  for mv in movements)
bel_close = sum(float(mv.bel_closing.sum())  for mv in movements)
ra_close  = sum(float(mv.ra_closing.sum())   for mv in movements)
lc_close  = sum(float(mv.loss_component_closing.sum()) for mv in movements)

print("=== 2026 Q1 period close ===")
print(f"BEL  (closing)         = {bel_close:>14,.0f}")
print(f"RA   (closing)         = {ra_close:>14,.0f}")
print(f"CSM  (opening)         = {csm_open:>14,.0f}")
print(f"CSM  (closing)         = {csm_close:>14,.0f}")
print(f"Loss component (close) = {lc_close:>14,.0f}")
```

출력:

```
=== 2026 Q1 period close ===
BEL  (closing)         =     11,252,051
RA   (closing)         =      1,476,720
CSM  (opening)         =        562,000
CSM  (closing)         =        400,573
Loss component (close) =        400,126
```

`read_inforce_policies` 가 마감파일의 결산 상태 컬럼을 떼어 `state` 로
돌려주고, spec 은 `model_points` 로 돌려줍니다 — 두 개체는 이미 행 순서가
맞춰져 있어 같은 인덱스로 `subset` 하면 됩니다. 기말 CSM (400,573) 이
기초 (562,000) 보다 크게 줄면서 손실요소 (400,126) 가 잡힌 것은, 이
샘플 마감파일의 직전 CSM이 일부 세그먼트에서 불리한 경험 / 재산정을
흡수하기에 얇기 때문입니다 — 유리·불리가 섞인 책은 원래 이렇게
닫힙니다 (샘플 포트폴리오는 신계약 시점에도 11 건 중 4 건이
손실부담이도록 만들어져 있습니다).

## 변동분석표 — Sec. 44 의 행들

기말 잔액 하나만 보면 "왜 줄었는지" 를 모릅니다. `reconcile` 이 한
세그먼트의 movement 를 **행별 정산표** 로 집계합니다.

```python
# 한 세그먼트의 정산표
key = ("HEALTH_A", "GA")
idx = np.where((np.asarray(model_points.product) == key[0]) &
               (np.asarray(model_points.channel) == key[1]))[0]
mv = fcf.gmm.settle(
    model_points.subset(idx),        # 이 세그먼트의 보유계약
    state.subset(idx),               # 결산 상태
    basis.resolve(key),              # 단일 Basis 로 resolve
    period_months=3,                 # 이번 분기 (3 개월)
)
r = fcf.reconcile([mv])[0]           # movement -> 변동분석표

print(f"=== settlement table -- {key[0]} / {key[1]}, 3 months ===")
print("CSM block (Sec. 44)")
for label, value in (
    ("opening",               r.csm_opening),
    ("interest accretion",    r.csm_accretion),
    ("experience unlocking",  r.csm_experience_unlocking),
    ("loss comp. reversed",   r.loss_component_reversed),
    ("loss comp. recognised", r.loss_component_recognised),
    ("release",               r.csm_release),
    ("closing",               r.csm_closing),
):
    print(f"  {label:<24}{value + 0.0:>12,.0f}")
print("BEL block")
for label, value in (
    ("opening",    r.bel_opening),
    ("interest",   r.bel_interest),
    ("release",    r.bel_release),
    ("experience", r.bel_experience),
    ("closing",    r.bel_closing),
):
    print(f"  {label:<24}{value + 0.0:>12,.0f}")
print(f"  {'finance wedge (B97(a))':<24}{r.finance_wedge:>12,.0f}")
```

출력:

```
=== settlement table -- HEALTH_A / GA, 3 months ===
CSM block (Sec. 44)
  opening                       88,000
  interest accretion               653
  experience unlocking          -1,569
  loss comp. reversed                0
  loss comp. recognised              0
  release                       -2,485
  closing                       84,599
BEL block
  opening                     -786,618
  interest                      -6,816
  release                      253,159
  experience                   -15,801
  closing                     -556,076
  finance wedge (B97(a))         7,317
```

각 블록은 **기초 + 모든 행 = 기말** 이 정확히 맞습니다 (run-off 행은
표시 음수). 행의 의미:

- **interest accretion** — 직전 CSM의 lock-in 율 이자부리 (Sec.
  44(b)/B72(b)). BEL 블록의 interest 는 현재 율 unwind.
- **experience unlocking** — 미래 서비스에 관한 이행현금흐름 변동분을
  CSM이 흡수 (Sec. 44(c)). **lock-in 율로 측정** 합니다 (Sec. B72(c)).
- **release** — 이번 분기 보장을 제공한 만큼의 환입 (Sec. 44(e)/B119).
  분모는 기초 시점의 잔여 보장단위 — `prior_count` 가 필요한 이유.
- **loss comp. reversed / recognised** — 손실요소 알게브라 (Sec.
  48/50(b)). 불리한 변동이 CSM을 0 까지 깎고 넘치면 손실요소로
  인식되고, 유리한 변동은 손실요소를 먼저 되돌린 뒤 CSM을 재수립합니다.
- **finance wedge** — CSM 조정은 lock-in 율, BEL / RA 변동은 현재 율로
  재는 데서 생기는 틈. CSM 블록 **밖** 의 보험금융손익 행입니다 (Sec.
  B97(a)). 세 항은 `experience unlocking + finance wedge ==
  -(BEL experience + RA experience)` 로 정확히 묶입니다.

## 분기 체이닝 — 다음 분기로 잔액 넘기기

기말 잔액은 다음 분기의 기초입니다. `closing_inputs()` 가 그 시드를
돌려줍니다 — `prior_csm` / `prior_loss_component` / `prior_count` 가
이번 분기의 종가로 채워진 `(ModelPoints, InforceState)` 쌍. 호출자는
그 쌍을 **다음 관측일로 전진** 시키기만 하면 됩니다 (`elapsed_months`
+3, `count` 는 다음 분기말의 실제 관측 잔존).

```python
from dataclasses import replace

# 기말 시드 -> 다음 분기 관측으로 전진
mp_close, state_close = mv.closing_inputs()
count_next = np.asarray(mp_close.count) * 0.995          # 다음 분기말 관측 잔존 (예시)
mp_next    = replace(mp_close,
                     elapsed_months=np.asarray(mp_close.elapsed_months) + 3,
                     count=count_next)
state_next = replace(state_close,
                     elapsed_months=np.asarray(state_close.elapsed_months) + 3,
                     count=count_next)

# 다음 분기 정산
mv_next = fcf.gmm.settle(mp_next, state_next, basis.resolve(key),
                         period_months=3)
print(f"Q1 closing CSM = {float(mv.csm_closing.sum()):>12,.0f}")
print(f"Q2 opening CSM = {float(mv_next.csm_opening.sum()):>12,.0f}")
print(f"Q2 closing CSM = {float(mv_next.csm_closing.sum()):>12,.0f}")
```

출력:

```
Q1 closing CSM =       84,599
Q2 opening CSM =       84,599
Q2 closing CSM =       73,237
```

이번 분기의 기말이 다음 분기의 기초로 그대로 들어갑니다. 경험이
기대대로 흐르면 (on-track) 6 개월 정산 두 번은 12 개월 정산 한 번과
정확히 같은 기말에 닿습니다 — 정산이 보고주기에 의존하지 않는다는
telescoping (망원경처럼 기간 분할이 접혀 합쳐지는 성질) 검증이 테스트로
박혀 있습니다.

## 진단 뷰 — `gmm.measure_inforce` 는 어디에 쓰나

`gmm.measure_inforce` 는 결산일 **한 시점** 의 보유계약 가치를 내는
**진단 / 런오프 프로젝터** 입니다. 정산이 아닙니다:

```python
# 진단 뷰 -- 결산일 한 시점의 보유계약 가치 (BasisRouter 통째 라우팅 지원)
val = fcf.gmm.measure_inforce(model_points, state, basis, period_months=3)

print(f"BEL = {float(np.sum(val.bel)):>14,.0f}")
print(f"RA  = {float(np.sum(val.ra)):>14,.0f}")
print(f"CSM = {float(np.sum(val.csm)):>14,.0f}  (carry-only)")
```

출력:

```
BEL =     11,252,051
RA  =      1,476,720
CSM =        548,921  (carry-only)
```

- **BEL / RA는 결산일의 현행추정으로 유효합니다** (Sec. 40 의 잔여보장부채
  구성요소). 실제로 `settle` 의 기말 BEL / RA (위 11,252,051 / 1,476,720)
  와 정확히 같습니다 — 같은 산수의 두 표면입니다.
- **CSM은 carry-only 근사**입니다 — 직전 CSM을 이자부리 / 환입만 하고
  Sec. 44(c) unlocking 을 건너뜁니다. 위에서 carry CSM (548,921) 과
  settlement CSM (400,573) + 손실요소 (400,126) 가 갈라진 것이 그
  차이입니다. 그래서 결과에는 `measurement_basis='settlement_carry'`
  마커가 찍히고, 회계 산출 소비처 (`group` / `group_of_contracts` /
  `roll_forward` / `report` / plot) 가 이를 거부합니다 — carry CSM이
  결산 숫자로 흘러들지 않습니다.
- **쓰는 자리**: 기중 (분기 사이) 모니터링, 잔존 책의 런오프 프로젝션,
  그리고 `settle` 의 검증 anchor (`settle` 의 기말 BEL / RA == carry
  headline, 경험이 on-track 이면 기말 CSM == carry CSM).

## 메모리를 넘는 규모

per-계약 movement 를 다 들 수 없는 책은 두 규모 변형으로 닫습니다.

```python
# 포트폴리오 합계만 -- bounded memory (chunk 단위 정산, 합계 누적)
agg = fcf.gmm.settle_aggregate(
    model_points.subset(idx), state.subset(idx), basis.resolve(key),
    period_months=3,                  # 이번 분기
    chunk_size=200_000,               # 한 번에 드는 계약 수
)
print(f"aggregate csm_closing = {agg.csm_closing:>12,.0f}")
```

출력:

```
aggregate csm_closing =       84,599
```

`settle_aggregate` 는 movement 의 모든 행을 합계 스칼라로 누적합니다 —
`reconcile(agg)` 로 같은 변동분석표가 나오고, 결과는 chunk 크기와
무관하게 per-계약 정산의 합과 정확히 같습니다. 단 per-계약 기말이
없으므로 `closing_inputs()` 체이닝은 불가합니다 (`ValueError`).

디스크 기반은 `gmm.settle_stream` 입니다 — parquet 마감파일을 chunk 로
읽어 정산하고 per-계약 movement 를 `part-*.parquet` 로 씁니다. 각 part
가 기말 체인 컬럼 (`count` / `lock_in_rate` / 기말 잔액) 을 나르므로
**part 파일만으로 다음 분기의 state 파일을 조립** 할 수 있습니다 —
`closing_inputs()` 의 디스크 버전:

```text
fcf.gmm.settle_stream(
    "inforce_2026Q1.parquet", "out/2026Q1",   # 마감파일 -> movement parts
    basis,
    coverages="coverages.parquet",
    calculation_methods="calculation_methods.csv",
    period_months=3,
)
# 다음 분기 state: parts 에서 prior_csm <- csm_closing,
# prior_loss_component <- loss_component_closing, prior_count <- count
```

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
model_points = fcf.apply_inforce_state(model_points, state)         # mp_id 로 join (행 순서 무관)
```

대형 portfolio 에서는 마감파일을 `.parquet` / `.feather` 로 두는 편이
좋습니다 (`.xlsx` 는 시트당 ~ 1M 행 한계). 더 큰 규모는 위의
`gmm.settle_stream` 으로 조각조각 흘려 보냅니다.

## 함정 / 검증

- **`prior_count` 를 잊지 말 것** — 기초 시점의 잔존 계약수. 기대 경로의
  스케일이자 Sec. B119 환입 분모라 없으면 `settle` 이 `ValueError` 로
  거부합니다. 마감파일에 한 컬럼 더 실어 두세요.
- **`period_months` 를 잊지 말 것** — 분기 결산이면 `period_months=3`.
  생략하면 기본 12 (연 단위) 로 기초가 1 년 앞 시점이 됩니다.
- **`lock_in_rate` 은 가입 시점 율** — 현재 할인율이 아닙니다. 마감파일이
  나르는 값을 그대로 쓰세요. CSM 블록 (이자부리 / unlocking) 이 이 율로,
  BEL / RA는 현재 율로 갑니다 — 둘의 틈이 `finance_wedge` 입니다.
- **음수 `prior_csm` 은 거부** — CSM은 0 에서 floor 됩니다. 직전 분기가
  손실부담이었다면 음수 CSM이 아니라 `prior_loss_component` 컬럼으로
  넘깁니다 (Sec. 47-52: 한 집합은 CSM 또는 손실요소 중 하나만 가짐).
- **`settlement_pattern` 책은 v1 거부** — 청구 정산 지연이 있는 책은
  기초·기말 양쪽에 발생사고부채 (LIC) 가 걸려 있는데 v1 movement 에 LIC
  행이 없습니다. 즉시 지급 (`settlement_pattern=None`) 으로 두세요.
- **mp_id 로 join** — `state` 는 `mp_id` 로 보유계약에 맞춰집니다. 두
  파일의 행 순서가 달라도 재정렬되고, `mp_id` 집합이 어긋나면
  `ValueError` 로 거부합니다.
- **검증** — ① 변동분석표의 블록 항등식 (기초 + 행들 = 기말), ② 기말
  BEL / RA 가 `measure_inforce` 의 현행추정과 일치, ③ on-track 이면 기말
  CSM이 carry 와 일치 + 보고주기 분할 불변 (6m x 2 == 12m). 셋 다
  테스트 스위트에 박혀 있고, 자기 책에서도 같은 항등식으로 점검할 수
  있습니다.

## 인접 레시피

- [변동분해](movement) — **신계약 (inception) 측정** 을 보고기간별로
  잘라 가정변경 / 경험 / 이자 / 상각으로 귀속. 보유계약 결산의 행별
  분해는 이 챕터의 `settle` + `reconcile` 이 담당.
- [검증 패턴](validation) — 한 계약의 측정 계산 경로 추적.
- 기본 튜토리얼 11 장 (`실무에서의 활용 (1)`) — 결산 워크플로의 개념 도입.

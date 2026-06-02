# 9.2 변동분해

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 한 측정 결과를 보고기간별 **변동분석표** 로 자르기 — `roll_forward` + `reconcile`
- IFRS 17 Sec. 101 의 다섯 행 — Opening / Future service / Finance / Release / Closing
- **가정변경** 이 BEL / CSM을 얼마씩 움직였는지 귀속 — `revised` / `revised_at`
- **경험조정** (실제 잔존이 예상과 다를 때) 귀속 — `actual_inforce` / `experience_at`
- 유리한 가정변경이 손익이 아니라 **CSM으로 흡수** 되는 Sec. 44 메커니즘
```

## 변동분해 — 왜 / 언제

결산이 한 시점의 **잔액** 을 낸다면, 변동분해 (analysis of change) 는 두
시점 **사이의 움직임** 을 설명합니다. 분기 결산에서 가장 자주 받는 질문이
이것입니다 — "직전 분기 대비 CSM이 왜 줄었나?", "BEL이 늘었는데 가정을
바꿔서인가, 경험이 나빠서인가, 그냥 시간이 흘러서인가?"

fastcashflow 의 변동분해는 그 움직임을 **네 갈래** 로 가릅니다:

1. **Future service (미래서비스)** — 가정 재산정 / 미래 현금흐름 추정 변경.
   GMM에서는 손익이 아니라 **CSM을 조정** (Sec. 44).
2. **Finance (보험금융손익)** — 시간이 흐르며 생기는 할인 / 이자 효과.
3. **Release (당기 서비스 제공분)** — 이번 기간 보장을 제공한 만큼 풀린 부분.
4. **Experience (경험조정)** — 실제 잔존이 예상과 다를 때의 차이.

이 분해가 메모리상 fastcashflow 의 **결정적 차별점** 입니다 — 숫자를
내는 것보다 **"왜 그 숫자가 됐는가"** 를 빠르게 설명하는 것.

## 모델링 매핑 — 두 함수

```{list-table}
:header-rows: 1
:widths: 26 74

* - 함수
  - 역할
* - `roll_forward`
  - 한 측정 결과를 `period_months` 길이의 보고기간들로 자름. 기간별로
    per-계약 움직임 (`PeriodMovement`) 을 돌려줌. `revised` / `actual_inforce`
    옵션으로 가정변경 / 경험조정을 얹음.
* - `reconcile`
  - 기간별 움직임을 포트폴리오 합계의 **변동분석표** (`Reconciliation`,
    Sec. 101 배열) 로 집계. 기초 + 모든 변동 = 기말 이 정확히 맞아떨어짐.
```

`roll_forward` 는 측정 모형마다 (GMM / PAA / VFA) 자동 분기합니다 —
GMM은 BEL / RA / CSM의 움직임, PAA는 잔여보장부채 (LRC) 의 roll,
VFA는 CSM의 roll. 아래는 GMM입니다.

## 최소 작동 예제 — 기간별 변동분석표

건강보험 한 세그먼트 (HEALTH_A / FC) 를 측정하고, 첫 보고기간 (12 개월) 의
변동분석표를 봅니다. 변동분해는 월별 궤적이 필요하므로 `full=True` (기본값)
로 측정합니다.

```python
import fastcashflow as fcf
import numpy as np

basis     = fcf.samples.basis()
portfolio = fcf.samples.model_points()

# 한 세그먼트 측정 (full trajectory)
key = ("HEALTH_A", "FC")
idx = np.where((np.asarray(portfolio.product_code) == key[0]) &
               (np.asarray(portfolio.channel_code) == key[1]))[0]
m = fcf.gmm.measure(portfolio.subset(idx), basis[key])

# 보고기간으로 자르고 → 변동분석표로 집계
movements = fcf.roll_forward(m, period_months=12)    # 12 개월 기간
recon     = fcf.reconcile(movements)                 # 기간 수만큼의 변동분석표

r = recon[0]                                         # 첫 보고기간
print(f"변동분해 -- HEALTH_A / FC, months {r.month_start + 1}-{r.month_end}")
print(f"{'':<16}{'BEL':>12}{'RA':>12}{'CSM':>12}")
for row, lab in (("opening", "Opening"), ("future_service", "Future service"),
                 ("finance", "Finance"), ("release", "Release"),
                 ("closing", "Closing")):
    bel = getattr(r, f"bel_{row}")
    ra  = getattr(r, f"ra_{row}")
    csm = getattr(r, f"csm_{row}")
    print(f"{lab:<16}{bel:>12,.0f}{ra:>12,.0f}{csm:>12,.0f}")
```

출력:

```
변동분해 -- HEALTH_A / FC, months 1-12
                         BEL          RA         CSM
Opening              939,473     304,669     632,252
Future service             0           0           0
Finance               39,450       8,729      17,648
Release            1,001,308     -29,440     -95,162
Closing            1,980,231     283,958     554,737
```

## 결과 해석

다섯 행을 읽는 법 (`reconcile` 은 run-off 행을 음수로 두어, **기초 + 모든
변동 = 기말** 이 정확히 맞아떨어집니다):

```{list-table}
:header-rows: 0
:widths: 30 70

* - Opening (기초잔액)
  - 보고기간 시작 시점의 잔액. 한 기간의 Closing 이 다음 기간의 Opening.
* - Future service (미래서비스 관련 변동)
  - 아직 제공하지 않은 미래 보장에 관한 변동 (가정 재산정 등). GMM 에서는
    CSM을 조정. **위 표에서 0 인 것은** 가정을 고정한 결정론적 투영을
    그대로 굴려 바뀔 것이 없기 때문 — 다음 절에서 가정을 흔들면 여기가
    움직입니다.
* - Finance (보험금융손익)
  - 할인 / 이자 효과. BEL / RA는 할인이 풀리고, CSM은 가입 시점에 고정된
    이자율로 부리됩니다.
* - Release (당기 서비스 제공분)
  - 이번 기간 보장을 제공한 만큼 풀린 부분. CSM 상각, RA 해제, 당기 예상
    현금흐름의 런오프. CSM의 -95,162 가 이번 기간 손익으로 인식된 마진.
* - Closing (기말잔액)
  - 기간 종료 시점 잔액. Opening 에 위 변동을 더하면 정확히 Closing.
```

`plot_analysis_of_change` 가 이 표를 waterfall (폭포형 누적) 차트로 그립니다:

```python
fcf.plot_analysis_of_change(recon[0])   # 기본 CSM; component 로 BEL / RA 선택
```

## 가정변경 귀속 — Sec. 44 의 핵심

위 표에서 Future service 가 0 인 것은 가정을 안 바꿨기 때문입니다. 분기
사이에 **가정을 재산정** 했다면 (예: 경험분석 결과 사망률을 10% 상향),
그 효과가 BEL / CSM을 얼마씩 움직였는지 알아야 합니다. `roll_forward` 에
**변경 후 기초로 다시 측정한 결과** (`revised`) 와 **변경 발효 시점**
(`revised_at`) 을 넘기면 됩니다.

```python
from dataclasses import replace

# 사망률 +10% 로 재산정한 산출기초 -- 기존 rate callable 을 감싸 배수.
# 엔진이 넘기는 인자를 그대로 전달 (*args) 하므로 rate 함수의 시그니처에
# 무관하게 동작합니다.
base_mort = basis[key].mortality_annual                 # 샘플 기초의 사망률 함수
revised_basis = replace(
    basis[key],
    mortality_annual=lambda *a: base_mort(*a) * 1.10,   # 사망률 +10%
)
m_revised = fcf.gmm.measure(portfolio.subset(idx), revised_basis)

# month 12 에 발효된 가정변경을 얹어 변동분해
movements = fcf.roll_forward(
    m,                       # 기준 측정 (변경 전)
    period_months=12,        # 12 개월 기간
    revised=m_revised,       # 변경 후 재측정
    revised_at=12,           # 변경 발효 시점 (period_months 의 배수)
)
recon = fcf.reconcile(movements)

r = recon[1]                 # 변경이 발효된 기간 (months 12-24)
print(f"변동분해 -- HEALTH_A / FC, months {r.month_start + 1}-{r.month_end}"
      f"  (mortality +10% at month 12)")
print(f"{'':<16}{'BEL':>12}{'RA':>12}{'CSM':>12}")
for row, lab in (("opening", "Opening"), ("future_service", "Future service"),
                 ("finance", "Finance"), ("release", "Release"),
                 ("closing", "Closing")):
    bel = getattr(r, f"bel_{row}")
    ra  = getattr(r, f"ra_{row}")
    csm = getattr(r, f"csm_{row}")
    print(f"{lab:<16}{bel:>12,.0f}{ra:>12,.0f}{csm:>12,.0f}")
```

출력:

```
변동분해 -- HEALTH_A / FC, months 13-24  (mortality +10% at month 12)
                         BEL          RA         CSM
Opening            1,980,231     283,958     554,737
Future service        -2,693        -595       3,287
Finance               72,840       8,105      15,515
Release              948,812     -28,417     -88,416
Closing            2,999,190     263,051     485,124
```

이제 Future service 행이 살아 있습니다. 읽는 법:

- **BEL Future service = -2,693** — 사망률을 올리니 건강보험 BEL이
  줄었습니다. 사람이 더 많이 사망하면 그만큼 미래에 질병 / 입원
  보험금을 청구할 사람이 줄기 때문입니다 (건강 담보에서 사망은 보장
  사건이 아니라 in-force 를 끝내는 decrement).
- **CSM Future service = +3,287** — BEL이 줄어든 만큼 (유리한 변경) 이
  손익으로 가지 않고 **CSM을 늘립니다**. 이것이 IFRS 17 Sec. 44 의
  핵심 — 미래서비스에 관한 가정변경은 CSM으로 흡수되어, 남은 보장기간에
  걸쳐 천천히 인식됩니다. (반대로 불리한 변경이 CSM 잔액을 넘어서면,
  초과분은 `loss_component` 로 즉시 손실 인식됩니다.)

```{admonition} revised_at 은 period_months 의 배수
:class: note

가정변경은 보고기간 **경계** 에서 인식됩니다. `period_months=12` 이면
`revised_at` 은 12, 24, ... 만 됩니다 (6 은 거부). 변경 효과는 그 달부터
*시작하는* 기간에 잡히므로, `revised_at=12` 의 효과는 `recon[0]`
(months 1-12) 이 아니라 `recon[1]` (months 13-24) 에 나타납니다.
```

## 변형 — 경험조정

가정은 그대로인데 **실제 잔존 계약수** 가 예상과 다를 때 (실제 해지가
예상보다 많았다 등) 는 `actual_inforce` (기간 말 실제 잔존, `(n_mp,)`
또는 기간마다 굴릴 `(n_periods, n_mp)`) 와 `experience_at` 을 넘깁니다.
가정변경과 마찬가지로 그 차이가 fulfilment cash flow 변동을 통해 CSM을
조정합니다 (v1 은 한 호출에서 가정변경 또는 경험조정 하나만).

```python
# 기간 말 실제 잔존 -- 실무에선 정책관리 시스템의 실제 in-force.
# 여기서는 모델포인트당 97% 가 남았다고 가정 (예시 값).
actual = np.full(m.bel_path.shape[0], 0.97)

movements = fcf.roll_forward(
    m, period_months=12,
    actual_inforce=actual,   # 기간 말 실제 잔존 (n_mp,)
    experience_at=12,        # 경험 반영 시점
)
```

## 함정 / 검증

- **`full=True` 필수** — 변동분해는 월별 궤적을 자릅니다. `full=False`
  (headline 만) 측정 결과를 넘기면 명시적 에러가 납니다.
- **PAA / VFA도 됨** — `roll_forward` 는 측정 모형으로 자동 분기합니다.
  단 `revised` / `actual_inforce` 옵션은 **GMM 전용** 입니다 (PAA / VFA에
  넘기면 거부).
- **부호 규약** — run-off (당기 제공분) 는 음수로 표시됩니다. 그래서
  Opening 에 모든 행을 더하면 Closing 이 됩니다. CSM Release 가 음수
  (-95,162) 인 것은 그만큼 마진이 풀려 손익 인식됐다는 뜻.
- **검증** — 한 계약의 CSM 이자부리 / 환입을 항별로 손계산과 맞추려면
  [검증 패턴](validation) 의 `gmm.trace_csm_step`.

## 인접 레시피

- [결산 / 보유계약 평가](settlement) — 변동분해의 출발점인 결산 측정.
- [시나리오 / 민감도 분석](sensitivity) — 가정 한 축을 흔들어 효과를
  추적 (`gmm.trace_diff` 로 단계별 전파).
- 기본 튜토리얼 12 장 (`실무에서의 활용 (2)`) — 변동분석 / 리포트의 개념 도입.

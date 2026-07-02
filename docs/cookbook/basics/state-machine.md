# 1.6 상태기계 — 상태·전이·경과

계약의 in-force 를 단일 생존 트랙이 아니라 **여러 상태 사이를 오가는
점유 (occupancy)** 로 다루는 것이 fastcashflow 의 상태기계 (state machine)
입니다. 납입면제·완납·장해·요양·재발 같은 한국 상품 구조가 여기서 표현됩니다.

이 챕터는 개별 상품 레시피 (3 장 markov / 4 장 semi-markov) 로 들어가기 전에,
**상태기계를 이루는 조각 하나하나가 무엇이고 왜 있는지** 를 한자리에 모읍니다.
공개 네임스페이스는 `fcf.multistate` 이고, 핵심 클래스는 `Model` / `State` /
`Transition` 셋입니다.

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **상태 (`State`) 와 전이 (`Transition`)** — 상태 번호, `to=None` (이탈) 과
  목적지 번호 (이동) 의 차이
- **전이의 두 종류** — 확률 (rate) 전이와 결정적 (at_premium_term /
  after_sojourn_months) 전이, 그리고 어느 쪽이 엣지가 되는지
- **경과 (sojourn)** — `sojourn_tracking_months` (코호트 추적) 와
  `sojourn_dependent` (전이별 경과 의존) 이 왜 둘 다 필요한지
- `Model` / `State` / `Transition` **인자 레퍼런스**
- **대부분은 `Model.from_preset` 한 줄** — 복잡한 인자는 특수 상품용 옵션
:::

## 상태와 전이

한 계약은 항상 **하나의 상태** 에 있습니다. 기본 2-state 모델의 상태는
`active` (정상 납입) 와 `waiver` (납입면제) 입니다. 매달 계약은 상태를 옮기거나
(전이), 아예 빠져나갑니다 (사망·해지).

상태에는 **번호** 가 매겨집니다 — `Model(states=(...))` 에 나열한 순서가 곧 상태
번호이고, 0 번이 최초 발행 상태입니다. 엔진의 점유 배열이 `(계약, 상태, 시간)`
모양이라, "waiver 의 점유" 는 배열의 1 번 축으로 꺼냅니다. 이름 (`"active"`) 은
사람이 읽으라고, 번호 (0) 는 배열 인덱싱 (= 속도) 을 위해 — 둘 다 갖습니다.
`Model.transitions` 가 그 구조를 풀어 보여줍니다:

```python
import fastcashflow as fcf
from fastcashflow.multistate import State, Model, Transition

model = Model.from_preset("ACTIVE_WAIVER")
print([s.name for s in model.states])
for t in model.transitions:
    print(f"{t.kind:8} {t.from_name} -> {t.to_name}")
```

출력:

```
['active', 'waiver']
death    active -> death
death    waiver -> death
lapse    active -> lapse
lapse    waiver -> lapse
transfer active -> waiver
```

읽는 법:

- `kind` 은 `death` / `lapse` / `transfer` 셋 중 하나.
- **`to_state=None` (`-> death` / `-> lapse`)** = 목적지 상태가 없음 = 사망·해지로
  **in-force 에서 완전히 이탈**. 죽거나 해지하면 어느 상태로 "가는" 게 아니라 그냥
  나가므로, 목적지가 없습니다.
- **목적지 번호가 있는 `transfer` (`active -> waiver`)** = 여전히 유효하되 **다른
  상태로 이동**. 납입면제가 발동해 active 점유 일부가 waiver 로 옮겨가는 자리.

`Model.transitions` 의 각 원소는 `TransitionRecord` (출발·도착·종류 + 이름표만
담은 서술 카드) 이고, 위험률·현금흐름 같은 숫자는 담지 않습니다 — 상태기계의
전이 구조를 한눈에 보는 목록입니다.

## 전이의 두 종류

`Transition` 에는 성격이 다른 두 부류가 있습니다.

**확률 (rate) 전이** — 매달 위험률만큼 일부가 이동. 상태의 전이들은 선언 순서대로
경쟁위험 (competing decrement) 으로 적용됩니다: 앞 전이의 생존자에게 다음 전이가
차례로 뭅니다. 이 전이는 컴파일 시 **엣지 (edge)** 가 됩니다.

```python
Transition("mortality")                    # rate="mortality", to=None -> 이탈
Transition("waiver_incidence", "waiver")   # rate="waiver_incidence", to="waiver" -> 이동
```

**결정적 (deterministic) 전이** — 위험률이 아니라 **조건이 맞으면 확률 1 (전원)**
로 이동. 두 형태가 있고, 컴파일 시 **엣지가 아니라 상태별 스칼라** 로 떨어집니다:

- `at_premium_term=True` — 그 계약의 **납입만기** 달에 전원 이동 (예: 납입 끝나면
  active 를 통째로 paidup 으로). 달력 트리거.
- `after_sojourn_months=K` — 그 상태 **체류 K 개월** 도달 시 전원 이동 또는 종료
  (예: 요양 36 개월 지급 후 보장 종료). 경과 트리거.

```python
Transition(at_premium_term=True, to="paidup")     # 납입만기에 paidup 으로
Transition(after_sojourn_months=36, to=None)       # 체류 36 개월째 보장 종료
```

결정적 전이는 엣지가 없으므로, `sum_at_risk` 처럼 엣지 기반으로 도는 도구에는
잡히지 않습니다 (해당 상품은 현재 `sum_at_risk` / `state_reserve` 미지원 — 명시적
`NotImplementedError`).

## 경과 (sojourn) — 두 인자가 왜 둘 다 필요한가

회복률·재발률은 "그 상태에 **얼마나 오래 있었나** (체류시간, sojourn)" 에 크게
좌우됩니다 — 장해 직후엔 회복이 잦고, 만성화되면 거의 멈춥니다. 이를 표현하는
것이 **semi-Markov** 이고, 인자 둘이 협력합니다:

- **`State.sojourn_tracking_months = D`** — 그 상태의 점유를 "들어온 지 몇 개월"
  별 코호트 D 개로 나눠 추적. 매달 코호트가 한 칸씩 밀리고, 마지막 코호트가 그
  이상을 모두 흡수. `0` 이면 추적 안 함 (순수 Markov).
- **`Transition.sojourn_dependent = True`** — **이 전이의 위험률이** 코호트별로
  다름 (배열에 경과축이 하나 더 붙음).

둘이 별개인 이유: **"추적한다 ≠ 모든 위험률이 경과 의존"**. 한 상태가 지급상한
(`periodic_benefit_term_months`) 이나 결정적 종료 (`after_sojourn_months`) 때문에
코호트를 추적하지만, 그 상태의 사망률은 경과와 무관한 평평한 값일 수 있습니다.
또 한 상태 안에서 회복은 경과 의존이고 사망은 평평할 수도 있습니다 — `sojourn_
dependent` (전이 단위) 가 "어느 전이가 경과축 배열을 소비하는지" 를 구분합니다.

```python
from fastcashflow.multistate import is_semi_markov

di = Model(states=(
    State("active", pays_premium=True, transitions=(
        Transition("mortality"),
        Transition("ci_incidence", to="disabled"),
        Transition("lapse"),
    )),
    State("disabled", pays_periodic_benefit=True, sojourn_tracking_months=24, transitions=(
        Transition("mortality"),                                    # 평평
        Transition("recovery", to="active", sojourn_dependent=True),  # 경과 의존
    )),
))
print("semi-Markov:", is_semi_markov(di))
print("states:", [s.name for s in di.states])
```

출력:

```
semi-Markov: True
states: ['active', 'disabled']
```

회복 (`disabled -> active`) 은 `disabled` 상태 안에 `to="active"` 로 적습니다 —
전이의 출발지는 그 전이를 담은 상태이기 때문입니다. active ↔ disabled 왕복
(사이클) 이어도 됩니다 (점유는 전이행렬로 굴러가므로 방향 제약이 없음).

## 인자 레퍼런스

세 클래스의 인자 전부. 대부분은 기본값이 있어, 안 쓰면 안 보입니다.

**`Model`**

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `states` | (필수) | 상태 튜플 (순서 = 상태 번호) |
| `seating` | `(0,)` | 입력 상태코드 (`STATE_ACTIVE/WAIVER/PAIDUP`) 를 어느 상태 번호에 앉힐지 |

**`State`**

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `name` | (필수) | 상태 이름 |
| `pays_premium` | `False` | 이 상태에서 보험료를 내나 |
| `pays_periodic_benefit` | `False` | 매월 정기 급부 (장해소득 등) 를 주나 |
| `transitions` | `()` | 이 상태에서 나가는 전이들 |
| `sojourn_tracking_months` | `0` | 경과 코호트 추적 수 (>0 이면 semi-Markov) |
| `periodic_benefit_term_months` | `0` | 정기 급부 지급 상한 개월 (0 = 무한) |
| `mortality_rate` | `"mortality"` | 이 상태의 사망을 어느 위험률 이름으로 라우팅 |
| `death_benefit_factor` | `1.0` | 이 상태 거주자 사망급부 배수 |

**`Transition`**

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `rate` | `None` | 어느 위험률 이름으로 굴러가나 (basis 에서 조회되는 키) |
| `to` | `None` | 목적지 상태 이름 (`None` = 이탈) |
| `pays_lump_sum` | `False` | 전이 시 일시금 (`disability_benefit`) 을 주나 |
| `sojourn_dependent` | `False` | 이 전이 위험률이 경과에 따라 달라지나 |
| `after_sojourn_months` | `0` | 체류 K 개월 도달 시 결정적 (확률 1) 전이 |
| `at_premium_term` | `False` | 납입만기 시 결정적 전이 |

`mortality_rate` 로 상태별 사망률을 바꾸려면, basis 에 그 이름의 위험률을 함께
공급합니다 — 예: `State(mortality_rate="dth_disabled")` + `Basis(state_mortality_
annual={"dth_disabled": ...})`. 안 쓰면 기본 `"mortality"` 로 폴백해 모든 상태가
같은 사망률을 씁니다.

## 복잡함은 옵션

위 인자가 많아 보이지만, 실제로 대부분의 보장성 상품은 **번들 프리셋 한 줄** 로
끝납니다:

```python
basis = fcf.Basis(
    state_machine=Model.from_preset("ACTIVE_WAIVER"),   # 2-state: active / waiver
    mortality_annual=0.005, lapse_annual=0.05, discount_annual=0.03,
    ra_confidence=0.75, mortality_cv=0.10,
    coverages=(fcf.CoverageRate("DEATH", 0.005),),
)
```

`Model.presets()` 로 번들 목록을 보고, `Model.from_preset(name)` 으로 고릅니다
(`ACTIVE_WAIVER` = active / waiver, `ACTIVE_WAIVER_PAIDUP` = active / waiver /
paidup). 워크북에선 `segments` 시트의 `state_machine` 컬럼에 그 이름을 적으면
됩니다. 프리셋으로 충분하면 `sojourn_*` · `after_sojourn_months` ·
`mortality_rate` 같은 인자는 아예 건드리지 않습니다 (전부 기본값).

`sojourn_dependent` · `periodic_benefit_term_months` · `after_sojourn_months`
등은 장해·요양·재발처럼 **경과기간이 중요한 소수 상품** 에서만 켭니다 — 그
사용법은 [4.1 재진단암](../semi-markov/reincidence) · [4.2 장해소득보상](
../semi-markov/disability-income) · [4.3 장기요양](../semi-markov/long-term-care)
에서 실제 상품으로 다룹니다.

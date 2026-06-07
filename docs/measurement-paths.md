# 측정 경로와 지원 기능

fastcashflow는 같은 계약을 여러 **측정 경로**로 평가합니다 — 속도·상세도·측정모형이
다릅니다. 대부분의 기능은 모든 경로에서 동작하지만, 일부 메커니즘은 **궤적(`full=True`)
경로 전용**입니다 (고속 fused 경로는 속도를 위해 일부를 생략). 지원하지 않는 조합을 쓰면
엔진은 **조용히 틀린 값을 내는 대신 `NotImplementedError`(또는 생성 시 `ValueError`)로
거부**합니다 — "잘못된 입력이 그럴듯한 BEL을 만들지 않게 한다"는 원칙입니다.

## 측정 경로

| 경로 | 호출 | 특징 |
|---|---|---|
| GMM 궤적 | `fcf.gmm.measure(mp, basis)` (기본 `full=True`) | 전 시점 trajectory(`*_path`). **모든 기능 지원.** |
| GMM 고속 | `fcf.gmm.measure(mp, basis, full=False)` | 시점 0 headline 4 숫자, fused 커널 (빠름, CPU). |
| GMM 고속 GPU | `fcf.gmm.measure(mp, basis, full=False, backend="gpu")` | CUDA 커널. |
| 결산(보유계약) | `fcf.gmm.measure_inforce(mp, state, basis, full=)` | full/fast 규칙은 GMM과 동일. |
| PAA | `fcf.paa.measure(mp, basis)` | 단기 간편법. |
| VFA | `fcf.vfa.measure(mp, basis)` | 변액(계좌가치). |

`backend` / `discount_curve` 인자는 **고속 경로(`full=False`) 전용**입니다.

## GMM: `full=True` 전용 기능 (`full=False`는 거부)

이 기능들은 궤적 커널에서만 적용됩니다. 고속 fused 경로는 해당 입력이 비-기본값이면
명확히 거부합니다 (class 0으로 뭉개 silently-wrong BEL을 내는 대신).

| 기능 | 입력 | `full=True` | `full=False` (고속) |
|---|---|:---:|:---:|
| 직업/위험등급 | `ModelPoints.issue_class` (≠ 0) | ✓ | ✗ |
| 체증·계단 보험금 | `coverage_escalation_annual` / `coverage_step_month` | ✓ | ✗ |
| 상태별 사망보험금 | `State.death_benefit_factor` (≠ 1) | ✓ | ✗ |
| 결정적 전이 | `Transition(after_sojourn_months=…)` | ✓ | ✗ |

→ 위 중 하나라도 쓰면 **`full=True`(기본값)** 로 측정하세요. 안 쓰는 포트폴리오는
`full=False` 가 더 빠릅니다.

## 경로별 추가 제약

| 제약 | 내용 |
|---|---|
| GPU + semi-Markov | `backend="gpu"` 는 semi-Markov state model 미지원 → `backend="cpu"`. |
| VFA + `death_benefit_factor` | VFA 경로는 상태별 사망보험금 미지원. |
| VFA + 결정적 전이 | VFA 경로는 `Transition(after_sojourn_months=…)` 미지원. |
| VFA 확률적 + 계약별 최저보증이율 | `return_scenarios` 는 계약별로 다른 `minimum_crediting_rate` 미지원 (스칼라만). |
| stochastic + cost_of_capital + 곡선 | `measure_stochastic` 의 cost-of-capital RA 는 평탄(1-D) 할인율만; 기간별 곡선(2-D)은 confidence-level RA 에서만. |

## RA / Basis 차원의 제약 (경로 무관)

| 제약 | 내용 |
|---|---|
| `expense_cv` (GMM / PAA) | GMM / PAA 의 RA 는 비용위험 PV 를 넣지 않음 → `expense_cv ≠ 0` 이면 `NotImplementedError`. **VFA 의 RA 만** `expense_cv` 를 사용. |
| `settlement_pattern` + 할인곡선 | 둘을 함께 두면 Basis 생성 시 거부 (시변 정산 할인 미구현). 스칼라 `discount_annual` 과는 함께 사용 가능. |

## 요약

- **모르겠으면 `full=True`** — 모든 기능을 지원하고 trajectory(변동분해용)도 줍니다.
- **`full=False` 는 속도용** — 위 full-only 기능을 안 쓰는 포트폴리오에서.
- 지원하지 않는 조합은 **조용히 틀리지 않고 명확한 에러로** 막힙니다.

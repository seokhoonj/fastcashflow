# 자산·지급여력 4 — 요구자본 통합 (assess_solvency)

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **요구자본 (SCR)** 전체: 보험 (life underwriting) + 시장 + 신용 + 운영 모듈 집계
- 부채측 보험위험 SCR (`fcf.solvency.required_capital`) 과 자산측 시장 · 신용 모듈을 묶는 법
  (`fcf.assets.assess_solvency`, `aggregate_required_capital`)
- **regime-agnostic** 엔진 -- 호출은 그대로, `regime` 만 Solvency II / K-ICS 로 바꿈
:::

> **작성 예정 (skeleton)** — 윤곽만 잡혀 있습니다.

## 다룰 내용

- 보험위험 모듈: 충격 -> 재측정 -> 상관집계 (요약; 자세히는 쿡북 8.6)
- 시장 · 신용 모듈 (2 챕터) 과의 모듈간 집계
- `assess_solvency` 가 돌려주는 `SolvencyAssessment` -- 가용자본 + 모듈별 SCR + 비율
- v1 한계: 모듈간 분산 처리의 단순화 노트

## 관련 API / 쿡북

- `fcf.assets` — `assess_solvency`, `SolvencyAssessment`, `aggregate_required_capital`
- `fcf.solvency.required_capital`, `fcf.solvency.SII`, `fcf.solvency.KICS`
- 쿡북 8.6 [요구자본 (Solvency II / K-ICS)](../cookbook/workflow/required-capital)

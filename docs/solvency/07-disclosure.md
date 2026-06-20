# 자산·지급여력 7 — 공시 재현

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- IFRS17 부채 + 지급여력 **공시 숫자** 를 {{ fcf }} 산출과 맞추는 법
- 자산 · 가용자본 · 요구자본 · 비율을 공시 양식에 대응시키기
- Solvency II 와 K-ICS 공시의 차이
:::

> **작성 예정 (skeleton)** — 윤곽만 잡혀 있습니다.

## 다룰 내용

- 부채 (BEL / RA / CSM) 와 자산 (가용자본) 을 공시 항목에 매핑
- 요구자본 모듈별 분해의 공시 표현
- 대량해지 재보험이 반영된 지급여력비율의 공시 (6 챕터와 연결)
- 한국 보험사 공개 공시 (DART) 와의 대조 검증

## 관련 API / 쿡북

- 쿡북 8.9 [공시 재현 -- IFRS17 + K-ICS](../cookbook/workflow/disclosure)
- `fcf.assets.assess_solvency`, `fcf.mass_lapse_reinsurance.report`

# 자산·지급여력 1 — 자산 포트폴리오 · 시장가치

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 자산을 **시장가치** 로 평가하는 자료구조 (`fcf.assets.Portfolio`, `Equity` / `Property` / `Cash`)
- 보유자산별 평가 (`holding_value`) 와 포트폴리오 합계 (`portfolio_value`)
- 자산측이 왜 부채 엔진과 **직교하는 별도 pillar** 인지, 그리고 지급여력 그림에서 어디에 놓이는지
:::

> **작성 예정 (skeleton)** — 윤곽만 잡혀 있습니다.

## 다룰 내용

- `Portfolio` 구성: 채권 (할인) · 주식 · 부동산 · 현금
- 시장가치 평가와 할인율의 역할
- 부채 (BEL + RA) 와 나란히 놓기 위한 준비 -- 다음 챕터의 가용자본으로 이어짐

## 관련 API / 쿡북

- `fcf.assets` — `Portfolio`, `portfolio_value`
- 쿡북 8.8 [자산 - 가용자본 - 지급여력비율](../cookbook/workflow/solvency-balance-sheet)

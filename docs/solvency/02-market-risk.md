# 자산·지급여력 2 — 시장위험 SCR

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 자산측 **시장위험** 하위모듈: 금리 · 주식 · 부동산 · 환 · 신용 · 집중
- 각 충격을 자산 (그리고 금리는 순자산) 에 적용해 SCR 을 내는 법 (`fcf.assets.market_module_scr`)
- Solvency II 와 K-ICS 의 시장위험 calibration 차이
:::

> **작성 예정 (skeleton)** — 윤곽만 잡혀 있습니다.

## 다룰 내용

- 금리위험: 순자산 (자산 - 부채) 기준 (`net_interest_scr` / `net_interest_kics_scr`)
- 주식 (`equity_scr`, 유형별 + 0.75 상관) · 부동산 (`property_scr`)
- 환 (`fx_scr`) · 신용 (`credit_scr`) · 집중 (`concentration_scr`)
- 운영위험 (`operational_scr`)
- 시장모듈 집계 (`market_module_scr`)

## 관련 API / 쿡북

- `fcf.assets` — `market_module_scr`, `net_interest_scr`, `equity_scr`, `credit_scr`, ...
- 쿡북 8.6 [요구자본 (Solvency II / K-ICS)](../cookbook/workflow/required-capital)

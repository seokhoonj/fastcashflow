# 자산·지급여력 3 — 가용자본 (자산 − 부채)

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **가용자본** = 자산 시장가치 빼기 부채 (BEL + RA) (`fcf.assets.available_capital`)
- 부채 엔진의 산출 (BEL, RA) 이 자산측과 만나 순자산가치가 되는 자리
- 가용자본이 지급여력비율의 **분자** 라는 것
:::

> **작성 예정 (skeleton)** — 윤곽만 잡혀 있습니다.

## 다룰 내용

- `available_capital(asset_portfolio_value, bel, risk_margin)` 의 구성
- 부채측 (요구자본 트랙의 BEL / RA) 과 자산측 (1-2 챕터) 의 결합
- 왜 가용자본이 규제 대차대조표의 핵심인지 -- 다음 챕터의 비율로 이어짐

## 관련 API / 쿡북

- `fcf.assets` — `available_capital`, `asset_portfolio_value`
- 부채 튜토리얼 [BEL / RA / CSM](../tutorial/02-bel-ra-csm)

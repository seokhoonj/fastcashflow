# 자산 · 지급여력 — {{ fcf }}

:::{admonition} 이 트랙이 다루는 것
:class: tip

- **자산 포트폴리오와 시장가치** -- 자산을 시장가치로 평가하는 법 (`fcf.assets.AssetPortfolio`)
- **시장위험 SCR** -- 금리 · 주식 · 부동산 · 환 · 신용 · 집중 충격
- **가용자본** -- 자산 빼기 부채 (`fcf.assets.available_capital`)
- **요구자본 통합** -- 보험 + 시장 + 신용 + 운영 모듈 집계 (`fcf.assets.assess_solvency`)
- **지급여력비율** -- 가용자본 나누기 요구자본, Solvency II / K-ICS
- **대량해지 재보험 자본경감** -- 비비례 lapse-XL 로 lapse SCR 을 줄이는 법
- **공시 재현** -- IFRS17 부채 + 지급여력 공시 숫자 맞추기
:::

이 트랙은 부채 (IFRS 17) 튜토리얼의 **자산 · 지급여력 짝** 입니다. 부채 튜토리얼이
모델포인트에서 BEL / RA / CSM 까지 부채 한 쪽을 선형으로 쌓아 올렸다면, 이 트랙은
**대차대조표의 나머지 절반** -- 자산, 시장위험, 가용자본 -- 을 세우고, 둘을 묶어
요구자본과 지급여력비율을 냅니다. 마지막에 대량해지 재보험이 그 비율을 어떻게
움직이는지까지 봅니다.

{{ fcf }} 는 오랫동안 부채 엔진이었고 자산측은 `fcf.assets` 로 나중에 자랐습니다. 이
트랙은 그 자산측에 **서사** 를 줍니다 -- 흩어진 쿡북 레시피 (8.6 요구자본, 8.8
지급여력비율, 8.9 공시) 를 하나의 길로 잇습니다. 각 챕터는 해당 쿡북 레시피와
교차 링크됩니다.

:::{admonition} 뼈대 (skeleton)
:class: note
이 트랙은 현재 **뼈대만** 잡혀 있습니다. 각 챕터는 학습 목표와 다룰 내용의
윤곽만 있고, 실행 예제 (`fcf` 코드 + 출력) 는 순차적으로 채워집니다. 자산측
엔진 (`fcf.assets`) 과 대량해지 엔진 (`fcf.mass_lapse_reinsurance`) 은 이미
구현 · 테스트되어 있으므로, 남은 일은 서술과 예제입니다.
:::

:::{toctree}
:maxdepth: 1
:caption: 차례

01-asset-portfolio
02-market-risk
03-available-capital
04-required-capital
05-solvency-ratio
06-mass-lapse-reinsurance
07-disclosure
:::

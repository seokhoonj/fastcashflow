# 자산·지급여력 6 — 대량해지 재보험 자본경감

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- **대량해지 재보험 (lapse-XL)** -- 해지 SCR 의 꼬리를 재보사에 넘기는 비비례 재보험
- 출재사 관점: lapse SCR 을 줄여 지급여력비율을 끌어올리는 법 (`fcf.reinsurance.mass_lapse.cedant_solvency_relief`)
- 재보사 관점: treaty 를 해지 분포 F(L) 위에서 가격 · 측정 (`price_treaty`, `measure_assumed_treaty`)
- **Model 과 Engine 의 경계** -- {{ fcf }} 는 Engine, 분포 F(L) 는 plug-in (`LapseDistribution`)
- 같은 도구가 **Solvency II 와 K-ICS** 둘 다 -- 대량해지 충격 (40% / 30%) 은 regime 에서 자동
:::

> **작성 예정 (skeleton)** — 윤곽과 절 구성만 잡혀 있습니다. 엔진
> (`fcf.reinsurance.mass_lapse`) 은 이미 구현 · 테스트되어 있으므로, 남은 일은 서술과
> 실행 예제입니다.

## 6.1 왜 대량해지 재보험인가

수익성 좋은 (또는 해약환급금이 부채보다 큰) 계약에서는 **대량해지** 가 종종 lapse
SCR 의 biting 시나리오가 됩니다. 금리 급등 -> 경쟁상품 매력 증가 -> 저축성 해지 급증
이 그 방아쇠입니다. 이 꼬리위험을 재보사에 넘기면 요구자본이 줄고 지급여력비율이
오릅니다.

## 6.2 treaty 구조 -- 손실밀도와 layer

- 손실밀도 `S = sum max(0, 해약값 - BEL)` (per-policy, DR Art 142(6))
- `LapseXL(attachment, detachment)` -- BE 초과 해지율 구간의 손실을 cover
- recovery = `S x clip(L - AP, 0, capacity)` (선형, cliff-edge 없음)

## 6.3 출재사 자본경감

- `cedant_solvency_relief` -- gross 대 net 대량해지 SCR
- lapse SCR = `max(up, down, mass)` 이라 net 이 다음 biting leg (up/down) 에서 바닥
- counterparty default add-back (재보사 신용위험, DR Art 192/200) 과 risk margin 감소
- 워터폴: 경감 -> 신용 add-back -> risk margin -> total benefit

## 6.4 측정기간 (measurement period)

- `MeasurementPeriod` (12 개월 reset / rolling) 과 `windowed_claim`
- multi-year 사건 (20% + 20% / 2년) 을 12 개월 창이 놓치고 24 개월 창이 잡는 이유 (EIOPA Annex 3.9)
- rolling 중첩창의 high-water mark (footnote 10)

## 6.5 재보사 관점 -- 분포 · 가격 · 측정

- 해지 꼬리분포 `LapseTailDistribution.from_anchors` (공개 앵커 15% @ 1-in-30, 40% @ 1-in-200)
- `price_treaty` -- 기대회수 · 인수자본 · premium (premium 이 출력)
- `measure_assumed_treaty` -- 재보사측 IFRS17 BEL / RA / CSM

## 6.6 Model 과 Engine

- {{ fcf }} 가 제공하는 것 = **Engine** (분포를 받아 SCR · RA · CSM · premium 계산)
- 재보사의 진짜 IP = **Model** (분포 F(L) 생성기: 꼬리 데이터 · 채널 군집 · 경제-해지 링크)
- plug-in 계약: `LapseDistribution` 의 `survival(x) = P(L > x)` 하나만 구현하면 교체

## 6.7 Solvency II 대 K-ICS

- 같은 호출, regime 만 바꿈 -- 대량해지 충격 (SII 40% / K-ICS 30%) 은 regime 에서 자동
- 재보사 분포는 regime 앵커로 (K-ICS 는 30% @ 1-in-200)

## 6.8 한 장 리포트와 한계

- `report(...)` -- 출재사 경감 + 재보사 가격 + IFRS17 측정을 한 ASCII 리포트로
- 한계 노트: 표준공식의 ceteris-paribus 가 **해지 ↔ 시장위험 상관** (금리↑ 시 강제매각 + 해지급증
  동시) 을 놓침 -- 경제적 완전판은 Model (Layer 2) 영역

## 관련 API / 쿡북

- `fcf.reinsurance.mass_lapse` — `LapseXL`, `cedant_solvency_relief`, `price_treaty`,
  `measure_assumed_treaty`, `LapseTailDistribution`, `report`
- 쿡북 6.1 [비례 재보험 (quota share)](../cookbook/reinsurance/proportional)
- 쿡북 8.6 [요구자본 (Solvency II / K-ICS)](../cookbook/workflow/required-capital)

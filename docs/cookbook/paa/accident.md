# 10.1 단기 정산형 상해보험 (PAA)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- 단기 계약을 **PAA (Premium Allocation Approach, 보험료배분접근법)** 로
  측정하는 자리 — `paa.measure`. CSM 대신 미경과보험료식 잔액 (LRC) 을 굴린다
- 청구가 사고 즉시가 아니라 여러 달에 걸쳐 지급되는 **정산 패턴**
  (`settlement_pattern`, 청구 run-off) 을 산출기초에 거는 법
- 정산 패턴이 있으면 할인은 **스칼라** 여야 하는 이유 (할인 곡선과는 결합 불가)
- 보험료가 손익분기 아래라 손실요소 (loss component) 가 잡히는 onerous 단기
  계약의 모습
```

장기 보장성 계약은 GMM (General Measurement Model) 으로, 미실현 이익을 CSM
으로 들고 갑니다. 보험기간이 1년 안팎인 단기 계약은 IFRS 17 이 **PAA** 라는
간편법을 허용합니다 — 잔여보장부채 (LRC, Liability for Remaining Coverage) 를
미경과보험료처럼 굴리고, 보험금은 발생하면 발생손해부채 (LIC, Liability for
Incurred Claims) 로 따로 잡습니다. fastcashflow 는 이를 `paa.measure` 로
측정합니다.

## 상품 소개 — 단기 상해 + 청구 정산

단체상해보험은 보통 1년 갱신입니다. 사고가 나면 보험금이 사고 당월에 한꺼번에
나가지 않고 — 조사·심사·분할지급을 거쳐 — 여러 달에 걸쳐 정산됩니다. 그
**청구 run-off** 를 사고 당월/익월/.../ 의 지급 비율로 적은 것이 정산 패턴
입니다. 예를 들어 `[0.4, 0.3, 0.2, 0.1]` 은 발생액의 40% 를 당월, 30% 를
다음 달, 20% / 10% 를 그 뒤 두 달에 지급한다는 뜻입니다 (합 1).

정산이 미래로 밀리면 그만큼 할인이 먹어 발생손해부채의 현재가치가 줄어듭니다 —
IFRS 17 은 발생한 청구를 **각자의 지급 시점으로 할인** 하라고 합니다.

## 모델링 매핑 — PAA

```{list-table}
:header-rows: 1
:widths: 42 58

* - 자리
  - 무엇
* - `paa.measure(mp, basis)`
  - PAA 측정 (장기 보장형의 `gmm.measure` 가 아님)
* - `Basis.settlement_pattern`
  - 청구 run-off — 발생액을 당월부터 몇 달에 걸쳐 지급하는 비율 (합 1).
    `None` 이면 사고 당월 즉시 전액 지급
* - `Basis.discount_annual` (스칼라)
  - 평가 할인율. 정산 패턴과 함께 쓰려면 **스칼라** 여야 함 (곡선은 거부)
* - `PAAMeasurement.lrc`
  - 잔여보장부채 — 미경과보험료식 잔액
* - `PAAMeasurement.loss_component`
  - 손실요소 — 보험료가 보장원가에 못 미치는 onerous 계약에서 양수
```

## 한 포트폴리오 — 번들 PAA 샘플

fastcashflow 에 번들된 PAA 샘플은 12개월 단체상해 계약 둘입니다. 정산 패턴이
걸려 있고, 보험료가 손익분기 아래라 둘 다 onerous 입니다.

```python
import fastcashflow as fcf

basis = fcf.samples.basis("paa")          # 스칼라 할인 + 4개월 정산 패턴
mp    = fcf.samples.model_points("paa")   # 12개월 상해 계약 2건

print(f"settlement_pattern = {basis.settlement_pattern}")

val = fcf.paa.measure(mp, basis)
for i, mp_id in enumerate(mp.mp_id):
    print(f"{mp_id}  LRC = {val.lrc[i]:>6,.0f}   loss_component = {val.loss_component[i]:>12,.2f}")
```

출력:

```text
settlement_pattern = [0.4 0.3 0.2 0.1]
PA001  LRC =      0   loss_component =    31,926.63
PA002  LRC =      0   loss_component =    82,787.14
```

만기 시점의 LRC 는 0 (보장이 모두 경과) 이고, 두 계약 모두 손실요소가 양수
입니다 — 보험료가 정산까지 마친 보장원가에 못 미치는 onerous 블록입니다.

## 결과 읽기 — 정산이 부채를 줄인다

정산 패턴을 켜고 끄면 그 효과가 드러납니다. 청구를 미래로 미루면 할인이 먹어
발생손해의 현재가치가 작아지고, 그만큼 손실요소도 작아집니다.

```python
import dataclasses

immediate = dataclasses.replace(basis, settlement_pattern=None)  # 사고 당월 즉시 지급
v_imm = fcf.paa.measure(mp, immediate)
v_set = fcf.paa.measure(mp, basis)                               # 4개월 정산

print(f"즉시 지급   loss = {v_imm.loss_component.sum():>12,.2f}")
print(f"4개월 정산  loss = {v_set.loss_component.sum():>12,.2f}")
```

출력:

```text
즉시 지급   loss =   115,563.64
4개월 정산  loss =   114,713.77
```

청구를 4개월에 걸쳐 정산하면 손실요소가 115,563.64 에서 114,713.77 로
줄어듭니다 — 미뤄진 지급에 할인이 먹은 만큼입니다. PAA 의 발생손해부채는 발생한
청구를 **각자의 지급 시점으로 할인** 하므로, 같은 청구를 보는 `gmm.measure` 와
손실요소가 정확히 일치합니다.

## 변형

### revenue_basis — 보험수익 인식 기준

`paa.measure(mp, basis, revenue_basis="time")` (기본, B126(a) — 경과기간
비례) 또는 `revenue_basis="claims"` (B126(b) — 예상청구 비례) 로 보험수익
인식 패턴을 고릅니다. 단기 정액 보장은 보통 시간 비례면 충분합니다.

### 정산 패턴은 산출기초의 한 입력

샘플에서는 `settlement_tables` 시트 (`table_id`, `month`, `weight`) 가 정산
패턴을 담고, `segments` 시트의 `settlement_table` 칸이 세그먼트에 연결합니다.
회사 데이터에서는 청구 삼각형 (claims triangle) 의 지급 프로파일을 이 비율로
적습니다.

## 함정

### 함정 1 — `gmm.measure` 가 아니라 `paa.measure`

단기 계약을 GMM 으로 돌리면 CSM 을 굴리느라 단기 간편법의 LRC / LIC 구조가
나오지 않습니다. PAA 는 `paa.measure` 입니다.

### 함정 2 — 정산 패턴 + 할인 곡선은 결합 불가

`settlement_pattern` 은 **스칼라** `discount_annual` 과만 씁니다. 매년 다른
할인율 (per-year 곡선) 과 함께 주면 엔진이 거부합니다 — 각 정산분을 제 지급
시점으로 할인하려면 시점별 할인계수가 필요한데 (현재 미지원), 곡선의 첫해
값으로 뭉뚱그리면 틀린 값이 나오기 때문입니다. 샘플 PAA 기초가 스칼라 할인
(`0.04`) 을 쓰는 이유입니다.

## 인접 레시피

- [9.1 결산 / 보유계약 평가](../workflow/settlement) — 보유계약을 결산일 기준
  으로 재측정하는 워크플로.
- [8.2 검증 패턴](../workflow/validation) — 한 계약의 측정 경로를 손계산으로
  대조.
- `fcf.show_trace_paa(0, mp, basis)` — 이 계약의 LRC roll-forward, 보험수익
  인식, LIC, 손실요소를 트리로 확인 (PAA 전용 트레이서).

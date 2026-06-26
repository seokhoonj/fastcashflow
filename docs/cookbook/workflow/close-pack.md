# 9.4 결산팩 — 공시 명세서 조립 (close)

[9.1 결산 / 보유계약 평가](settlement) 는 보험계약집합 하나하나를 `gmm.settle`
로 굴려 **변동분석표** (정산 reconciliation) 를 냅니다. 결산의 마지막 한 걸음은
그 변동분석표들을 **IFRS 17 공시 명세서** — 재무상태표 (SoFP) · 보험금융손익 ·
(선택) 보험서비스손익 — 로 모으는 일입니다. `close` 가 그 조립을 하고,
`write_close_pack` 이 결과를 여러 시트의 엑셀 한 권으로 떨굽니다.

핵심은 **`close` 는 측정을 재계산하지 않는다**는 점입니다. 이미 나온 정산표를
보험계약집합 단위로 **집계** 할 뿐이라, 명세서의 모든 숫자는 위층 settle 의
숫자와 정확히 맞아떨어집니다.

## 최소 작동 예제 — 전 세그먼트 결산팩

```python
import numpy as np
import fastcashflow as fcf

basis = fcf.samples.basis()
book  = fcf.samples.model_points()
state = fcf.samples.inforce_state()

# 보험계약집합(= 상품 / 채널 세그먼트) 별로 결산 정산 -> movement / 변동분석표 수집
movements, recons, group_labels = [], [], []
for segment in basis.segments:
    rows = np.flatnonzero((book.product == segment[0]) & (book.channel == segment[1]))
    if rows.size == 0:
        continue
    mp       = book.subset(rows)  # 이 세그먼트의 보유계약
    st       = state.subset(np.flatnonzero(np.isin(state.mp_id, mp.mp_id)))
    valued   = fcf.apply_inforce_state(mp, st)  # 결산일 기준 재평가
    movement = fcf.gmm.settle(valued, st, basis.resolve(segment), period_months=12)
    movements.append(movement)                   # per-MP 상세 (상세 파일용)
    recons.append(fcf.reconcile([movement])[0])  # 보험계약집합 단위 집계
    group_labels.append("/".join(segment))

# 결산팩 조립 -- 재계산이 아니라 정산표의 집계
pack = fcf.close(recons, group_ids=group_labels)
print(pack)
```

출력:

```
IFRS 17 close pack -- 12-month period
  Net statement of financial position
    LRC excluding loss component           1,575,567       8,871,199      10,446,766
    Loss component                                 0       2,879,823       2,879,823
    Liability for incurred claims                  0               0               0
    Total                                  1,575,567      11,751,022      13,326,589
```

`pack.__str__` 은 순(net) 재무상태표만 요약해 보여 줍니다. 한 행이 **기초 →
변동 → 기말** 이고, 부채 캐리어모운트를 잔여보장요소 (LRC) 의 손실요소 제외분 /
손실요소 / 발생손해부채 (LIC) 로 가릅니다 (문단 99-101).

## 워크북에 실리는 시트

`ClosePackage` 는 명세서별 프레임으로 들고 있습니다 — `to_frames` 로 꺼냅니다.
보험서비스손익은 `close(..., reports=[...])` 로 `report` 를 함께 넘길 때만
더해집니다 (revenue / service expense 는 정산표가 아니라 보고서에서 옴).

```python
print(sorted(pack.to_frames()))
```

출력:

```
['finance', 'reconciliation', 'sofp']
```

## 재보험 차감 — net of reinsurance

보유 재보험계약을 같은 결산팩에 넣으면 재무상태표에 **재보험 보유** 종류와
**순(net)** 행이 생깁니다. 순액은 원수 발행 부채에서 재보험 회수가능액을
차감한 것 (문단 78) 인데, 엔진은 두 값을 같은 부호 프레임에서 다루므로 —
재보험 회수가능액은 **음(-)의 캐리어모운트** — 순액은 둘의 대수합입니다.

```python
from fastcashflow import InforceState

treaty    = fcf.samples.treaty()                 # 30% 비례재보험 (번들 약정)
segment   = ("TERM_LIFE_A", "FC")
seg_basis = basis.resolve(segment)
rows      = np.flatnonzero((book.product == segment[0]) & (book.channel == segment[1]))
mp        = book.subset(rows)
st        = state.subset(np.flatnonzero(np.isin(state.mp_id, mp.mp_id)))
valued    = fcf.apply_inforce_state(mp, st)

# 원수 발행 + 보유 재보험을 같은 보고기간으로 정산
issued = fcf.reconcile([fcf.gmm.settle(valued, st, seg_basis, period_months=12)])[0]

# 재보험 보유는 자기 CSM 을 가짐(원수 CSM 아님) -- 실무에선 재보험 마감파일이
# 나르고, 여기선 인셉션 측정의 기초 시점 CSM 으로 시드.
reins    = fcf.reinsurance.measure(mp, seg_basis, treaty=treaty)
opening  = np.asarray(st.elapsed_months) - 12
re_state = InforceState(
    mp_id=st.mp_id,
    elapsed_months=st.elapsed_months,
    count=st.count,
    prior_csm=reins.csm_path[np.arange(mp.mp_id.shape[0]), opening],  # 재보험 CSM 시드
    lock_in_rate=st.lock_in_rate,
    prior_count=st.prior_count,
)
held = fcf.reconcile([fcf.reinsurance.settle(
    valued, re_state, seg_basis, treaty=treaty, period_months=12)])[0]

pack_re = fcf.close([issued, held],
                    group_ids=["TERM_LIFE_A/FC issued", "TERM_LIFE_A/FC reins"])

import polars as pl
def total(kind):
    rows = pack_re.sofp.filter((pl.col("kind") == kind) & (pl.col("component") == "Total"))
    return float(rows["closing"][0])

print(f"issued        {total('Insurance contracts issued'):>14,.0f}")
print(f"reinsurance   {total('Reinsurance contracts held'):>14,.0f}")
print(f"net of reins  {total('Net'):>14,.0f}")
```

출력:

```
issued             3,415,627
reinsurance         -330,448
net of reins       3,085,180
```

재보험 보유가 음의 캐리어모운트 (-330,448 = 회수가능 자산) 라, 순 부채를
3,415,627 에서 3,085,180 으로 낮춥니다. (이 약정 시점에 재보험 CSM 이 이연된
순원가를 안고 있으면 캐리어모운트가 양(+) 이 되어 순액을 늘릴 수도 있습니다 —
어느 쪽이든 부호 일관된 대수합입니다.)

## 엑셀 산출물 — `write_close_pack`

`write_close_pack` 은 결산팩을 여러 시트의 `.xlsx` 한 권으로 직렬화합니다 —
표지(00_Index) · SoFP · 보험금융손익 · (있으면) 보험서비스손익 · reconciliation
명세 (감사용 `line_code` / IFRS 17 문단 anchor / memo 플래그 / 정렬순서를
join 으로 materialize). 모델포인트 단위 movement 는 엑셀 행 한계 (~1,048,576) 를
넘기 쉬워, `movements=` 로 넘기면 본 워크북 옆에 **parquet 상세 파일** (본 파일
옆에 따로 두는 보조 파일) 로 분리해 씁니다 — 정산 movement 하나당 한 파일.

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "close_pack_2026.xlsx"
    fcf.write_close_pack(pack, out, movements=movements)  # 워크북 + per-MP 상세 파일
    sidecars = sorted(p.name for p in Path(tmp).glob("*_per_mp_*.parquet"))
    print(f"{out.name}  (+{len(sidecars)} per-MP parquet sidecars)")
```

워크북 옆에 `close_pack_2026_per_mp_0.parquet` ... 처럼 정산 movement 하나당 한
상세 파일이 떨어집니다 (`movements=` 를 생략하면 집계 워크북만 씁니다).

## 감사 추적 컬럼 — `line_metadata`

엑셀에 박히는 참조 메타데이터 (각 라인의 기계 코드 · IFRS 17 문단 · P&L 메모
여부 · 정렬순서) 는 한 레지스트리에서 나옵니다. 사용자측 join 을 위해
`line_metadata` 로 노출됩니다.

```python
meta = fcf.line_metadata()
print(meta.columns)
```

출력:

```
['model', 'block', 'line', 'line_code', 'ifrs17_paragraph', 'is_memo', 'sort_order']
```

## 인접 레시피

- [9.1 결산 / 보유계약 평가](settlement) — 결산팩의 입력인 세그먼트별 정산표.
- [9.2 변동분해](movement) — 신계약 측정을 보고기간별로 가른 변동분석.
- [6.1 비례 재보험 (quota share)](../reinsurance/proportional) — 보유 재보험 측정.

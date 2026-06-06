# 1.1 한눈에 보기

쿡북의 어떤 챕터에서 어떤 파일을 어떤 함수로 다루는지, 전체 그림을 먼저
잡고 들어갑니다. 모든 후속 챕터의 코드가 아래 한 장의 흐름 안에서 움직입니다.

## 측정 흐름 — 두 입력이 만나는 곳

엔진은 두 개의 독립된 입력 — **계약** (`ModelPoints`) 과 **가정** (`Basis`) — 을
각각 만들어 `measure(model_points, basis)` 한 곳에서 합칩니다. 아래가 전체
토폴로지, 다음 절의 두 트리가 각 입력의 컬럼·함수 상세입니다.

```{mermaid}
flowchart TB
    subgraph IN["입력 — 두 독립 스트림"]
        direction LR
        subgraph CON["계약 · 무엇을 / 얼마나"]
            direction TB
            rmp["read_model_points<br/>read_inforce_policies<br/>read_vfa_model_points"]
            cmp["ModelPoints.single<br/>samples.model_points"]
            MP["ModelPoints<br/>benefits · premium · term<br/>issue_age · sex (선택자)<br/>calculation_methods"]
            rmp --> MP
            cmp --> MP
        end
        subgraph ASM["가정 · 어떤 율로"]
            direction TB
            rb["read_basis<br/>→ SegmentedBasis (dict)"]
            cb["Basis(...)<br/>samples.basis"]
            BS["Basis<br/>mortality_annual · lapse_annual<br/>discount_annual · CoverageRate<br/>expense_items"]
            rb --> BS
            cb --> BS
        end
    end
    M["measure(model_points, basis)<br/>basis 가 dict 면 세그먼트 자동 라우팅"]
    MP --> M
    BS --> M
    M --> GMM["gmm.measure<br/>GMMMeasurement"]
    M --> PAA["paa.measure<br/>PAAMeasurement"]
    M --> VFA["vfa.measure<br/>VFAMeasurement"]
    M --> REI["reinsurance.measure<br/>ReinsuranceMeasurement"]
    GMM --> RES["측정결과 *Measurement<br/>bel · ra · csm · loss · cashflows"]
    PAA --> RES
    VFA --> RES
    REI --> RES
    RES --> AN["분석 / 검증<br/>roll_forward · reconcile · transition<br/>group · report · show_trace<br/>plot_* · write_measurement"]
    classDef input fill:#f4f7fa,stroke:#9aa9b5,color:#24313a
    classDef stock fill:#eaf1f8,stroke:#547fa6,color:#17344e
    classDef step fill:#f7f2e8,stroke:#b38a45,color:#493617
    classDef margin fill:#eef6e8,stroke:#78a65a,color:#29421b
    class rmp,cmp,rb,cb input
    class MP,BS,RES stock
    class M,GMM,PAA,VFA,REI step
    class AN margin
```

- **합류점은 `measure` 하나.** 계약과 가정은 끝까지 따로 — 별도 `read_*`,
  별도 손-생성자로 만들어 측정 호출에서만 만납니다.
- **`read_basis` 는 dict** (`SegmentedBasis`), **`read_model_points` 는 단일
  개체.** 워크북은 여러 시트 + 세그먼트 라우팅이라 dict 로 풀립니다.
- **`.single` 은 `ModelPoints` 에만.** `Basis` 는 "여럿 중 하나" 축이 없어
  대응 생성자가 없고, 율을 숫자 / 표 / 콜러블로 바로 받는 식으로 단순화됩니다.
- **담보의 두 측면이 두 스트림에 나뉩니다** — *산출방식* (`CalculationMethod`,
  어떻게 계산되나) 은 계약 쪽, *율* (`CoverageRate`) 은 가정 쪽.

## 입력 파일과 사용자 함수

```{include} ../_shared/inputs_and_api.md
```

자세한 결산 모드 워크플로는 [튜토리얼 11장](../../tutorial/11-in-practice)
참조.

코드에서 reader 가 도는 순서가 그대로입니다 — `read_basis` 가
먼저 (engine 가정), 그 다음 `read_model_points` 가 policies / coverages /
calculation_methods 셋을 읽어 한 ModelPoints 개체로 묶습니다.

## 어느 챕터에서 어디까지 쓰나

| 챕터 | 사용하는 자리 |
|---|---|
| [담보와 산출방식 매칭](calculation-methods) | `calculation_methods.csv` 의 자리. 다섯 산출방식의 의미. |
| [담보별 산출로직](coverage-mechanics) | DEATH / MORBIDITY / DIAGNOSIS 의 kernel 알고리즘. |
| [정기보험](../simple/term-life) | `samples.export` → `read_*` → `measure` → `print` |
| [검증 패턴](../workflow/validation) | `gmm.trace` / `gmm.trace_bel_step` / `gmm.trace_csm_step` / `gmm.trace_diff` |
| [튜토리얼 11장](../../tutorial/11-in-practice) | 파일 입출력의 자세한 schema 와 결산 워크플로 |

각 챕터는 이 그림의 일부만 다룹니다. 챕터를 읽다 모르는 함수 / 파일이
나오면 위 두 트리에서 어디에 있는지 한 번 확인.

## 파일 구조가 처음이면 — `samples.export` 로 실물 보기

:::{admonition} 샘플 파일을 떨어뜨려 컬럼을 직접 확인하세요
:class: tip

자기 데이터를 fastcashflow 형식으로 맞추기 전에, **어떤 컬럼이 어떤 순서로
들어가는지** 실물로 보는 게 가장 빠릅니다. 패키지가 각 입력 파일의 작동하는
예시 한 세트를 디스크에 써 주는 `samples.export` 를 제공합니다:

```python
import fastcashflow as fcf

fcf.samples.export("samples", template="gmm", quiet=True)   # basis.xlsx + policies / coverages / calculation_methods (+ inforce)
```

써진 파일을 Excel / 텍스트 편집기로 열어 컬럼 이름과 한두 행의 값만 훑어보면,
후속 챕터의 `read_*` 가 무엇을 기대하는지 바로 감이 옵니다. `.csv` 외에
`.xlsx` / `.parquet` / `.feather` 로도 써집니다 (확장자로 결정).
:::

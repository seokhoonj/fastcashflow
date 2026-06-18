# 1.1 한눈에 보기

쿡북의 어떤 챕터에서 어떤 파일을 어떤 함수로 다루는지, 전체 그림을 먼저
잡고 들어갑니다. 모든 후속 챕터의 코드가 아래 한 장의 흐름 안에서 움직입니다.

## 측정 흐름 — 두 입력이 만나는 곳

엔진은 두 peer 객체 — **계약** (`ModelPoints`, 계산 대상) 과 **가정** (`Basis`,
계산 파라미터) — 을 각각 read 해 `measure(model_points, basis)` 에서 합칩니다.
메인 경로는 양쪽 다 **read → 객체 → measure** 로 대칭입니다.

:::{mermaid}
flowchart TB
    subgraph IN["입력 — 두 peer 객체"]
        direction LR
        subgraph C["계약 = 계산 대상"]
            direction TB
            rmp["read_model_points<br/>(policies + coverages + calc)"]
            MP["ModelPoints<br/>계약 = 행, 세그먼트 = 컬럼"]
            rmp --> MP
        end
        subgraph A["가정 = 계산 파라미터"]
            direction TB
            rb["read_basis<br/>(basis.xlsx)"]
            BS["BasisRouter<br/>(product, channel) → Basis 묶음"]
            rb --> BS
        end
    end
    MP --> M["measure(model_points, basis)<br/>세그먼트 키로 Basis 라우팅"]
    BS --> M
    M --> GMM["gmm.measure → GMMMeasurement"]
    M --> PAA["paa.measure → PAAMeasurement"]
    M --> VFA["vfa.measure → VFAMeasurement"]
    M --> REI["reinsurance.measure → ReinsuranceMeasurement"]
    GMM --> RES["측정결과 *Measurement<br/>bel · ra · csm · loss · cashflows"]
    PAA --> RES
    VFA --> RES
    REI --> RES
    RES --> AN["분석 / 검증<br/>roll_forward · reconcile · transition<br/>group · report · trace<br/>plot_* · write_measurement"]
    classDef input fill:#f4f7fa,stroke:#9aa9b5,color:#24313a
    classDef stock fill:#eaf1f8,stroke:#547fa6,color:#17344e
    classDef step fill:#f7f2e8,stroke:#b38a45,color:#493617
    classDef margin fill:#eef6e8,stroke:#78a65a,color:#29421b
    class rmp,rb input
    class MP,BS,RES stock
    class M,GMM,PAA,VFA,REI step
    class AN margin
:::

- **두 peer 객체, 합류점은 `measure`.** `mp = read_model_points(...)` /
  `basis = read_basis(...)` — 양쪽 다 read → 객체 → measure. 사용 모양은 대칭.
- **역할은 다릅니다.** `ModelPoints` = 계산 대상(계약 데이터), `Basis` = 계산
  파라미터. `read_basis` 는 **세그먼트별 Basis 묶음** (`BasisRouter`) 을
  돌려주고, `measure` 가 각 계약의 `(product, channel)` 로 해당 Basis 를 골라
  적용합니다. 단일 세그먼트면 `Basis` 하나를 그대로 넘겨도 전체에 균일 적용.
- **손-생성(코드)은 곁가지.** `ModelPoints(...)` / `ModelPoints.single` (1계약),
  `Basis(...)` (1세그먼트), `samples.*` — 검산 · 토이 · 민감도용. `.single` 은
  생성자가 배열 모양이라 ModelPoints 전용 sugar 고, `Basis` 는 생성자가
  스칼라 / 콜러블을 직접 받아 별도 sugar 가 없습니다.
- **담보의 두 측면** — **산출방법** (`CalculationMethod`) 은 계약 쪽,
  **율** (`CoverageRate`) 은 가정 쪽.

## 율(rate) 입력 형태

`Basis` 의 율 슬롯 (`mortality_annual` · `lapse_annual` · 담보 `rate` · 전이율
등) 은 **네 가지 형태**를 다 받습니다 — 회사가 가진 데이터 그대로 넣으면 됩니다:

| 형태 | 예 | 의미 |
|---|---|---|
| **스칼라** | `0.012` | 평탄 (전 성별·연령·기간 동일) |
| **배열** | `[0.08, 0.05, 0.03]` | 연율 by 정책연차 (1·2·3년차). `길이 × 12 ≥ term_months` (부족하면 에러) |
| **표** (polars / pandas) | `pl.DataFrame({"sex":…, "age":…, "rate":…})` | 컬럼에서 축 자동감지 (sex / age / issue_age / duration), 계약 키로 select |
| **함수** (callable) | `lambda s, a, d: …` | 분석적 · 복잡한 경우의 탈출구 |

대부분의 손계산 · 예제는 **스칼라**면 충분합니다. 한 값을 탈퇴·발생 두 슬롯에
공유하려면 변수로 묶어 양쪽에 넘깁니다 (1.3 참조):

```text
death = 0.012
fcf.Basis(
    mortality_annual = death,                        # 탈퇴
    coverages        = (fcf.CoverageRate("DEATH", death),),  # 발생 (같은 값 공유)
    lapse_annual = ..., discount_annual = ..., ...
)
```

성별 · 연령으로 갈리는 실무 위험률은 **표** (DataFrame) 또는 워크북의
`incidence_rate_tables` / `mortality_tables` 로 — 이때 계약의 `sex` · `issue_age`
가 그 표를 고르는 **선택자** 가 됩니다. 어떤 형태든 엔진 안에서는 하나의 율
함수로 정규화됩니다.

## 입력 파일과 사용자 함수

:::{include} ../_shared/inputs_and_api.md
:::

자세한 결산 모드 워크플로는 [튜토리얼 11장](../../tutorial/11-in-practice)
참조.

코드에서 reader 가 도는 순서가 그대로입니다 — `read_basis` 가
먼저 (engine 가정), 그 다음 `read_model_points` 가 policies / coverages /
calculation_methods 셋을 읽어 한 ModelPoints 개체로 묶습니다.

## 어느 챕터에서 어디까지 쓰나

| 챕터 | 사용하는 자리 |
|---|---|
| [담보와 산출방법 매칭](calculation-methods) | `calculation_methods.csv` 의 자리. 다섯 산출방법의 의미. |
| [담보별 산출방법](coverage-mechanics) | DEATH / MORBIDITY / DIAGNOSIS 의 kernel 알고리즘. |
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

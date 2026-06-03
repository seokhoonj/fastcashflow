# 1.1 한눈에 보기

쿡북의 어떤 챕터에서 어떤 파일을 어떤 함수로 다루는지, 전체 그림을 먼저
잡고 들어갑니다. 모든 후속 챕터의 코드가 이 두 트리 안에서 움직입니다.

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
| [보장 청구 메커니즘](coverage-mechanics) | DEATH / MORBIDITY / DIAGNOSIS 의 kernel 알고리즘. |
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

fcf.samples.export("samples", template="gmm")   # basis.xlsx + policies / coverages / calculation_methods (+ inforce)
```

써진 파일을 Excel / 텍스트 편집기로 열어 컬럼 이름과 한두 행의 값만 훑어보면,
후속 챕터의 `read_*` 가 무엇을 기대하는지 바로 감이 옵니다. `.csv` 외에
`.xlsx` / `.parquet` / `.feather` 로도 써집니다 (확장자로 결정).
:::

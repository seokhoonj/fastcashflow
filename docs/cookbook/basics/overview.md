# 1.1 한눈에 보기

쿡북의 어떤 챕터에서 어떤 파일을 어떤 함수로 다루는지, 전체 그림을 먼저
잡고 들어갑니다. 모든 후속 챕터의 코드가 이 두 트리 안에서 움직입니다.

## 입력 파일과 사용자 함수

```{include} ../_shared/inputs_and_api.md
```

자세한 결산 모드 워크플로는 [튜토리얼 11장](../../tutorial/11-in-practice)
참조.

**왜 네 파일로 나눠 있나** — 갱신 주기가 모두 다릅니다:

| 파일 | 누가 만드나 | 갱신 주기 |
|---|---|---|
| `policies.csv` / `inforce_*.csv` | 정책관리 시스템 | 일 / 분기 |
| `coverages.csv` | 정책관리 시스템 | 일 / 분기 |
| `benefit_patterns.csv` | 회사 (보험계리) | 연 1 회 미만 |
| `assumptions.xlsx` | 회사 (보험계리) | 분기 / 연 |

가장 자주 갱신되는 정책관리 데이터 (policies + coverages) 가 위로, 가장
드물게 갱신되는 회사 차원 결정 (benefit_patterns) 과 분기별 가정
(assumptions) 이 아래로.

## 어느 챕터에서 어디까지 쓰나

| 챕터 | 사용하는 자리 |
|---|---|
| [지급 패턴에 따른 계산방식의 결정](benefit-patterns-catalog) | `benefit_patterns.csv` 의 자리. 다섯 패턴의 의미. |
| [보장 청구 메커니즘](coverage-mechanics) | DEATH / MORBIDITY / DIAGNOSIS 의 kernel 알고리즘. |
| [정기보험 평가](../simple/term-life) | `save_sample_*` → `read_*` → `measure / value` → `print` |
| [검증 패턴](../workflow/validation) | `show_trace` / `show_bel_step` / `show_csm_step` / `show_trace_diff` |
| [튜토리얼 11장](../../tutorial/11-in-practice) | 파일 입출력의 자세한 schema 와 결산 워크플로 |

각 챕터는 이 그림의 일부만 다룹니다. 챕터를 읽다 모르는 함수 / 파일이
나오면 위 두 트리에서 어디에 있는지 한 번 확인.

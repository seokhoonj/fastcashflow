# 쿡북

쿡북은 fastcashflow 로 한국 시장의 다양한 상품을 평가하는 **실전 레시피**
모음입니다. 기본 튜토리얼이 IFRS 17 의 개념과 측정 흐름을 다룬다면,
여기서는 **"내 상품을 fastcashflow 로 어떻게 짜는가"** 에 답합니다.

읽는 방식은 **인덱스에서 골라 보기**입니다. 한 챕터 10-15분 안에
읽고, 끝의 작동 예제를 copy-paste 해 자기 데이터에 적용할 수 있도록
만들어졌습니다.

기본 튜토리얼 (`튜토리얼`) 의 IFRS 17 개념 (BEL, RA, CSM) 을 이해하고
오시면 가장 부드럽지만, 각 챕터는 **그 챕터만 봐도 충분히 이해되도록**
필요한 배경을 짧게 도입합니다.

## 쿡북의 구성

:::{list-table}
:header-rows: 1
:widths: 16 30 54

* - Part
  - 다루는 영역
  - 한 줄 요약
* - 기초
  - 엔진의 기본 동작 원리
  - 모든 상품 챕터의 사전 개념. CalculationMethod 의 의미와 엔진 안에서의
    청구 메커니즘.
* - 단순 상품
  - 상태 추적 없는 정액형
  - 가장 빠른 fast_path. 정기보험 / 사망+진단 / 면책·감액 같은 1-상태 상품.
* - Markov 상태
  - active / waiver / paid-up 같은 추가 상태
  - 보험료 납입면제, paid-up 분리 같은 상태 의존.
* - Semi-Markov 상태
  - 상태 안에서의 경과 시간 의존
  - 재진단 / 회복 / 등급 진행 — 코호트 추적이 필요한 영역.
* - 변액 (VFA)
  - 계좌가치 + 최저보증
  - 변액보험을 VFA로 측정. GMDB / GMAB의 intrinsic 과 시간가치 (TVOG).
* - 재보험 (출재)
  - 보유 재보험계약 측정
  - 비례 재보험 (quota share) 을 일반모형으로. 전가위험과 순원가 / 이익 (CSM).
* - I/O (Excel 워크북)
  - 데이터 입출력
  - 회사 워크북을 fastcashflow 가 읽는 형식으로 맞추는 자리.
* - 분석 / 검증
  - 시나리오 / 손계산 검증
  - 가정을 흔들어 보고, 결과의 한 항씩 풀어 보는 워크플로 도구.
* - 결산 워크플로
  - 보유계약 정산 + 변동분해
  - 분기말 기초 → 기말 정산 (`gmm.settle`) 과 변동의 행별 귀속
    (이자 / 경험 / 상각 / 손실요소).
* - 단기 (PAA)
  - 보험료배분접근법
  - 1년 안팎 단기 계약을 PAA로. LRC / LIC, 청구 정산 패턴 (settlement_pattern).
* - 확장 로드맵 (미구현)
  - 미구현 엔진 기능의 설계 노트
  - 아직 코드에 없는 기능의 설계 스케치. 실행 레시피가 아니라 로드맵.
:::

기초 → 단순 → Markov → Semi-Markov 의 순서는 **학습 곡선**입니다.
하지만 회사 상품에 해당하는 챕터로 바로 점프해도 됩니다 — 각 챕터는
필요한 사전 개념을 짧게 도입하고 시작합니다.

## 챕터 목록

### 1. 기초

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 1.1
  - [한눈에 보기](basics/overview)
  - 네 갈래의 입력 파일 (policies / coverages / calculation_methods /
    basis) 과 fastcashflow 사용자 API 의 트리 구조. 후속 챕터를
    어디서 어떻게 호출하는지 미리 그림.
* - 1.2
  - [담보와 산출방법 매칭](basics/calculation-methods)
  - 5 종 산출방법 (DEATH / MORBIDITY / DIAGNOSIS / ANNUITY / MATURITY) 의
    의미. 담보별 산출방법 (`calculation_methods.csv`) 작성.
* - 1.3
  - [탈퇴와 발생](basics/mortality-roles)
  - 계약에서 빠지는 일(탈퇴, `mortality_annual`)과 보험금 사유가 생기는 일
    (발생, DEATH rate)이 처음부터 다른 개념인 이유. 단순 사망보험에서만 둘이
    같은 숫자가 되는 까닭과, 두 슬롯에 같은 callable 을 넘기는 입력 패턴.
* - 1.4
  - [담보별 산출방법](basics/coverage-mechanics)
  - DEATH 의 공유 `inforce` vs DIAGNOSIS 의 `undiagnosed` 풀.
    같은 식이 두 자리에 작동하는 이유.
* - 1.5
  - [할인율 곡선 구성 — Smith-Wilson](basics/discount-curve)
  - 시장금리(국고채)에서 `discount_annual` 곡선을 `fcf.smith_wilson` 으로 구성.
:::

### 2. 단순 상품

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 2.1
  - [정기보험](simple/term-life)
  - 사망 단독 정기보험. fast_path. BEL / RA / CSM의 의미와 부호.
* - 2.2
  - [사망 + 단순 진단 일시금](simple/death-diagnosis)
  - 진단 담보 추가. 면책 / 감액 없는 간단한 결합.
* - 2.3
  - [다종 진단 + 면책 / 감액](simple/diagnosis-rules)
  - 가입 90일 면책 / 가입 2년 감액. coverage rule (담보 룰 축).
* - 2.4
  - [갱신형 보험과 계약의 경계](simple/renewable)
  - IFRS 17 Sec. 34 계약의 경계 — 차기갱신 vs 최종만기. `contract_boundary_months`
    로 측정 범위를 차기갱신에서 끊기.
:::

### 3. Markov 상태

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 3.1
  - [보험료 납입면제 (waiver)](markov/waiver)
  - `STATE_MODELS["WAIVER"]` 입문. active → waiver 진입.
* - 3.2
  - [paid-up 분리 (3-state)](markov/paid-up)
  - active / waiver / paidup 을 각각 별도 state 로. 납입후 해지율 점프.
:::

### 4. Semi-Markov 상태

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 4.1
  - [재진단암](semi-markov/reincidence)
  - 한국 시장 highlight. 1차/2차 진단 일시금, 재진단 면책기간. Semi-Markov
    (상태 경과 의존) 의 첫 챕터.
* - 4.2
  - [장해소득보상 (DI)](semi-markov/disability-income)
  - 매월 장해소득 + duration-since-disabled 의존 회복률. 회복 re-entry 와
    disabled life reserve (DLR).
* - 4.3
  - [간병 / 치매 (LTC)](semi-markov/long-term-care)
  - 진단금 일시금 + 보증한도 월정액 (`periodic_benefit_term_months`) + 간병상태 상승
    사망률 (`State.mortality_rate_name`). 상태지속 정액 보장의 sojourn 한도.
:::

### 5. 계좌형 — 변액 (VFA) · 유니버설

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 5.1
  - [변액보험 최저보증 — 결정론 측정](variable/gmdb-gmab)
  - 계좌가치 + 최저보증. `vfa.measure` 결정론 측정, 보증의 intrinsic value 를
    GMDB / GMAB 로 분리.
* - 5.2
  - [최저보증의 시간가치 (TVOG)](variable/gmdb-gmab-tvog)
  - 같은 계약에 `return_scenarios` 를 넣어 보증의 시간가치를 측정.
    intrinsic 대 시간가치 분해.
* - 5.3
  - [적립이율 보증 (크레딧 floor)](variable/crediting-floor)
  - 계좌를 매월 `max(수익률, 보증이율)` 로 떠받치는 세 번째 보증. 0% 원금보존,
    `vfa.tvog`, 월 래칫이 비싼 이유.
* - 5.4
  - [유니버설 (적립 방식)](account/universal-life)
  - 유니버설 = 세 번째 축 (적립 방식). 계좌-백 사망담보, 금리연동형은
    `gmm.measure` / 변액은 `vfa.measure`, COI on NAR, `gmm.trace` 검산.
:::

### 6. 재보험

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 6.1
  - [비례 재보험 (quota share)](reinsurance/proportional)
  - 보유 quota-share 재보험계약 측정. 전가위험 (RA) 과 순원가 / 이익 (CSM).
:::

### 7. I/O (Excel 워크북)

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 7.1
  - [워크북 — 단일 segment](io/workbook-single)
  - `basis.xlsx` 의 매 시트 / 매 컬럼 자세히. 사용자 진입점.
* - 7.2
  - [워크북 — 다중 segment / 다종 상품](io/workbook-multi)
  - `measure` + 상품 / 채널 별 다른 StateModel.
:::

### 8. 분석 / 검증

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 8.1
  - [시나리오 / 민감도 분석](workflow/sensitivity)
  - rate 함수 교체로 mortality +10% 등의 효과 측정. CSM 흡수 / onerous 전환,
    gmm.trace_diff.
* - 8.2
  - [검증 패턴](workflow/validation)
  - 한 계약의 BEL / CSM 계산 경로 추적. 손계산 매칭, shock 전파, residual 검증.
* - 8.3
  - [수익성 분석 / profit-testing](workflow/profit-testing)
  - 측정 결과에서 NBV / 마진 / 이익 시그니처 / IRR 추출. 전통형 (통계) NLP 준비금과
    이익원천 (보험료차 / 이자차) emergence — `fcf.pricing`.
* - 8.4
  - [전통형 금리보증 비용 (TVOG)](workflow/interest-guarantee)
  - 최저보증이율의 비용을 intrinsic / 시간가치로 분해. `statutory_reserve` (준비금) +
    `esg.simulate` (금리시나리오) 를 엮은 `interest_guarantee_tvog`.
:::

### 9. 결산 워크플로

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 9.1
  - [결산 / 보유계약 평가](workflow/settlement)
  - 분기말 마감파일 한 장으로 Sec. 44 기초 → 기말 정산. `gmm.settle`,
    변동분석표 (`reconcile`), 분기 체이닝 (`closing_inputs`), 진단 뷰
    (`gmm.measure_inforce`), 단기 계약의 `paa.settle` (Sec. 55(b)).
* - 9.2
  - [변동분해](workflow/movement)
  - 신계약 측정을 보고기간별로 잘라 BEL / CSM 움직임을 미래서비스 / 이자 /
    상각으로 귀속. `roll_forward` / `reconcile`, 가정변경 (`revised`) /
    경험 (`actual_inforce`).
* - 9.3
  - [손실부담 계약과 경험조정의 결산](workflow/onerous-settle)
  - settle 의 고급 라인. 손실요소 체계적 배분 (Sec. 50(a)-52), 발생손해부채
    (`settlement_pattern`, Sec. 40(b)), 보험료 (Sec. B96(a)) / 투자요소
    (Sec. B96(c)) 경험조정.
* - 9.4
  - [결산팩 — 공시 명세서 조립](workflow/close-pack)
  - 세그먼트별 정산표를 IFRS 17 공시 명세서로 조립. `close` (SoFP /
    보험금융손익 / 보험서비스손익), 재보험 차감 (Sec. 78), 멀티시트 엑셀
    (`write_close_pack`), 감사 컬럼 (`line_metadata`).
* - 9.5
  - [샘플 워크북으로 결산하기](workflow/sample-walkthrough)
  - 번들 샘플 엑셀을 시트·컬럼 하나하나 짚으며 결산파일까지 추적. `segments` /
    위험률 / 계약 파일 / `inforce_state` → `settle` → `close` →
    `write_close_pack` 의 입력-출력 매핑.
:::

### 10. 단기 측정 (PAA)

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 10.1
  - [단기 정산형 상해보험](paa/accident)
  - 1년 단기 계약을 `paa.measure` 로. LRC / 손실요소, 청구 정산 패턴
    (`settlement_pattern`) 과 스칼라 할인.
:::

### 11. 확장 로드맵 (미구현)

:::{list-table}
:header-rows: 1
:widths: 8 28 64

* - 번호
  - 챕터
  - 다루는 것
* - 11.1
  - [⚠ 동적해지율 엔진설계](design/dynamic-lapse)
  - 시나리오 / moneyness 에 반응하는 해지율. **⚠ 미구현** — 정적 격자에서 루프
    내 평가로 옮기는 설계 스케치. 실행 불가.
:::

## 모든 챕터의 공통 구조

각 챕터는 같은 7 섹션으로 구성됩니다. 사용자가 한 챕터를 익히면 다른
챕터에서도 같은 위치에 같은 종류의 정보를 찾을 수 있습니다.

1. **상품 소개** — 한국 시장에서 이 상품이 어떻게 팔리는가, 어떤 보장
2. **모델링 매핑** — fastcashflow 의 어떤 API 가 상품의 어떤 mechanic 에 대응하는가
3. **최소 작동 예제** — copy-paste 가능한 Python 코드. 즉시 실행
4. **결과 해석** — BEL / RA / CSM 값이 무엇을 의미하는가
5. **변형** — 회사 / 채널 / 상품 세대 별 차이는 어떻게 다루는가
6. **함정 / 검증** — 흔한 실수, 손계산으로 확인하는 방법
7. **인접 레시피** — 관련된 다른 챕터와 기본 튜토리얼의 어느 장

기초의 개념 챕터는 상품 챕터와 성격이 달라 위 7 단계를
강제하지 않습니다 — 개념의 정의 → 사례 → 함정 의 흐름을 따릅니다.
**확장 로드맵** 챕터는 레시피가 아니라 미구현 기능의 설계 노트라, 실행
예제 없이 배경 → 설계 결정 → 제안 → 검증·순서 의 흐름을 따릅니다.

## 코드 실행 환경

모든 챕터는 다음 환경을 가정합니다:

```python
# Python 3.10 이상
# fastcashflow 설치
# pip install git+https://github.com/seokhoonj/fastcashflow.git
```

각 챕터 코드 블록은 위의 `import` 구문부터 출력 (`print`) 까지 전체를
포함합니다 — 그대로 복사해서 실행하면 됩니다.

:::{toctree}
:hidden:
:caption: 1. 기초

basics/overview
basics/calculation-methods
basics/mortality-roles
basics/coverage-mechanics
:::

:::{toctree}
:hidden:
:caption: 경제적 가정 — 할인율 곡선 · 시나리오

basics/discount-curve
basics/scenario-generation
:::

:::{toctree}
:hidden:
:caption: 2. 단순 상품

simple/term-life
simple/death-diagnosis
simple/diagnosis-rules
simple/renewable
simple/escalating-benefits
simple/deferred-annuity
:::

:::{toctree}
:hidden:
:caption: 3. Markov 상태

markov/waiver
markov/paid-up
:::

:::{toctree}
:hidden:
:caption: 4. Semi-Markov 상태

semi-markov/reincidence
semi-markov/disability-income
semi-markov/long-term-care
:::

:::{toctree}
:hidden:
:caption: 5. 계좌형 — 변액 (VFA) · 유니버설

variable/gmdb-gmab
variable/gmdb-gmab-tvog
variable/crediting-floor
account/universal-life
:::

:::{toctree}
:hidden:
:caption: 6. 재보험

reinsurance/proportional
:::

:::{toctree}
:hidden:
:caption: 7. I/O (Excel 워크북)

io/workbook-single
io/workbook-multi
:::

:::{toctree}
:hidden:
:caption: 8. 분석 / 검증

workflow/sensitivity
workflow/validation
workflow/profit-testing
workflow/interest-guarantee
:::

:::{toctree}
:hidden:
:caption: 9. 결산 워크플로

workflow/settlement
workflow/movement
workflow/onerous-settle
workflow/close-pack
workflow/sample-walkthrough
:::

:::{toctree}
:hidden:
:caption: 10. 단기 측정 (PAA)

paa/accident
:::

:::{toctree}
:hidden:
:caption: 11. 확장 로드맵 (미구현)

design/dynamic-lapse
:::

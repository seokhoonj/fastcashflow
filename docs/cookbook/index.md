# 쿡북

쿡북은 {{ fcf }} 로 한국 시장의 다양한 상품을 평가하는 **실전 레시피**
모음입니다. 기본 튜토리얼이 IFRS 17 의 개념과 측정 흐름을 다룬다면,
여기서는 **"내 상품을 {{ fcf }} 로 어떻게 짜는가"** 에 답합니다.

읽는 방식은 **인덱스에서 골라 보기**입니다. 한 챕터 10-15분 안에
읽고, 끝의 작동 예제를 copy-paste 해 자기 데이터에 적용할 수 있도록
만들어졌습니다.

## 누구를 위한 자료인가

- **사용자** — 자사 상품을 {{ fcf }} 로 평가하려는 실무 actuary
- **검토 / 검증 담당자** — 평가 엔진이 어떻게 동작하는지 확인하려는 분
- **신규 도입을 검토하는 분** — 우리 회사 상품군이 {{ fcf }} 로
  표현 가능한지 사례로 확인

기본 튜토리얼 (`튜토리얼`) 의 IFRS 17 개념 (BEL, RA, CSM) 을 이해하고
오시면 가장 부드럽지만, 각 챕터는 **그 챕터만 봐도 충분히 이해되도록**
필요한 배경을 짧게 도입합니다.

## 4 단계 — 쉬운 것부터 어려운 것 순서

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - Part
  - 다루는 영역
  - 한 줄 요약
* - I
  - 기초 (Basics)
  - 상태 추적 없는 단순 상품. 사망 단독 / 진단 일시금. {{ fcf }} 의
    가장 빠른 경로 (fast_path).
* - II
  - Markov Process
  - active / waiver / paid-up 같은 추가 상태가 있는 상품.
    보험료 납입면제, 다종 진단담보, 면책 / 감액.
* - III
  - Semi-Markov Process
  - 상태에 들어온 후 경과 시간에 의존하는 비율 — 재진단 / 회복 / 등급
    진행. 코호트 추적이 필요한 영역.
* - IV
  - 기법 / 워크플로
  - 상품 무관. Excel 자동화 / 시나리오 / 검증 / 가정 레이어 등.
```

Part I → IV 의 순서는 **학습 곡선**입니다. 하지만 인덱스에서 자사 상품에
해당하는 챕터로 바로 점프해도 됩니다 — 각 챕터는 필요한 사전 개념을
짧게 도입하고 시작합니다.

## 챕터 목록

### Part I. 기초

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - 챕터
  - 제목
  - 다루는 것
* - 1
  - [정기보험 평가](01-term-life)
  - 사망 단독 정기보험. fast_path 경로. BEL/RA/CSM 의 의미와 부호.
* - 2
  - 사망 + 단순 진단 일시금 (작성 예정)
  - 진단 rider 추가. 면책 / 감액 없는 간단한 결합.
```

### Part II. Markov Process

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - 챕터
  - 제목
  - 다루는 것
* - 3
  - 보험료 납입면제 (waiver) (작성 예정)
  - `STATE_MODELS["WAIVER"]` 입문. active → waiver 진입.
* - 4
  - 다종 진단담보 + 면책 + 감액 (작성 예정)
  - 가입 90일 면책 / 가입 2년 감액. coverage rule 본격 활용.
* - 5
  - paid-up 분리 (3-state) (작성 예정)
  - active / waiver / paidup 을 각각 별도 state 로.
```

### Part III. Semi-Markov Process

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - 챕터
  - 제목
  - 다루는 것
* - 6
  - 재진단암 보험 (작성 예정)
  - 한국 시장 highlight. 1차/2차 진단 일시금, 재진단 면책기간.
* - 7
  - 장해소득보상 (DI) (작성 예정)
  - 매월 장해소득 + duration-since-disabled 의존 회복률.
```

### Part IV. 기법 / 워크플로

```{list-table}
:header-rows: 1
:widths: 8 30 60

* - 챕터
  - 제목
  - 다루는 것
* - 8
  - Excel 워크북 — 단일 segment (작성 예정)
  - assumptions.xlsx 의 매 시트 / 매 컬럼 자세히. 사용자 진입점.
* - 9
  - Excel 워크북 — 다 segment / 다 상품 (작성 예정)
  - value_segmented + 상품 / 채널 별 다른 StateModel.
* - 10
  - 시나리오 / 민감도 분석 (작성 예정)
  - rate 함수 교체로 mortality +10% 등의 효과 측정.
* - 11
  - 검증 패턴 (작성 예정)
  - 손계산 + measure↔value parity. 결과 신뢰 빌드업.
```

## 모든 챕터의 공통 구조

각 챕터는 같은 7 섹션으로 구성됩니다. 사용자가 한 챕터를 익히면 다른
챕터에서도 같은 위치에 같은 종류의 정보를 찾을 수 있습니다.

1. **상품 소개** — 한국 시장에서 이 상품이 어떻게 팔리는가, 어떤 보장
2. **모델링 매핑** — {{ fcf }} 의 어떤 API 가 상품의 어떤 mechanic 에 대응하는가
3. **최소 작동 예제** — copy-paste 가능한 Python 코드. 즉시 실행
4. **결과 해석** — BEL / RA / CSM 값이 무엇을 의미하는가
5. **변형** — 회사 / 채널 / 상품 세대 별 차이는 어떻게 다루는가
6. **함정 / 검증** — 흔한 실수, 손계산으로 확인하는 방법
7. **인접 레시피** — 관련된 다른 챕터와 기본 튜토리얼의 어느 장

## 코드 환경 — 한 번 준비

모든 챕터는 다음 환경을 가정합니다:

```python
# Python 3.10 이상
# fastcashflow 설치
# pip install git+https://github.com/seokhoonj/fastcashflow.git
```

각 챕터 코드 블록은 위의 `import` 구문부터 출력 (`print`) 까지 전체를
포함합니다 — 그대로 복사해서 실행하면 됩니다.

```{toctree}
:hidden:
:caption: 목차

01-term-life
01-term-life-modified
```

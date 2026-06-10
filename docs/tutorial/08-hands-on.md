# 8장. 직접 실행하기

```{admonition} 이 장에서 배우는 것
:class: tip

- fastcashflow 설치하기
- 코드를 어디에 쓰고 어떻게 실행하는가
- 모델포인트와 산출기초를 코드로 만들기
- measure()로 측정하고 BEL·RA·CSM 읽기
- measure(full=False)와 샘플 데이터로 대규모 평가하기
```

1장부터 7장까지, 보험계약부채를 측정한다는 것이 무엇인지 — 추정,
할인, 위험조정, 이익 분리 — 를 손으로 따라왔습니다. 이 장에서는 그
계산을 fastcashflow 엔진으로 직접 실행해 봅니다.

파이썬에 익숙하지 않아도 괜찮습니다. 코드는 한 줄씩 풀어 설명하고,
5~7장에서 손으로 구한 바로 그 계약을 그대로 다룹니다. 끝까지 따라오면
엔진이 손계산과 같은 답을 내놓는 것을 직접 확인하게 됩니다.

## 8.1 설치

fastcashflow는 **Python 3.10 이상**에서 동작합니다. 먼저 파이썬이
깔려 있는지 확인합니다. 윈도우는 명령 프롬프트, macOS·리눅스는
터미널을 열고 다음을 입력하세요.

```
python --version
```

`Python 3.10` 이상이 보이면 됩니다. 보이지 않으면 python.org에서
파이썬을 먼저 설치합니다.

이제 fastcashflow를 설치합니다. **pip**는 파이썬 패키지를 내려받아
설치해 주는 도구로, 파이썬과 함께 깔려 있습니다. fastcashflow는 아직
PyPI에 올라가 있지 않아 GitHub에서 직접 받습니다. 같은 창에 입력하세요.

```
pip install git+https://github.com/seokhoonj/fastcashflow.git
```

numpy·numba·polars·matplotlib 같은 핵심 의존 패키지는 자동으로 들어옵니다.

```{admonition} 가상환경
:class: note

여러 프로젝트를 다룬다면 프로젝트마다 **가상환경**(virtual
environment)을 따로 두는 것이 좋습니다. 패키지들이 서로 충돌하지 않게
해 주는 격리된 공간입니다. 지금은 없이 진행해도 튜토리얼을 따라오는 데
지장은 없습니다.
```

설치가 끝났으면 코드를 실행할 차례입니다. 이 장의 파이썬 코드 블록을
만나는 순서대로 한 파일 — 이를테면 `run.py` — 에 이어 붙인 뒤,
터미널에서 `python run.py`로 실행하면 됩니다. 한 줄씩 결과를 보고
싶다면 Jupyter 노트북 같은 대화형 환경을 써도 됩니다.

## 8.2 입력 만들기

엔진의 입력은 둘뿐입니다(1.6절) — **산출기초**와 **모델포인트**.

먼저 두 줄로 필요한 도구를 불러옵니다.

```python
import numpy as np
import fastcashflow as fcf
```

`import`는 다른 라이브러리를 불러오는 명령입니다. `numpy`는 숫자
배열을 다루는 라이브러리이고, `as np`는 앞으로 짧게 `np`라 부르겠다는
뜻입니다. fastcashflow는 `fcf`로 부릅니다.

산출기초는 `Basis`로 만듭니다. 5~7장 예제의 가정을 그대로
옮깁니다.

```python
# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda sex, issue_age, duration: np.full(
    issue_age.shape, 1 - (1 - 0.01) ** 12,
)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda sex, issue_age, duration: np.full(duration.shape, 0.0)

# 산출기초
basis = fcf.Basis(
    mortality_annual = death_fn,         # 보유계약 사망률 (위 death_fn)
    lapse_annual     = lapse_fn,         # 해지율 (해지 없음)
    discount_annual  = 1.005 ** 12 - 1,  # 연 할인율 (월 0.5% 의 연 환산)
    ra_confidence    = 0.75,             # 위험조정 신뢰수준 75%
    mortality_cv     = 0.10,             # 사망률 변동계수 10%
    coverages        = (
        fcf.CoverageRate("DEATH", death_fn),  # 사망 보장 1종 (청구 rate = death_fn)
    ),
)
```

먼저 `death_fn`은 사망률 함수입니다. 숫자 하나가 아니라 **함수**로 두는
까닭은 사망률이 성별·나이·경과에 따라 달라질 수 있기 때문이죠(3.2절).
`lambda`는 간단한 함수를 한 줄로 적는 파이썬 문법인데, 여기서는
성별·나이·경과와 상관없이 늘 같은 값을 돌려주는 함수입니다. 5장의
월 사망률 1%를 연으로 환산하면 `1 - (1 - 0.01) ** 12 ≈ 0.1136`
(연 11.36%)이고, 엔진이 내부에서 다시 constant-force 방식으로 월
사망률 1%로 환산해 씁니다.

```{admonition} rate 함수의 진짜 인자는 다섯 개
:class: warning

엔진은 모든 rate 함수를 **5-인자** `(sex, issue_age, duration, issue_class,
elapsed)` 로 부릅니다 (issue_class=직업/언더라이팅 등급, elapsed=상태 경과).
위처럼 3-인자 `(sex, issue_age, duration)` 로 써도 자동으로 5-인자로 감싸여
잘 돕니다. **함정**: 배수 같은 상수를 **네 번째 기본인자** 로 끼워 넣지 마세요 —
`lambda s, a, d, f=0.9: ...` 는 4-인자 rate 로 읽혀 엔진이 `issue_class` 를
`f` 자리에 밀어넣어 **조용히 덮어씁니다** (에러 없이 틀린 율). 상수는 클로저로
잡으세요 (`def shock(f): return lambda s, a, d: base(s, a, d) * f`).
```

같은 `death_fn`을 `Basis`의 **두 자리에 함께** 넘긴 것이 눈에 띌
겁니다. 그 둘은 다른 양입니다.

- `mortality_annual` — **보유계약 감소율**. 사람이 죽으면 더 이상 보장이
  진행되지 않으니, 4장의 보유계약 감쇠 식에 들어가는 사망률이죠.
- `coverages`의 `CoverageRate("DEATH", death_fn)` — **사망보험금이 발생하는
  율**. 그 달의 기대 사망보험금을 구하는 데 쓰이는 사망률입니다.

엔진이 갈라 다루는 두 양이지만, 손계산 예제처럼 같은 사망률을 쓰는
보통의 경우에는 한 변수를 두 자리에 같이 넘기면 됩니다. 이렇게 적으면
한 자리만 바꾸다 두 값이 어긋날 일도 없죠.

나머지 가정들을 한 줄씩 보면:

- `lapse_annual` — 연 해지율. 같은 방식으로 늘 0(해지 없음)을
  돌려줍니다.
- `discount_annual` — 연 할인율. `**`는 거듭제곱이라
  `1.005 ** 12 - 1`은 월 0.5%를 연 단위로 환산한 값입니다. 엔진은
  연율을 받아 월율로 바꿔 씁니다.
- `ra_confidence` — 위험조정 신뢰수준 75%(6장).
- `mortality_cv` — 사망위험 변동계수 0.10(6장).
- `coverages` — 위 두 번째 자리. 보험금이 어떤 율로 발생하는지를 보장
  코드별로 등록합니다. 여기서는 사망 한 갈래뿐이라 `"DEATH"` 한 줄입니다.
  사망 외에 다른 보장(진단·입원 ...)이 있다면 여기에 한 줄씩 더 등록
  합니다(11장에서 자세히).

사망률 함수 속 `np.full(issue_age.shape, ...)`은 "`issue_age`와 같은
모양의 배열을 만들어 같은 값으로 채워라"는 뜻입니다. 엔진이 사망률
함수를 부를 때 성별·나이·경과를 배열로 통째로 넘기므로, 함수도 같은
모양의 배열을 돌려줘야 합니다.

모델포인트는 `ModelPoints`로 만듭니다. 계약 한 건이면 `single()`이
편합니다.

```python
# 모델 포인트 -- 5~7장에서 손으로 따라온 그 한 계약
model_points = fcf.ModelPoints.single(
    issue_age   = 40,           # 가입연령 40세
    sex         = 0,            # 성별 (0=남, 1=여)
    benefits    = {0: 12_000},  # 또는 {"DEATH": 12_000} — 0번 보장(= DEATH) 보험금 12,000
    premium     = 100,          # 월납 보험료 100
    term_months = 2,            # 보험기간 2개월
)
```

가입연령 40세, 사망보험금 12,000, 월 보험료 100, 보험기간 2개월 —
5~7장의 그 계약입니다. `12_000`의 밑줄은 자릿수를 읽기 쉽게 나눈
것일 뿐, `12000`과 같습니다. `benefits={0: 12_000}`은 위에서 등록한
첫 번째(= 0번) 보장 — 곧 `"DEATH"` — 에 12,000을 매단다는 뜻입니다.
0번이라는 **인덱스 대신 담보 코드를 그대로 키**로 써도 됩니다 —
`benefits={"DEATH": 12_000}`. 등록 순서를 외울 필요가 없어 더 또렷하고,
담보가 여럿일 때 안전합니다.

## 8.3 측정 실행과 결과 읽기

입력이 준비됐으면 측정은 한 줄입니다.

```python
m = fcf.gmm.measure(model_points, basis)   # 일반모형(GMM) 으로 측정
```

`measure()`에 모델포인트와 가정을 넘기면, 1.2절의 4단계 — 추정, 할인,
위험조정, 이익 분리 — 를 모두 수행하고 결과를 `m`에 담아 줍니다.
`m`은 여러 값을 품은 결과 개체입니다. 거기서 원하는 값을 점(`.`)으로
꺼냅니다.

```python
print(m.bel[0])             # BEL
print(m.ra[0])              # RA
print(m.csm[0])             # CSM
print(m.loss_component[0])  # 손실요소
```

`print()`는 값을 화면에 보여 주는 명령입니다. 실행하면 이렇게
나옵니다.

```
39.10819409369891
16.026932498439958
0.0
55.135126592138874
```

`m.bel`은 모델포인트별 BEL을 담은 배열입니다 — 각 계약의 **최초 인식
시점** 값. `[0]`은 그 배열에서 첫 번째 모델포인트를 골라낸 것이고, 계약이
하나뿐이니 `[0]`이면 됩니다. 시점별 BEL 궤적(5.3절의 BEL 곡선)이 필요하면
`m.bel_path`를 씁니다.

이 숫자들을 알아보시겠습니까? 5장에서 손으로 구한 BEL 39.10,
6장에서 구한 RA 16.01입니다. 손계산은 할인계수를 반올림해 썼으니
끝자리만 미세하게 다를 뿐, 엔진이 내놓은 39.11과 16.03은 같은
값입니다. FCF = BEL + RA가 양수라 7.1절에서 본 대로 **손실부담계약** —
CSM은 0, 손실요소는 55.14입니다.

지금까지의 코드를 한 파일에 모으면 이게 전부입니다.

```python
import numpy as np
import fastcashflow as fcf

# 사망률 함수 -- 월 사망률 1% 의 연 환산 (모든 sex/age/duration 에 동일)
death_fn = lambda sex, issue_age, duration: np.full(
    issue_age.shape, 1 - (1 - 0.01) ** 12,
)

# 해지율 함수 -- 해지 없음
lapse_fn = lambda sex, issue_age, duration: np.full(duration.shape, 0.0)

# 산출기초
basis = fcf.Basis(
    mortality_annual = death_fn,         # 보유계약 사망률 (위 death_fn)
    lapse_annual     = lapse_fn,         # 해지율 (해지 없음)
    discount_annual  = 1.005 ** 12 - 1,  # 연 할인율 (월 0.5% 의 연 환산)
    ra_confidence    = 0.75,             # 위험조정 신뢰수준 75%
    mortality_cv     = 0.10,             # 사망률 변동계수 10%
    coverages        = (
        fcf.CoverageRate("DEATH", death_fn),  # 사망 보장 1종 (청구 rate = death_fn)
    ),
)

# 모델 포인트 -- 5~7장에서 손으로 따라온 그 한 계약
model_points = fcf.ModelPoints.single(
    issue_age   = 40,           # 가입연령 40세
    sex         = 0,            # 성별 (0=남, 1=여)
    benefits    = {0: 12_000},  # 또는 {"DEATH": 12_000} — 0번 보장(= DEATH) 보험금 12,000
    premium     = 100,          # 월납 보험료 100
    term_months = 2,            # 보험기간 2개월
)

# 측정
m = fcf.gmm.measure(model_points, basis)
print(m.bel[0])             # BEL
print(m.ra[0])              # RA
print(m.csm[0])             # CSM
print(m.loss_component[0])  # 손실요소
```

```
39.10819409369891
16.026932498439958
0.0
55.135126592138874
```

계약이 여럿이면 `print(m)` 한 줄이 모델포인트별 BEL·RA·CSM·손실요소와
합계를 표로 보여 줍니다 (`m.bel.sum()` 으로 합계만).

일곱 장에 걸쳐 손으로 따라온 측정을, 엔진은 이 스무 줄 남짓으로
똑같이 해냅니다.

## 8.4 대규모 평가

계약 한 건이 아니라 여러 건이라면 어떨까요? `measure()`는 시점별
궤적까지 다 담아 무겁습니다. 대규모에는 **`measure(full=False)`**를 씁니다 —
모델포인트마다 BEL·RA·CSM·손실요소 네 숫자만 돌려주는 빠른 경로입니다.

직접 해 보려면 입력이 필요한데, 8.2절처럼 코드로 짓는 대신
fastcashflow에 들어 있는 **샘플 데이터**를 쓰면 됩니다.

```python
# 샘플 portfolio 로드 (정기보험 / 건강보험 / 종신보험 11 건)
model_points = fcf.samples.model_points()  # ModelPoints 개체
basis        = fcf.samples.basis()         # {(product, channel): Basis}

# 세그먼트별 자동 라우팅으로 측정 -- 각 계약을 자기 (상품, 채널) 가정에 맞춤
val = fcf.gmm.measure(model_points, basis, full=False)

print(val.bel)      # 모델포인트별 BEL 배열 (길이 11)
print(val.csm)      # 모델포인트별 CSM 배열 (길이 11)
```

`samples.model_points()`는 패키지에 든 작은 포트폴리오(계약 11건,
정기보험·건강보험·종신보험)를, `samples.basis()`는 그에 맞는
가정을 `{(product, channel): Basis}` 딕셔너리로
돌려줍니다. `measure()`는 각 계약을 자기 (상품, 채널) 세그먼트의
가정에 맞춰 자동 라우팅해 한 번에 평가합니다. 계약마다 주계약에 더해
진단·입원·재해사망·연금·생존 같은 특약이 붙어 있죠. 결과는 모델포인트
순서대로 늘어선 배열이라, 11건이면 길이 11입니다.

세그먼트가 하나뿐인 동질 포트폴리오라면 `measure(mp, basis, full=False)`로 단일 가정을 그대로
넘기면 됩니다. 샘플은 세 상품 × 여러 채널이 섞여 있어 라우팅이 필요합니다.

샘플은 이익계약과 손실부담계약이 섞여 있어, 11건 가운데 7건이 이익이
예상(CSM > 0)되고 4건이 손실부담계약(BEL > 0, CSM = 0, 손실요소 > 0)입니다
— 7장에서 본 두 분기점이 모두 나타나는 셈입니다.

속도는 이 패키지의 장점입니다. 100만 계약을 120개월 평가하는 데 약
0.05초, 500만 건이면 약 0.3초입니다.

## 8.5 다음 장

지금까지 일반모형으로 — 계약을 코드로 만들거나 샘플 데이터로 불러와 —
엔진을 직접 돌려 봤습니다. 9장과 10장에서는 나머지 두 회계모형,
보험료배분접근법(PAA)과 변동수수료접근법(VFA)을 봅니다. 1.5절에서
개념만 짚었던 두 모형을, 이번에는 측정까지 따라갑니다.

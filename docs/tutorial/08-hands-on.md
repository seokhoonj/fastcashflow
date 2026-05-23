# 8장. 직접 실행하기

```{admonition} 이 장에서 배우는 것
:class: tip

- fastcashflow 설치하기
- 코드를 어디에 쓰고 어떻게 실행하는가
- 모델포인트와 계리적 가정을 코드로 만들기
- measure()로 측정하고 BEL·RA·CSM 읽기
- value()와 샘플 데이터로 대규모 평가하기
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
pip install "git+https://github.com/seokhoonj/fastcashflow.git#egg=fastcashflow[viz]"
```

`[viz]`는 결과를 그리는 내장 차트에 필요한 matplotlib까지 함께
설치합니다. 차트가 필요 없다면 `[viz]` 부분을 빼고
`pip install git+https://github.com/seokhoonj/fastcashflow.git` 로 설치합니다.
어느 쪽이든 numpy·numba·polars 같은 핵심 의존 패키지는 자동으로 들어옵니다.

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

엔진의 입력은 둘뿐입니다(1.6절) — **계리적 가정**과 **모델포인트**.

먼저 두 줄로 필요한 도구를 불러옵니다.

```python
import numpy as np
import fastcashflow as fcf
```

`import`는 다른 라이브러리를 불러오는 명령입니다. `numpy`는 숫자
배열을 다루는 라이브러리이고, `as np`는 앞으로 짧게 `np`라 부르겠다는
뜻입니다. fastcashflow는 `fcf`로 부릅니다.

계리적 가정은 `Assumptions`로 만듭니다. 5~7장 예제의 가정을 그대로
옮깁니다.

```python
assumptions = fcf.Assumptions(
    mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, 0.01),
    lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, 0.0),
    discount_annual=1.005 ** 12 - 1,
    expense_acquisition=0.0,
    expense_maintenance_annual=0.0,
    expense_inflation=0.0,
    ra_confidence=0.75,
    mortality_cv=0.10,
)
```

괄호 안은 `이름=값` 꼴로 가정을 하나씩 적은 것입니다. 한 줄씩 보면:

- `mortality_annual` — 연 사망률. 숫자 하나가 아니라 **함수**로
  줍니다. 사망률이 성별·나이·경과에 따라 달라질 수 있기 때문이죠(3.2절).
  `lambda`는 간단한 함수를 한 줄로 적는 파이썬 문법인데, 여기서는
  성별·나이·경과와 상관없이 늘 0.01(연 1%)을 돌려주는 함수입니다.
  엔진이 내부에서 constant-force 방식으로 월율로 환산합니다.
- `lapse_annual` — 연 해지율. 같은 방식으로 늘 0(해지 없음)을
  돌려줍니다.
- `discount_annual` — 연 할인율. `**`는 거듭제곱이라
  `1.005 ** 12 - 1`은 월 0.5%를 연 단위로 환산한 값입니다. 엔진은
  연율을 받아 월율로 바꿔 씁니다.
- `expense_acquisition` 등 — 사업비. 이 예제에서는 모두 0입니다.
- `ra_confidence` — 위험조정 신뢰수준 75%(6장).
- `mortality_cv` — 사망위험 변동계수 0.10(6장).

사망률 함수 속 `np.full(issue_age.shape, 0.01)`은 "`issue_age`와 같은
모양의 배열을 만들어 전부 0.01로 채워라"는 뜻입니다. 엔진이 사망률
함수를 부를 때 성별·나이·경과를 배열로 통째로 넘기므로, 함수도 같은
모양의 배열을 돌려줘야 합니다.

모델포인트는 `ModelPoints`로 만듭니다. 계약 한 건이면 `single()`이
편합니다.

```python
model_points = fcf.ModelPoints.single(
    issue_age=40, death_benefit=12_000,
    level_premium=100, term_months=2,
)
```

가입연령 40세, 사망보험금 12,000, 월 보험료 100, 보험기간 2개월 —
5~7장의 그 계약입니다. `12_000`의 밑줄은 자릿수를 읽기 쉽게 나눈
것일 뿐, `12000`과 같습니다.

## 8.3 측정 실행과 결과 읽기

입력이 준비됐으면 측정은 한 줄입니다.

```python
m = fcf.measure(model_points, assumptions)
```

`measure()`에 모델포인트와 가정을 넘기면, 1.2절의 4단계 — 추정, 할인,
위험조정, 이익 분리 — 를 모두 수행하고 결과를 `m`에 담아 줍니다.
`m`은 여러 값을 품은 결과 객체입니다. 거기서 원하는 값을 점(`.`)으로
꺼냅니다.

```python
print(m.bel[0, 0])           # BEL
print(m.ra[0, 0])            # RA
print(m.csm[0, 0])           # CSM
print(m.loss_component[0])   # 손실요소
```

`print()`는 값을 화면에 보여 주는 명령입니다. 실행하면 이렇게
나옵니다.

```
39.1082
16.0269
0.0
55.1351
```

`m.bel`은 한 숫자가 아니라 시점별 BEL을 담은 표입니다(5.3절의 BEL
곡선). `[0, 0]`은 그 표에서 **첫 번째 모델포인트의 최초 인식 시점**
값을 골라낸 것입니다. 계약이 하나뿐이니 `[0, 0]`이면 됩니다.

이 숫자들을 알아보시겠습니까? 5장에서 손으로 구한 BEL 39.10,
6장에서 구한 RA 16.01입니다. 손계산은 할인계수를 반올림해 썼으니
끝자리만 미세하게 다를 뿐, 엔진이 내놓은 39.11과 16.03은 같은
값입니다. FCF = BEL + RA가 양수라 7.1절에서 본 대로 **손실부담계약** —
CSM은 0, 손실요소는 55.14입니다.

지금까지의 코드를 한 파일에 모으면 이게 전부입니다.

```python
import numpy as np
import fastcashflow as fcf

assumptions = fcf.Assumptions(
    mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, 0.01),
    lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, 0.0),
    discount_annual=1.005 ** 12 - 1,
    expense_acquisition=0.0,
    expense_maintenance_annual=0.0,
    expense_inflation=0.0,
    ra_confidence=0.75,
    mortality_cv=0.10,
)
model_points = fcf.ModelPoints.single(
    issue_age=40, death_benefit=12_000,
    level_premium=100, term_months=2,
)
m = fcf.measure(model_points, assumptions)
print(m.bel[0, 0], m.ra[0, 0], m.csm[0, 0], m.loss_component[0])
```

일곱 장에 걸쳐 손으로 따라온 측정을, 엔진은 이 스무 줄로 똑같이
해냅니다.

## 8.4 대규모 평가

계약 한 건이 아니라 여러 건이라면 어떨까요? `measure()`는 시점별
궤적까지 다 담아 무겁습니다. 대규모에는 **`value()`**를 씁니다 —
모델포인트마다 BEL·RA·CSM·손실요소 네 숫자만 돌려주는 빠른 경로입니다.

직접 해 보려면 입력이 필요한데, 8.2절처럼 코드로 짓는 대신
fastcashflow에 들어 있는 **샘플 데이터**를 쓰면 됩니다.

```python
model_points = fcf.load_sample_model_points()
basis        = fcf.load_sample_assumptions()       # {(product, channel): Assumptions}
assumptions  = basis[("term_a", "GA")]             # 한 세그먼트 선택
val          = fcf.value(model_points, assumptions)

print(val.bel)
print(val.csm)
```

`load_sample_model_points()`는 패키지에 든 작은 포트폴리오(계약 8건,
정기보험과 건강보험)를, `load_sample_assumptions()`는 그에 맞는 가정을
`{(product, channel): Assumptions}` 딕셔너리로 돌려줍니다. 샘플은
`term_a` 상품의 GA / FC 두 세그먼트를 담고 있어, 한 줄로 한 세그먼트를
골라 `value()`에 넘깁니다. 계약마다 주계약 사망에 더해 진단·입원·재해사망·
연금·생존 같은 특약이 붙어 있습니다. 파일을 따로 준비할 필요가 없죠.
`value()`의 결과는 모델포인트 순서대로 늘어선 배열이라, 8건이면 길이 8입니다.

샘플 8건은 모두 보험료가 보장에 견주어 넉넉히 매겨진 계약이라, BEL이
음수(이익이 예상됨)이고 CSM이 0보다 크며 손실요소는 0입니다.

속도는 이 패키지의 장점입니다. 100만 계약을 120개월 평가하는 데 약
0.05초, 500만 건이면 약 0.3초입니다.

## 8.5 다음 장

지금까지 일반모형으로 — 계약을 코드로 만들거나 샘플 데이터로 불러와 —
엔진을 직접 돌려 봤습니다. 9장과 10장에서는 나머지 두 회계모형,
보험료배분접근법(PAA)과 변동수수료접근법(VFA)을 봅니다. 1.5절에서
개념만 짚었던 두 모형을, 이번에는 측정까지 따라갑니다.

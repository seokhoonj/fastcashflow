# 5.4 유니버설 (적립 방식 — 계좌형 사망보장)

```{admonition} 이 챕터에서 배우는 것
:class: tip

- **유니버설은 담보도, 측정모델도 아닌 세 번째 축 — 적립 방식 (funding)** 이다.
  상품에 얹히는 "방식" 이라 유니버설종신 / 유니버설연금 / 변액유니버설처럼
  아무 상품에나 붙는다
- 유니버설 계약은 **계좌-백 사망담보** 하나로 선언한다 —
  `CoverageRate("DEATH", coi_annual, funds_from_account=True, pays_account_balance=True)`
- **고정 / 금리연동형 = `gmm.measure`, 변액 = `vfa.measure`** — 계좌가 굴러가는
  방식은 똑같고 할인율만 다르다 (locked-in 곡선 vs 기초자산 수익률)
- 계좌가 매월 굴러가는 메커닉 — 보험료 (적립) -> COI (순보장액 NAR 에 부과) ->
  공시이율 크레딧 — 과 사망보험금 `max(계좌가치, face)` 를 `gmm.trace` 로 검산
```

앞 챕터들 (5.1~5.3) 의 변액은 **측정모델** 이야기였습니다 — 계좌가 기초자산
수익률로 굴러가고 (VFA), 최저보증의 가치를 봤습니다. 유니버설은 다른 축입니다.

## 유니버설 = 적립 방식 (세 번째 축)

상품을 세 개의 **직교하는 축** 으로 보면 유니버설의 자리가 분명해집니다:

```{list-table}
:header-rows: 1
:widths: 26 34 40

* - 축
  - 무엇을 답하나
  - 엔진 표현
* - **담보 (benefit)**
  - 무엇을 보장하나
  - `CalculationMethod` (DEATH / DIAGNOSIS / ANNUITY ...)
* - **적립 방식 (funding)**
  - 어떻게 자금을 굴리고 비용을 매기나
  - **유니버설 = 계좌 섀시 on**
* - **측정모델 (measurement)**
  - 어떻게 할인 / 측정하나
  - 어느 `.measure` 진입점 (`gmm` / `vfa`)
```

"유니버설" 은 **보험료가 적립금으로 쌓이고, 위험보험료 (COI) 가 거기서 차감되고,
공시이율이 붙는** 적립 방식입니다. 무엇을 보장하느냐 (담보) 도, 어떻게 할인하느냐
(측정모델) 도 아닙니다. 그래서 형용사처럼 아무 상품에나 얹힙니다 —
유니버설**종신**, 유니버설**연금**, **변액**유니버설**종신**. `변액` 도 같은 부류의
적립-방식 수식어 (투자형 계좌 -> VFA) 라, `변액유니버설` 은 두 방식이 겹친 것입니다.

## 계좌가 매월 굴러가는 메커닉

유니버설 계좌 (account value, AV) 는 매월 정해진 순서로 굴러갑니다:

```text
AV[t]
  + 보험료 (load 차감)        납입보험료 x (1 - premium_load)
  - 위험보험료 (COI)          coi_rate x NAR,  NAR = max(0, face - AV)
  - 유지비
  x (1 + 공시이율)            max(투자수익률, 최저보증이율) 로 크레딧
= AV[t+1]
```

사망보험금은 `max(계좌가치, face)` 입니다 — 계좌가 돌려지고, 부족하면 face 까지
얹어줍니다. 보험사의 진짜 위험은 그 **순보장액** `NAR = max(0, face - AV)` 뿐이고,
COI 는 바로 그 NAR 에 부과됩니다 (최저적립이율 보증은 [5.3](crediting-floor) 참고).

## 모델링 매핑

```{list-table}
:header-rows: 1
:widths: 42 58

* - 자리
  - 무엇
* - `ModelPoints.account_value`
  - 가입 시점 계좌가치 (AV[0])
* - `ModelPoints.minimum_death_benefit`
  - 사망보장 face — 사망보험금 = `max(계좌가치, face)`
* - `ModelPoints.minimum_crediting_rate`
  - 최저적립이율 보증 (`max(수익률, 보증이율)`; [5.3](crediting-floor))
* - `Basis.coi_annual`
  - 위험보험료율 — 계좌가 NAR 에 부과하는 COI. 보장원가이지 best-estimate
    탈퇴율이 아니며, 그 스프레드가 사망 마진
* - `Basis.premium_load`
  - 계좌 적립 전 떼는 부가보험료율 (`prem_to_av = premium x (1 - load)`)
* - `Basis.investment_return`
  - 계좌 크레딧 기준 공시이율 (최저보증으로 floor)
* - `CoverageRate("DEATH", coi_annual, funds_from_account=True, pays_account_balance=True)`
  - 계좌-백 사망담보 — `funds_from_account` 는 COI 를 계좌서 차감,
    `pays_account_balance` 는 급부가 계좌잔액 (`max(av, face)`) 을 읽음
```

## 측정 — 금리연동형은 `gmm`, 변액은 `vfa`

계좌가 굴러가는 방식 (generation) 은 둘이 **똑같습니다.** 다른 건 **할인율** 뿐:

- **금리연동형 / 고정 (간접참가) -> `gmm.measure`** — locked-in 할인곡선 (Sec. 36)
- **변액 (직접참가) -> `vfa.measure`** — 기초자산 수익률 (`investment_return`)

"어느 모델이냐" 가 곧 "어느 `.measure` 를 부르느냐" 입니다 — 다른 모든 상품과
똑같습니다. 별도 `measure_ul` 도, `measurement_model=` 플래그도 없습니다.

```python
import fastcashflow as fcf

mp    = fcf.samples.model_points("ul")     # 계좌-백 DEATH 담보를 든 합성 포트폴리오
basis = fcf.samples.basis("ul")            # coi_annual / premium_load / investment_return

fixed    = fcf.gmm.measure(mp, basis)      # 금리연동형 UL -- locked-in 할인
variable = fcf.vfa.measure(mp, basis)      # 변액 UL       -- 기초자산 수익률 할인

print(f"고정(GMM) CSM[0] = {fixed.csm[0]:,.0f}")
print(f"변액(VFA) CSM[0] = {variable.csm[0]:,.0f}")
```

출력:

```
고정(GMM) CSM[0] = 875,109
변액(VFA) CSM[0] = 2,649,946
```

계좌가 굴러가는 방식 (generation) 은 둘이 똑같고, 할인 기준만 달라 BEL / CSM 이
달라집니다 (위 합성 포트폴리오에선 변액의 4% 수익률 할인이 더 큰 CSM 을 냅니다).

BEL 은 미래 급부·비용의 현가에서 미래보험료의 현가 **와 보유 계좌가치 (fund)** 를
뺀 값입니다 (`PV(급부+비용) - PV(보험료) - fund`). RA 는 순보장액 (NAR) 에 실린
사망위험 — 계좌 위 face 초과분, 유일한 보험위험 — 에 비용위험을 더해 매깁니다.
CSM 은 `max(0, -(BEL + RA))`.

```{note}
유니버설을 직접 만들 땐 `Basis.coverages` 에 계좌-백 DEATH 담보를 등록하고
`ModelPoints` 에 `account_value` / `minimum_death_benefit` /
`minimum_crediting_rate` 를 채웁니다. face 는 `minimum_death_benefit` 에서 오고,
담보의 `coverage_amount` 는 사망급부에 쓰이지 않습니다 (계좌잔액을 읽으므로).
```

## 검산 — `gmm.trace`

`gmm.trace` 가 측정 과정을 ASCII 트리로 풀어 보여줍니다. 그 트리의 **계좌 섹션**
은 이렇게 나옵니다 (key month, 우측 `death` / `fund` 열은 지면상 생략):

```text
+- Universal-life account (key months)
|   +- account_value0 (av0)  =            0.00
|   +- face (min_death_ben)  =  100,000,000.00
|   +- premium_load          =            0.06  (prem_to_av = premium * (1 - load))
|   +- investment_return     =            0.04  (account crediting basis)
|   +- death = max(av_mid, face);  NAR = max(0, face - av_mid);  COI = coi_m * NAR
|   +- t=   0m: av=           0.00  av_mid=     449,975.53  coi=   20,759.21  nar=  99,550,024.47  ...
|   +- t=  60m: av=  30,018,045.96  av_mid=  30,523,388.16  coi=   14,498.28  nar=  69,476,611.84  ...
|   `- t= 120m: av=  66,999,455.42       (boundary)
```

```python
fcf.gmm.trace(0, mp, basis)        # 위 계좌 섹션을 포함한 전체 트리를 출력
```

계좌가 보험료로 쌓이며 (av 증가) NAR 이 줄고, 그에 따라 COI 도 줄어드는 게
보입니다 — 사망보험금은 계좌가 face 를 넘기 전까지 `max(av_mid, face) = face`.

## 한국 상품과의 매핑

유니버설 = 적립 방식이므로 담보 / 측정모델과 자유롭게 조합됩니다:

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - 상품
  - 적립 (공시 구분)
  - 측정 진입점
* - 유니버설종신
  - 금리연동형
  - `gmm.measure`
* - 변액유니버설종신
  - 변액
  - `vfa.measure`
* - 저축형 (변액)유니버설
  - 변액
  - `vfa.measure` (face 작음 -> NAR 작음)
```

```{admonition} v1 범위
:class: note

현재는 **사망 레그** (`max(계좌, face)` + NAR-COI) 를 다룹니다. 계좌에서 비용을
빼가는 특약 (예: 유니버설 재진단암) 과 계좌를 연금화하는 연금 레그는 데이터모델이
자리를 비워뒀지만 (담보 상호작용 플래그) 아직 커널 구현 전입니다.
```

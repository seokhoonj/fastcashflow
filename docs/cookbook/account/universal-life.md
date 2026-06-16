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

## 연금화 — 적립에서 종신연금으로 전환 (2단계)

유니버설**연금**은 두 단계로 굴러갑니다. **1단계 (적립)** 는 위 계좌 메커닉
그대로 — 보험료가 쌓이고 COI 가 차감되고 공시이율이 붙습니다.
`annuitization_months` 에 이르면 **2단계 (지급)** 로 전환됩니다: 그 시점의
계좌잔액을 보증최저적립금 (GMAB) 으로 floor 한 뒤, 잠근 보증연금환산율
(GAO rate — guaranteed annuity option, 전환 시 한 번 고정되는 환산율) 을 곱해
**종신 보증연금** 을 삽니다 — 전환 잔액은
`converted_balance = max(계좌잔액, GMAB)`, 보증연금액은
`locked_annuity_payment = converted_balance x annuitization_rate` 입니다.

전환 후에는 보험료 / COI / 해지가 없고 만기 환급금도 없습니다 (잔액을 이미
연금으로 바꿨으니 만기금까지 주면 이중지급). 지급 단계의 in-force 는 **사망으로만**
줄어듭니다 (지급 중인 종신연금은 해지할 수 없으니까). 그래서 지급기의
위험조정 (RA) 은 사망률이 아니라 **장수리스크 (longevity — 연금수급자가 예상보다
오래 살수록 보험사가 손해)** 로 매깁니다 (`Basis.longevity_cv`).

```{list-table}
:header-rows: 1
:widths: 42 58

* - 자리
  - 무엇
* - `ModelPoints.annuitization_months`
  - 전환 시점 (개월). 0 = 전환 없음 (일반 계좌 -> 만기금). `<= term_months`
* - `ModelPoints.annuitization_rate`
  - 보증연금환산율 (GAO) — `locked_annuity_payment = converted_balance x rate`,
    전환 시 한 번 잠금
* - `ModelPoints.minimum_accumulation_benefit`
  - 보증최저적립금 (GMAB) — 전환 잔액의 하한
* - `ModelPoints.annuity_frequency_months`
  - 연금 지급 주기 (기본 1 = 매월)
```

전환 규칙 두 가지: 보험료 납입은 전환월까지 끝나야 하고
(`premium_term_months <= annuitization_months`), 전환하는 계약은 만기 환급금이
0 이어야 합니다.

번들 샘플 `"ul-annuity"` 는 두 계약입니다 — 0 번은 15년 적립 후 전환 (월납),
1 번은 일시납 후 10년차 전환. 둘 다 `gmm.measure` 로 측정합니다 (금리연동형):

```python
ann_mp    = fcf.samples.model_points("ul-annuity")
ann_basis = fcf.samples.basis("ul-annuity")
m = fcf.gmm.measure(ann_mp, ann_basis)

print(f"contract 0 (적립15년): BEL={m.bel[0]:,.0f}  RA={m.ra[0]:,.0f}  CSM={m.csm[0]:,.0f}")
print(f"contract 1 (일시납):   BEL={m.bel[1]:,.0f}  RA={m.ra[1]:,.0f}  CSM={m.csm[1]:,.0f}")
```

출력:

```
contract 0 (적립15년): BEL=-14,870,126  RA=2,047,038  CSM=12,823,088
contract 1 (일시납):   BEL=-6,855,295  RA=1,476,207  CSM=5,379,088
```

`gmm.trace` 의 **연금화 섹션** 이 전환 산식을 그대로 풀어 보여줍니다 — 전환월의
계좌잔액, GMAB floor, 잠긴 연금액까지:

```python
fcf.gmm.trace(0, ann_mp, ann_basis)        # 전환 + 지급 섹션을 포함한 전체 트리
```

출력 (연금화 섹션 발췌):

```text
+- Universal-life annuitization (conversion + payout)
|   +- annuitization_months  =             180  (account stops, converts to income)
|   +- balance at conversion =   88,714,736.96  (av[180], no month-180 credit)
|   +- GMAB floor            =   40,000,000.00  (minimum_accumulation_benefit)
|   +- converted_balance     =   88,714,736.96  (= max(balance, GMAB))
|   +- annuitization_rate    =           0.004  (GAO rate, locked once)
|   +- locked_annuity_payment=      354,858.95  (= converted_balance x rate)
|   +- phase 2: annuity-due on surviving in-force; no premium / COI / surrender, no maturity lump
```

15년 적립으로 계좌가 88.7M 까지 쌓였고 (GMAB 40M 은 안 뭅니다), 거기에 0.004 (월)
의 GAO 를 곱한 354,858 이 매월 종신연금으로 잠깁니다. 적립단계에 받은 보험료의
현가가 지급단계에 내줄 연금의 현가보다 커서 BEL 이 음수 (이익) 가 되고, 그만큼이
CSM 으로 인식됩니다.

## 실적배당형 변액연금 payout (예정이율 기준 재부유)

위 GAO 연금은 전환 후 **고정**입니다. 실적배당형 변액연금은 지급액이 **펀드 실적
으로 매월 재부유**합니다 — 전환 시 적립금으로 "연금 unit" 을 사고, 매월 unit 가치가
실제수익률 대비 **예정이율 (AIR, assumed interest rate)** 만큼 오르내립니다:

```text
payment_k = locked_annuity_payment x ((1 + 실제수익률) / (1 + 예정이율))^k,   k = 경과월
```

실적이 예정이율보다 좋으면 연금이 늘고, 나쁘면 줍니다. 계약별로 `annuity_air_annual`
(예정이율) 을 주면 그 계약이 실적배당, NaN (기본) 이면 고정 GAO 입니다.

**변액 payout 은 `vfa.measure` 전용입니다.** VFA 는 할인율 = 펀드수익률이라, 지급액의
펀드 재부유와 할인의 펀드가 **상쇄**되어 BEL 이 "예정이율짜리 정기연금 (AIR-적립금)"
으로 떨어집니다 — 투자위험은 계약자가 전부 지고 보험사엔 장수위험만 남습니다. GMM
(locked-in 할인) 에선 상쇄가 안 돼 무의미하므로 `gmm.measure` 는 변액 payout 책을
거부합니다.

번들 샘플 `"ul-var-annuity"` 는 0 번 = 실적배당 (예정이율 2%), 1 번 = 고정 GAO 의
혼재 책입니다 (`vfa.measure` 가 둘 다 측정):

```python
va_mp    = fcf.samples.model_points("ul-var-annuity")
va_basis = fcf.samples.basis("ul-var-annuity")
m = fcf.vfa.measure(va_mp, va_basis)

print(f"contract 0 (실적배당@2%): BEL={m.bel[0]:,.0f}  RA={m.ra[0]:,.0f}  CSM={m.csm[0]:,.0f}")
print(f"contract 1 (고정 GAO):    BEL={m.bel[1]:,.0f}  RA={m.ra[1]:,.0f}  CSM={m.csm[1]:,.0f}")
```

출력:

```
contract 0 (실적배당@2%): BEL=-14,121,491  RA=2,028,567  CSM=12,092,924
contract 1 (고정 GAO):    BEL=-8,152,879  RA=1,361,061  CSM=6,791,818
```

```{admonition} 최저보증연금은 아직
:class: note

실적이 나빠도 깔아주는 **최저지급보증** (payment floor) 은 금융보증이라 그 가치는
확률적 시나리오 (TVOG) 가 필요합니다 — GMDB/GMAB 의 시간가치를 TVOG 로 분리한 것과
같은 경계로, 아직 범위 밖입니다. v1 은 순수 실적배당입니다.
```

## 계좌차감 특약 (계좌서 비용을 빼가는 특약)

유니버설 계좌는 사망 레그의 COI 만 빼가는 게 아닙니다. **계좌차감 특약** (예:
유니버설 재진단암) 은 매월 위험보험료를 **계좌에서 차감**하되, 급부는 계좌잔액이
아니라 **고정 진단금**입니다. 사망 레그와의 차이는 담보 상호작용 플래그 하나뿐:

```{list-table}
:header-rows: 1
:widths: 30 22 22 26

* - 담보
  - `funds_from_account`
  - `pays_account_balance`
  - 차감 / 급부
* - 사망 레그
  - `True`
  - `True`
  - NAR-COI `coi x (face - av)` 차감 / `max(av, face)` 지급
* - 계좌차감 특약
  - `True`
  - `False`
  - 고정 `rate x amount` 차감 / 고정 진단금 (계좌 무관)
```

그래서 계좌는 매월 **두 가지** 비용을 빼갑니다 — 사망 레그의 NAR-COI 와 모든
계좌차감 특약의 고정 charge (`rate x amount`). 특약의 **급부**는 보통 특약처럼
claim 쪽에서 지급됩니다 (재진단암 -> 반복지급 `MORBIDITY`); 옮겨진 건 **비용**
뿐입니다 — 별도 보험료가 아니라 계좌에서 차감. 그 charge 만큼 계좌가 덜 쌓여
환급 / 사망 / 만기 급부가 줄고, 그게 보험사가 특약 원가를 회수하는 경로입니다.
특약 진단금은 morbidity 위험을 지므로 계좌형 RA 에 `morbidity_cv` 항으로 가격됩니다.

번들 샘플 `"ul-cost-deduct"` 는 사망 레그 + 재진단암 계좌차감 특약을 든 두 계약입니다:

```python
cd_mp    = fcf.samples.model_points("ul-cost-deduct")
cd_basis = fcf.samples.basis("ul-cost-deduct")
m = fcf.gmm.measure(cd_mp, cd_basis)

print(f"contract 0: BEL={m.bel[0]:,.0f}  RA={m.ra[0]:,.0f}  CSM={m.csm[0]:,.0f}")
print(f"contract 1: BEL={m.bel[1]:,.0f}  RA={m.ra[1]:,.0f}  CSM={m.csm[1]:,.0f}")
```

출력:

```
contract 0: BEL=-2,466,380  RA=145,797  CSM=2,320,583
contract 1: BEL=-1,159,364  RA=78,412  CSM=1,080,952
```

특약 (`CANCER`) 은 `CoverageRate("CANCER", rate, funds_from_account=True,
pays_account_balance=False)` 로 선언하고, 진단금은 `benefits={"CANCER": ...}` +
`calculation_methods={"CANCER": CalculationMethod.MORBIDITY}` 로 줍니다.

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

현재는 **사망 레그** (`max(계좌, face)` + NAR-COI), **연금화 레그** (적립 ->
종신연금 2단계 전환), **실적배당형 변액연금 payout** (예정이율 기준 재부유),
**계좌차감 특약** (계좌서 비용 차감 + 고정 진단금, 위 절들) 을 다룹니다. 변액
payout 의 **최저지급보증** 과 보증 시간가치 (TVOG), 연금자 전용 위험률,
보증기간부 / 원금보증은 아직 범위 밖입니다.
```

# 8.6 요구자본 (Solvency II / K-ICS)

:::{admonition} 이 챕터에서 배우는 것
:class: tip

- 위험기반 **요구자본 (SCR)** 을 충격 -> 재측정 -> 상관집계로 산출하는 법
  (`fcf.solvency.required_capital(mp, basis, regime=...)`)
- **Solvency II 와 K-ICS 가 같은 엔진, 다른 calibration** 이라는 것 -- 호출은 그대로,
  `regime` 만 바꾸면 제도가 바뀜 (`fcf.solvency.SII` / `fcf.solvency.KICS`)
- 위험마진 두 방식 (SII cost-of-capital / K-ICS percentile) 과 보험위험 SCR 의
  하위위험 분해
- 부채측 SCR 을 가용자본과 묶어 **지급여력비율** 을 내는 법, 그리고 자산측이
  왜 사용자 입력인지 (`fcf.solvency.solvency_ratio`)
:::

위험기반 지급여력제도 (Solvency II, K-ICS) 는 요구자본을 **충격 후 재평가** 로 정합니다 --
best-estimate 가정 하나를 규정대로 충격하고, 부채를 다시 측정해 그 증가분 (순자산가치
감소) 을 그 하위위험의 자본으로, 그다음 하위위험들을 상관행렬로 합칩니다. 두 제도는 이
구조가 **똑같고** 충격 크기 / 상관 셀 / 위험마진 방식만 다릅니다 (K-ICS 가 Solvency II 의
틀을 빌렸습니다). 그래서 fastcashflow 는 **regime-agnostic 엔진** 하나에 제도별
`RegimeSpec` (calibration) 을 갈아 끼웁니다.

범위 (v1): 부채측 -- 생명·장기 보험위험 (사망 / 장수 / 장해·질병 / 해지 + 대량해지 /
사업비) 과 금리위험. 자산측 (주식·부동산·신용·운영), 가용자본, 시장위험 자산측은 자산
모델이 필요해 범위 밖입니다 (가짜 자산모델을 만들지 않습니다).

## 모델링 매핑 -- 하위위험은 Basis 충격 + 재측정

:::{list-table}
:header-rows: 1
:widths: 28 36 36

* - 하위위험
  - Solvency II 충격
  - K-ICS 충격
* - 사망
  - 사망률 +15%
  - +12.5%
* - 장수
  - 사망률 -20%
  - -17.5%
* - 장해·질병
  - +25% (간이)
  - 정액 +13% / 실손 +10%
* - 해지
  - 옵션 +/-50%, 대량해지 40%
  - 옵션 +/-35%, 대량해지 30%
* - 사업비
  - +10%, 인플레 +1%p
  - 같음
* - 금리
  - EIOPA 만기별 표
  - (사용자 공급 곡선)
* - 위험마진
  - cost-of-capital (6%)
  - percentile (x0.40)
:::

각 충격은 `Basis` / `ModelPoints` 를 바꿔 `gmm.measure` 를 다시 돌리고 `max(ΔBEL, 0)` 을
취합니다 (사망 충격은 in-force 감소율 **과** 사망보험금 청구율 둘 다 -- [8.1
민감도](sensitivity) 의 충격 관용구).

## 최소 작동 예제 -- Solvency II

10년 정기보험 한 건. 하위위험별 자본 -> 보험위험 SCR + 금리 -> 위험마진.

```python
import fastcashflow as fcf

mp = fcf.ModelPoints.single(40, 600_000.0, 120, benefits={"DEATH": 100_000_000.0},
                            calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
basis = fcf.Basis(mortality_annual=0.01, lapse_annual=0.03, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.10,
                  coverages=(fcf.CoverageRate("DEATH", 0.01),))

s = fcf.solvency.required_capital(mp, basis, regime=fcf.solvency.SII)
for name, c in s.sub_risk_capital.items():
    print(f"  {name:<10}{c:>14,.0f}")
print(f"  {'insurance':<10}{s.insurance_scr:>12,.0f}  interest {s.interest_capital:>12,.0f}")
print(f"  {'total SCR':<10}{s.total_scr:>12,.0f}  risk margin {s.risk_margin:>10,.0f}")
```

출력:

```text
  mortality      1,379,075
  longevity              0
  disability             0
  expense                0
  revision               0
  lapse         17,872,243
  catastrophe      209,175
  insurance   17,982,661  interest    3,301,100
  total SCR   21,283,761  risk margin  4,814,887
```

정기보험이라 사망과 해지만 뭅니다 (장수는 사망률 감소가 부채를 줄여 0, 장해·사업비·
revision 은 해당 담보·연금이 없어 0). 해지가 지배적인 것은 보장성 계약에서 대량해지가
깊게 무는 전형입니다. 보험위험 SCR 은 Annex IV 상관행렬로 합산되고, 금리위험 (EIOPA
만기별 충격) 이 더해지며, 위험마진은 자본비용 6% 를 자본 run-off 에 부과합니다.

## 같은 호출, 제도만 바꾸기 -- K-ICS

`regime` 만 `fcf.solvency.KICS` 로 바꿉니다. 코드는 그대로입니다.

```python
k = fcf.solvency.required_capital(mp, basis, regime=fcf.solvency.KICS)
for name, c in k.sub_risk_capital.items():
    print(f"  {name:<10}{c:>14,.0f}")
print(f"  {'insurance':<10}{k.insurance_scr:>14,.0f}  (aggregated)")
print(f"  {'risk margin':<10}{k.risk_margin:>10,.0f}")
```

출력:

```text
  mortality      1,150,285
  longevity              0
  morbidity              0
  lapse         13,404,182
  expense                0
  insurance     13,453,447  (aggregated)
  risk margin 5,381,379
```

K-ICS 가 덜 보수적입니다 -- 충격이 작고 (사망 +12.5% vs +15%, 해지 +/-35% vs +/-50%,
대량해지 30% vs 40%) 금리위험은 사용자 공급이라 여기선 0 입니다 (`KICS.interest_curves`
는 None). 그래서 같은 계약에 SCR 이 21.2M (SII) 에서 13.5M (K-ICS) 으로 작습니다.
위험마진도 방식이 달라 -- SII 는 자본비용 6%, K-ICS 는 보험위험액 x 0.40 입니다.

## 대재해위험액 -- factor 방식, 보험위험액에 합산

사망/장수/해지 등은 가정 충격-재측정이지만, **대재해위험액** (전염병·대형사고 같은 극단
사건) 은 가정 변동성으로 못 잡아 **가입금액에 위험계수** 를 매깁니다 (해설서 2-8). 전염병
= 사망담보 가입금액 x 0.1%, 대형사고 = 사망/장해/장기재물 가입금액에 지역노출 factor.
담보의 대재해 분류는 매핑 결정이라 **가입금액 버킷을 인자로** 받습니다.

```python
sa = 100_000_000.0                              # death sum assured (benefit x count)
cat = fcf.solvency.catastrophe_scr(pandemic_death=sa, accident_death=sa)
kc = fcf.solvency.required_capital(mp, basis, regime=fcf.solvency.KICS, catastrophe=cat)
print(f"catastrophe amount     = {cat:>14,.0f}")
print(f"insurance SCR (ex-cat) = {k.insurance_scr:>14,.0f}")
print(f"insurance SCR (w/ cat) = {kc.insurance_scr:>14,.0f}")
print(f"risk margin (ex-cat)   = {kc.risk_margin:>14,.0f}")
```

출력:

```text
catastrophe amount     =        100,013
insurance SCR (ex-cat) =     13,453,447
insurance SCR (w/ cat) =     13,480,841
risk margin (ex-cat)   =      5,381,379
```

대재해위험액 (100,013) 은 표6 상관 (사망 0.25, 장수 0, 장해질병/해지/사업비 0.25) 으로
보험위험액에 합산돼 13.45M -> 13.48M 으로 올라갑니다. **위험마진은 대재해를 제외** 합니다
(해설서: 위험마진 = 보험위험액 (대재해 제외) x 0.40). 보장성 책에선 대재해가 작지만,
전염병/대형사고 노출이 큰 책에선 더 커집니다.

## 장기재물·기타위험액 -- 코드 지정 +16% 충격

K-ICS 보험위험액의 일곱 번째 하위위험 **장기재물·기타** (장기손해보험의 재물/비용/배상/기타
담보) 는 해당 담보 위험률을 **+16% 충격** 해 재측정합니다 (해설서 2-5). 어느 담보가 장기재물
인지는 매핑 결정이라 **코드로 지정** 합니다 (`property_codes`). 표6 상관 (사망/장수/장해질병/
해지 0, 사업비 0.5, 대재해 0.25) 으로 보험위험액에 합산되고, 위험마진엔 **포함** 됩니다
(대재해와 달리).

```python
pmp = fcf.ModelPoints.single(45, 90_000.0, 240,
    benefits={"DEATH": 50_000_000.0, "FIRE": 30_000_000.0},
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH,
                         "FIRE": fcf.CalculationMethod.MORBIDITY})
pbasis = fcf.Basis(mortality_annual=0.005, lapse_annual=0.02, discount_annual=0.03,
    ra_confidence=0.75, mortality_cv=0.10, morbidity_cv=0.10,
    coverages=(fcf.CoverageRate("DEATH", 0.005), fcf.CoverageRate("FIRE", 0.02)))
pk = fcf.solvency.required_capital(pmp, pbasis, regime=fcf.solvency.KICS, property_codes=("FIRE",))
print(f"property capital       = {pk.sub_risk_capital['property']:>14,.0f}")
print(f"insurance SCR (w/ prop)= {pk.insurance_scr:>14,.0f}")
print(f"risk margin (incl prop)= {pk.risk_margin:>14,.0f}")
```

출력:

```text
property capital       =      1,191,442
insurance SCR (w/ prop)=      1,717,763
risk margin (incl prop)=        687,105
```

`FIRE` 담보를 장기재물로 지정하면 그 위험률을 16% 올려 재측정한 ΔBEL (1,191,442) 이
장기재물위험액이 되고, 표6 으로 다른 하위위험과 묶여 보험위험액에 들어갑니다. 대재해와
함께 주면 둘은 상관 0.25 로 묶입니다.

## 지급여력비율과 임베디드밸류 연결

요구자본 (분모) 은 이 엔진이 내지만, **가용자본 (분자) 은 자산-부채라 사용자 입력**
입니다 (자산 모델 밖).

```python
print(f"  SII    ratio @ AC=20,000,000 : {fcf.solvency.solvency_ratio(s, 20_000_000.0):.1%}")
print(f"  K-ICS  ratio @ AC=20,000,000 : {fcf.solvency.solvency_ratio(k, 20_000_000.0):.1%}")
```

출력:

```text
  SII    ratio @ AC=20,000,000 : 94.0%
  K-ICS  ratio @ AC=20,000,000 : 148.7%
```

같은 가용자본이라도 SII 가 SCR 이 커 비율이 낮습니다. SCR 은 [8.5 임베디드밸류](embedded-value)
의 자본비용으로도 들어갑니다 -- `embedded_value(..., required_capital=s.scr_path,
frictional_spread=basis.cost_of_capital_rate)` 로 요구자본 보유비용을 VNB 에서 차감.

## 함정 / 검증

- **부채측만 (v1)** -- 자산측 시장위험 (주식·부동산·신용), 가용자본, 운영위험은 자산
  모델이 필요해 범위 밖. `solvency_ratio` 의 가용자본은 사용자 입력이고, 자산위험이 큰
  책 (book) 에서는 비율의 상한입니다.
- **대재해 제외** -- SII 대재해 (+0.15%p) 는 v1 대칭을 위해 뺐고 (additive 변형으로 추가
  가능), K-ICS 대재해는 가입금액 factor (ΔBEL 아님) 라 엔진 밖입니다.
- **SII 장해 간이** -- +25% 단일 (원문 +35% 1년차 / +25% 이후 / 회복률 -20% 의 정상
  수준). 충격은 1차 출처 그대로, 연차 분리만 미모델.
- **대량해지 해지환급금** -- 해지 시점 환급금 유출을 반영합니다: basis 의
  `surrender_value_curve` 를 valuation date (가입 후 `elapsed_months`) 에서 평가해
  떠나는 비율에 지급 (`fraction x sum(count x 환급금)`). 따라서 대량해지 자본 =
  `fraction x sum(count x (환급금 - 계약별 BEL))` 로, 잃는 사업 가치에 즉시 유출이
  더해집니다. 환급금 곡선이 없는 basis 는 count haircut 만 (과거 동작). 계좌형 (변액·UL)
  의 계좌가치 기반 해지환급금은 아직 미포함 -- VFA 측 작업.
- **K-ICS 금리** -- AFDNS 모델 도출이라 정적 표가 아님. `KICS.interest_curves` 는 None
  이고 공식 충격곡선을 caller 가 공급합니다. SII 는 EIOPA 표가 내장.

## 인접 레시피

- [8.5 임베디드밸류](embedded-value) -- SCR 을 자본비용으로 받아 신계약가치에서 차감.
- [8.1 시나리오 / 민감도](sensitivity) -- 하위위험 충격이 쓰는 Basis 충격 관용구.
- [경제적 가정 -- 시나리오 생성](../basics/scenario-generation) -- 금리 곡선 충격의 토대.

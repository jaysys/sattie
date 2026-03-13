# 촬영가능성 계산 절차 조사

기준일: 2026-03-07

## 한 줄 결론

`촬영가능성 계산(feasibility analysis)`은 단순히 "지나가면 찍을 수 있다"를 확인하는 단계가 아니다. 실제로는 `기하 접근 가능성`, `광학/레이더 운용 조건`, `위성 자세·전력·열·메모리`, `지상국 downlink`, `기존 임무와의 충돌`, `날씨/구름`, `고객 요구 조건`을 함께 걸러서 `실제로 계약 가능한 촬영 제안서`를 만드는 과정이다.

## 왜 별도 절차가 필요한가

OGC Sensor Planning Service는 EO 위성 tasking에서 feasibility를 별도 기능으로 정의한다. 즉, 촬영 요청을 받으면 먼저 `feasibility`를 확인하고, 그 뒤에 `submit`, `reserve`, `status`, `cancel` 같은 후속 절차가 이어진다. OGC EO Satellite Tasking Extension은 더 나아가 `feasibility study`를 정식 결과 객체로 다루며, 이후 Submit/Reserve 요청은 이 feasibility 결과와 동일한 tasking parameter를 유지해야 한다고 규정한다.

실무적으로 이 말은 다음을 뜻한다.

- feasibility는 단순 사전조회가 아니라 `주문/계약 직전의 기술 판정`
- feasibility를 통과하지 못하면 uplink 단계로 갈 수 없음
- feasibility는 `계획 가능`, `안전`, `충돌 없음`, `다운링크 가능`까지 포함해야 함

ESA EO Framework도 mission planning 서비스가 `conflict free, feasible and safe instrument tasking and satellite downlink activities`를 제공해야 한다고 명시한다.

## 한국 사례에서 feasibility가 놓이는 위치

한국항공우주연구원은 위성 지상국 시스템을 `위성관제시스템`, `수행안테나시스템`, `영상처리시스템`으로 구분하고, 이 중 영상처리시스템이 `사용자 요청 접수, 촬영계획 수립, 영상처리 및 자료 배포`를 수행한다고 설명한다. 쎄트렉아이도 상용 지상시스템에서 `MCS(Mission Control System)`의 `Mission Planning`, `IRPS(Image Receiving & Processing System)`의 `Image Collection Planning`을 별도 기능으로 제시한다.

즉, 한국 운영 체계에서도 feasibility는 보통 아래 사이에 위치한다.

`사용자 요청 접수 -> 촬영가능성 계산 -> 촬영계획 수립 -> 명령계획 -> uplink`

## 촬영가능성 계산의 목적

전문가 관점에서 feasibility의 목적은 보통 5가지다.

1. `기술적 가능성`: 센서와 궤도가 목표를 물리적으로 관측 가능한가
2. `운용 가능성`: 위성이 해당 시점에 자세기동, 촬영, 저장, 다운링크를 수행할 수 있는가
3. `품질 가능성`: 고객이 요구한 구름량, 입사각, 해상도, 레벨 조건을 만족할 가능성이 있는가
4. `일정 가능성`: 요청한 기간 안에 시도 기회를 충분히 확보할 수 있는가
5. `계약 가능성`: 공급자가 제안 가능한 조건과 책임범위를 고객에게 수치로 설명할 수 있는가

## 절차 개요

촬영가능성 계산은 보통 아래 순서로 진행된다.

| 단계 | 무엇을 계산하는가 | 탈락 조건 예시 | 결과 |
| --- | --- | --- | --- |
| 1. 요청 정규화 | AOI, 시간창, 센서, 모드, 품질 조건 정리 | 필수 파라미터 누락, 최소 주문 단위 미달 | 계산용 표준 요청 |
| 2. 센서/상품 제약 검증 | 서비스 정책, 모드, 제품 레벨, 파일 형식 검증 | 지원하지 않는 모드, 불가능한 제품 조합 | 정책 적합 여부 |
| 3. 기하 접근 계산 | 궤도와 센서 시야로 AOI 접근 가능 시각 계산 | 시간창 내 접근 없음 | 후보 pass 목록 |
| 4. 자세/센서 운용성 계산 | off-nadir, incidence, slewing, duty cycle 검토 | 최대 기동각 초과, 센서 모드 불가 | 촬영 가능한 segment |
| 5. 광학/환경 조건 계산 | 태양고도, 구름, 계절성, SAR/광학 특성 반영 | 광학 야간, 구름 확률 과다 | 성공 확률 추정 |
| 6. 위성 자원 계산 | 메모리, 전력, 열, onboard duty cycle 검토 | recorder overflow, thermal margin 부족 | 자원 적합 여부 |
| 7. 지상국/downlink 계산 | 가시 패스, 링크 속도, backlog, station availability 검토 | 내려받을 창구 없음 | acquisition-to-downlink 연결성 |
| 8. 충돌 해소 및 우선순위 반영 | 기존 촬영·재난임무·긴급주문과 deconflict | 같은 시간대 경쟁에서 밀림 | 채택/보류/거절 |
| 9. 제안서 생성 | 성공가능도, 첫 시도일, 권장 완화조건 산출 | - | Feasibility proposal |

## 단계별 상세

### 1. 요청 정규화

이 단계에서는 고객 입력을 기계가 계산 가능한 구조로 바꾼다.

대표 입력은 다음과 같다.

- AOI
- acquisition window
- cloud cover limit
- off-nadir limit 또는 incidence angle range
- sensor type / imaging mode / polarization
- product level
- priority

한국 SI Imaging Services 주문 페이지도 이와 유사한 항목을 그대로 요구한다. AOI는 원, 사각형, SHP/KML/KMZ로 넣을 수 있고, cloud cover, off-nadir, 제품 레벨, 촬영모드, polarization, 제품 처리 레벨을 지정하도록 되어 있다.

정규화 단계에서 흔한 탈락 사유는 다음이다.

- AOI 최소 폭 미달
- 제품 레벨과 센서 모드 조합 불일치
- 특정 우선순위 옵션에서 허용되지 않는 제약 입력

예를 들어 SIIS는 `Priority Plus` 옵션에서는 feasibility study를 수행하지 않으며 주문 시 cloud cover와 tilt angle을 지정할 수 없다고 밝힌다. 반대로 `Assured`, `Priority`, `Standard`는 feasibility study 후 proposal을 제공한다.

### 2. 서비스/정책 제약 검증

이 단계는 물리 계산 전에 `사업 규칙`을 검증하는 단계다.

검증 항목 예시는 다음과 같다.

- 최소 주문 면적 충족 여부
- 특정 센서/위성의 제공 가능 여부
- 특정 제품 레벨 제공 가능 여부
- 주문 마감 시간 이전 확정 여부
- 우선순위 옵션별 서비스 규칙 충족 여부

SIIS 공개 정책에는 다음이 포함된다.

- 최소 주문 단위 `100 km²`
- AOI 최소 폭 `5 km`
- Priority는 촬영 시작 `하루 전 03:00 UTC` 전까지 주문 확정
- Standard는 촬영 시작 `2일 전 03:00 UTC` 전까지 주문 확정

즉, feasibility는 우주공학 계산만이 아니라 `영업/운영 SLA 계산`도 포함한다.

### 3. 기하 접근 가능성 계산

이 단계는 "위성이 시간창 안에 그 목표를 시야에 넣을 수 있는가"를 본다.

핵심 입력은 다음과 같다.

- 위성 궤도요소(TLE 또는 정밀 궤도력)
- 센서 시야폭(swath)
- 최대 기동각(max off-nadir / incidence angle)
- AOI 좌표와 면적
- 요청 시간창

실무적으로는 후보 `pass list`를 만든다. 각 후보 pass마다 다음이 계산된다.

- 접근 시각
- 목표 중심점과 궤도의 cross-track 거리
- 요구 기동각
- 예상 촬영 duration
- segment 수와 covering ratio

근사식으로는 다음 같은 1차 계산을 많이 쓴다.

```text
one-sided cross-track reach d_max ≈ h * tan(theta_max)
```

여기서:

- `h`: 위성 고도
- `theta_max`: 허용 최대 off-nadir 각

예를 들어 고도 `500 km`, 최대 off-nadir `25°`이면:

```text
d_max ≈ 500 * tan(25°) ≈ 233 km
```

즉, 목표가 궤도 지상투영선에서 대략 `233 km`보다 더 멀면 그 pass는 기하적으로 탈락할 가능성이 높다. 실제 구현에서는 지구 곡률, 센서 footprint, 자세 제약, swath overlap까지 더 정밀하게 계산한다.

### 4. 자세·센서 운용성 계산

기하적으로 접근 가능하다고 끝이 아니다. 그 자세로 실제 센서 운용이 가능한지 봐야 한다.

검토 항목 예시는 다음과 같다.

- 최대 roll/pitch/yaw 기동 한계
- 기동 속도와 settle time
- 촬영 전후 다른 임무와의 자세 충돌
- 센서 warm-up 시간
- stripmap / spotlight / wide swath 같은 모드 제약
- SAR polarization 조합 제약

NASA는 mission planning software가 `satellite dynamics`와 `component capability` 모델을 포함해야 하며, 이벤트는 순서와 시간 간격을 가진 action series로 계획되고 시뮬레이션을 통해 반복 조정된다고 설명한다.

즉, feasibility 계산은 보통 다음 질문을 포함한다.

- 이 AOI를 찍기 위해 필요한 회전량이 reaction wheel 또는 자세제어 규칙 안에 드는가
- 이전 촬영에서 다음 촬영까지 자세를 복귀시킬 시간이 있는가
- 센서를 정해진 모드로 켜고 끄는 시간이 맞는가

### 5. 광학/기상 또는 레이더 조건 계산

이 단계에서 `광학`과 `SAR`의 feasibility 특성이 크게 갈린다.

광학 위성은 보통 다음을 본다.

- 태양고도
- 태양방위각
- 계절/현지시각
- cloud cover 또는 haze 조건
- shadow 영향

SAR 위성은 구름 영향은 작지만 다음을 더 본다.

- incidence angle 범위
- layover/shadow 위험 지형
- desired polarization
- repeat-pass 조건

OGC EO Tasking Extension은 feasibility 결과가 `cell` 또는 `segment` 단위로 제공될 수 있고, 각 셀의 `성공 가능성`과 `whole area before the end of the requested time window`에 대한 `overall chance`를 포함할 수 있다고 설명한다. 또한 이 확률은 `climate, weather and satellite workload conditions`를 반영할 수 있다고 명시한다.

이 점이 중요하다. feasibility는 단순 yes/no가 아니라 아래처럼 확률형으로 나오는 것이 정상이다.

- 첫 시도 가능 시각
- 예상 성공일
- 셀 단위 성공 가능도
- 전체 AOI 성공 가능도

### 6. 위성 자원 계산

여기서는 촬영 자체보다 `찍고 저장하고 유지할 수 있는지`를 본다.

대표 제약은 다음과 같다.

- recorder free space
- downlink 전까지 누적 backlog
- 전력 budget
- 배터리 SOC
- thermal limit
- instrument duty cycle
- 하루/궤도당 최대 촬영 수

데이터량은 보통 다음 구조로 계산한다.

```text
D_task ≈ R_raw * t_img / C_comp
```

여기서:

- `D_task`: 촬영 1건 데이터량
- `R_raw`: 원시 데이터 생성률
- `t_img`: 촬영 지속시간
- `C_comp`: onboard compression ratio

CCSDS는 onboard data compression이 `data storage`와 `downlink capacity`의 제약을 완화하기 위해 필요하다고 설명한다.

### 7. downlink 및 지상국 자원 계산

실무에서 많은 주문이 여기서 탈락한다. `찍는 것`보다 `내려받는 것`이 더 빡빡한 경우가 많기 때문이다.

검토 항목은 다음과 같다.

- 촬영 후 첫 downlink 가능한 지상국 패스
- 지상국 사용 가능 여부
- 동시간대 다른 위성과의 충돌
- link rate와 required volume
- bit error margin
- backlog clearing time

NASA는 mission planning and scheduling이 `link and loading analyses`와 `operational schedules` 생성을 포함한다고 설명한다. 또한 ground station scheduling software는 orbit simulation으로 contact conflicts를 계산하고 schedule optimization을 지원한다고 설명한다.

수용 가능한 downlink 용량은 보통 다음 식으로 잡는다.

```text
C_down ≈ R_link * t_pass * eta
```

여기서:

- `R_link`: 링크 속도
- `t_pass`: 유효 downlink 시간
- `eta`: 프로토콜/링크 효율

`D_task`가 `C_down`을 지속적으로 초과하면, 기하적으로 촬영 가능해도 운영상 infeasible이 될 수 있다.

### 8. 충돌 해소와 우선순위 반영

기술적으로 feasible인 주문 여러 개가 동시에 존재하면, 운영시스템은 이를 다 수용하지 못할 수 있다. ESA도 mission planning이 station unavailability, interference, deconfliction을 고려해야 한다고 명시한다.

이 단계에서 보통 다음이 적용된다.

- 긴급재난 > assured > priority > standard
- 국책 임무 > 상용 임무
- 이미 예약된 촬영의 보호
- 다운링크 병목 시 고가치 데이터 우선

따라서 feasibility 결과는 다음 셋 중 하나로 정리되는 경우가 많다.

- `Feasible`
- `Conditionally feasible`
- `Currently infeasible / rejected`

### 9. Feasibility proposal 산출

최종 산출물은 내부적으로는 task segment/cell 리스트일 수 있고, 고객에게는 보통 `proposal` 형식으로 전달된다.

보통 들어가는 내용은 다음과 같다.

- 촬영 가능한 기간
- 예상 첫 시도일
- 예상 시도 횟수
- 요구 조건 완화 시 개선 효과
- 실패 위험 요인
- 우선순위 옵션에 따른 SLA

SIIS도 `Assured`, `Priority`, `Standard` 옵션에서 feasibility proposal을 제공한다고 밝히고 있다.

## 수치 모델: 전문가 실무에서 자주 쓰는 판정 방식

### 1. hard constraint와 soft constraint를 분리한다

전문가들은 feasibility를 보통 아래처럼 나눈다.

| 유형 | 예시 | 판정 방식 |
| --- | --- | --- |
| `Hard constraints` | 시간창 내 접근 없음, 최대 기동각 초과, 지원하지 않는 모드, recorder overflow, downlink 창 없음 | 하나라도 실패하면 즉시 reject |
| `Soft constraints` | 구름 가능성, workload 경합, 예보 불확실성, 지상국 재배치 가능성 | 확률 또는 score로 반영 |

### 2. 성공 확률을 곱셈형으로 근사한다

후보 촬영 기회 `i`의 성공확률을 다음처럼 근사할 수 있다.

```text
P_i ≈ P_geo,i * P_env,i * P_resource,i * P_downlink,i
```

그리고 후보 pass가 여러 개 있으면 적어도 한 번 성공할 확률은:

```text
P_total = 1 - Π(1 - P_i)
```

주의:

- 이 식은 각 pass의 성공 여부가 독립이라는 강한 가정을 둔다.
- 실제 운영에서는 날씨와 workload가 상관되어 있으므로 과대평가될 수 있다.
- 따라서 운영기관은 내부적으로 보수계수(safety margin)를 둔다.

### 3. 결과를 수치 등급으로 제시한다

실무 보고서에서는 보통 이런 식이 읽기 쉽다.

| P_total | 해석 |
| --- | --- |
| `>= 0.85` | 높은 타당성 |
| `0.60 ~ 0.85` | 조건부 타당 |
| `0.30 ~ 0.60` | 낮은 타당성, 조건 완화 권고 |
| `< 0.30` | 현재 기준 infeasible에 가까움 |

이 기준은 운영기관별로 다르며, 아래 수치 예제는 설명용 모델이다.

## 수치 예제 1: 광학 주문의 feasibility 계산

### 조건

- 유형: 저궤도 광학 위성 신규촬영
- AOI: `20 km x 20 km`
- 요청 기간: `7일`
- 최대 cloud cover: `20%`
- 최대 off-nadir: `25°`
- 위성 고도: `500 km` 가정

### 1단계. 기하 접근 필터

최대 cross-track reach를 근사하면:

```text
d_max ≈ 500 * tan(25°) ≈ 233 km
```

7일 안 후보 pass 5개에 대해, 목표의 cross-track 거리 추정값이 아래와 같다고 가정한다.

| pass | cross-track distance | 기하 판정 |
| --- | --- | --- |
| P1 | 80 km | 통과 |
| P2 | 260 km | 탈락 |
| P3 | 150 km | 통과 |
| P4 | 310 km | 탈락 |
| P5 | 120 km | 통과 |

즉, `5개 후보 중 3개`가 기하적으로 남는다.

### 2단계. 태양/품질 조건 필터

광학 위성이므로 남은 3개 중 태양고도와 그림자 조건을 본다.

- P1: 태양고도 적합
- P3: 현지시각이 이르며 그림자 조건 불리
- P5: 태양고도 적합

따라서 `실질 후보`는 `P1`, `P5` 두 개만 남는다고 본다.

### 3단계. 기상 확률 반영

운영기관이 예보와 계절 통계를 반영해 다음과 같이 추정했다고 가정한다.

- `P1의 cloud 조건 만족 확률 = 0.45`
- `P5의 cloud 조건 만족 확률 = 0.35`

기하, 자원, downlink는 두 pass 모두 충분하다고 보고:

- `P_geo = 1.0`
- `P_resource = 0.98`
- `P_downlink = 0.97`

그러면 각 pass의 성공확률은:

```text
P1 ≈ 1.0 * 0.45 * 0.98 * 0.97 ≈ 0.428
P5 ≈ 1.0 * 0.35 * 0.98 * 0.97 ≈ 0.333
```

적어도 한 번 성공할 확률은:

```text
P_total = 1 - (1 - 0.428) * (1 - 0.333)
        ≈ 1 - (0.572 * 0.667)
        ≈ 0.619
```

즉, `약 61.9%`다.

### 해석

- 현재 요구조건대로는 `조건부 타당`
- 고객이 cloud cover 허용치를 `20% -> 30%`로 완화하거나 시간창을 `7일 -> 14일`로 늘리면 P_total이 크게 증가할 가능성이 높다
- 따라서 proposal에는 `현재 조건 기준 성공확률 약 0.62, 기간 연장 권고`가 적절하다

## 수치 예제 2: SIIS Standard 정책과 확률 해석

SIIS는 `Standard` 옵션에서 고객이 만족하는 유효 영상을 획득하지 못하더라도 `최대 10회 촬영`을 진행한다고 공개한다.

이 정책은 feasibility 관점에서 매우 중요하다. pass 1회 성공확률이 낮아도 시도 횟수를 늘리면 전체 성공확률이 급격히 올라가기 때문이다.

예를 들어 pass당 유효 영상 확보 확률이 `p = 0.30`이라고 가정하면:

```text
P_total(10 attempts) = 1 - (1 - 0.30)^10
                     = 1 - 0.7^10
                     ≈ 0.972
```

즉, `약 97.2%`다.

같은 계산을 보면:

- `3회 시도`: `1 - 0.7^3 = 65.7%`
- `5회 시도`: `1 - 0.7^5 = 83.2%`
- `10회 시도`: `97.2%`

### 해석

- feasibility는 단일 pass 성공 여부가 아니라 `시간창 전체에서의 누적 성공확률`로 봐야 한다
- Standard 옵션이 장기 time window를 가지는 이유를 수치적으로 설명할 수 있다
- 이 계산은 pass 독립을 가정한 설명용 예제이며, 실제로는 연속된 악천후 때문에 성공확률이 이보다 낮아질 수 있다

## 수치 예제 3: SAR 주문은 기상보다 자원·다운링크가 지배적일 수 있다

### 조건

- 유형: SAR urgent tasking
- 요청 기간: `48시간`
- incidence angle 허용 범위: `25° ~ 40°`
- 후보 pass 3개

### 1단계. 기하/입사각 판정

예상 incidence angle이 다음과 같다고 가정한다.

| pass | incidence angle | 판정 |
| --- | --- | --- |
| S1 | 28° | 통과 |
| S2 | 42° | 탈락 |
| S3 | 33° | 통과 |

구름은 거의 의미가 없으므로, S1과 S3가 주 후보다.

### 2단계. 데이터량과 downlink 계산

한 번 촬영 시:

- 촬영 지속시간 `70 s`
- 원시 데이터율 `320 Mbit/s`
- onboard compression ratio `1.8`

그러면 task당 데이터량은:

```text
D_task ≈ 320 * 70 / 1.8
       ≈ 12,444 Mbit
       ≈ 12.4 Gbit
```

다음 X-band pass에서:

- 링크 속도 `310 Mbit/s`
- 유효 시간 `400 s`
- 효율 `0.75`

이면 downlink 가능량은:

```text
C_down ≈ 310 * 400 * 0.75
       ≈ 93,000 Mbit
       ≈ 93 Gbit
```

즉, `12.4 Gbit` 데이터 1건은 downlink 관점에서 충분히 feasible하다.

### 3단계. recorder backlog 반영

하지만 이미 recorder에 `36 Gbit` backlog가 쌓여 있고, 안전여유를 뺀 실사용 가능 메모리가 `45 Gbit`라면:

```text
36 + 12.4 = 48.4 Gbit
```

즉, `메모리 측면에서는 infeasible`이 된다.

### 해석

- 같은 주문이 `기하적으로는 feasible`
- `기상적으로도 feasible`
- 하지만 `recorder capacity` 때문에 현재 궤도 사이클에서는 infeasible

이 경우 운영자는 다음 중 하나를 제안한다.

- 기존 backlog를 먼저 downlink한 뒤 다음 cycle에 촬영
- 저우선순위 task를 밀어 메모리를 확보
- 촬영 길이 또는 모드를 축소

이 예제는 feasibility가 단순 접근 계산이 아니라 `mission-wide schedulability`라는 점을 보여준다.

## 케이스 해석: 어떤 요청이 실제로 더 타당한가

### 케이스 A. 광학, 짧은 기간, 강한 cloud 제약

- 장점: 제품 품질 높음
- 약점: 기상 불확실성 큼
- 타당성: `0.4 ~ 0.7` 수준으로 흔들리기 쉬움

### 케이스 B. 광학, 긴 기간, Standard 옵션

- 장점: 시도 횟수 확보 가능
- 약점: 납기 지연
- 타당성: 누적확률이 크게 올라감

### 케이스 C. SAR, 긴급 주문

- 장점: 구름 영향 적음
- 약점: incidence angle, backlog, downlink가 핵심 병목
- 타당성: 기상보다는 자원·스케줄이 판정 지배

## 전문가가 feasibility proposal에 넣어야 하는 숫자

좋은 feasibility proposal은 단순 `가능/불가능`보다 아래 숫자를 제시해야 한다.

- 후보 촬영 기회 수
- 첫 시도 가능일
- 성공확률 `P_total`
- 주요 실패 요인별 기여도
- recorder/downlink margin
- 조건 완화 시 기대 개선폭

예를 들면 다음처럼 쓰는 것이 실무적으로 유용하다.

| 항목 | 예시 |
| --- | --- |
| 요청 | 서울 인근 AOI 20x20 km, 7일, cloud <= 20% |
| 후보 pass | 5 |
| 기하 통과 | 3 |
| 품질 통과 예상 | 2 |
| 누적 성공확률 | 61.9% |
| 주요 리스크 | 구름, 짧은 시간창 |
| 완화 권고 | 기간 14일 연장 시 확률 80%+ 예상 |

## 최종 정리

촬영가능성 계산의 핵심은 `보일까?`가 아니라 `요청 조건대로, 안전하게, 제때, 내려받을 수 있게, 계약 가능한 확률로 실행할 수 있는가?`다.

전문가 수준에서 feasibility는 보통 아래를 동시에 계산한다.

- `geometry`
- `attitude and sensor operability`
- `illumination or radar geometry`
- `weather/climate probability`
- `recorder/power/thermal margins`
- `ground-station/downlink capacity`
- `task conflicts and priority`

그리고 최종 결과는 yes/no보다 아래 형태가 적절하다.

- `feasible`
- `conditionally feasible`
- `infeasible`
- `recommended relaxation`
- `numerical probability of success`

즉, 좋은 feasibility 분석은 고객 요청을 위성 운용 언어로 번역하고, 그 결과를 다시 고객이 이해할 수 있는 `확률과 조건`으로 되돌려주는 작업이다.

## 출처

1. OGC, Sensor Planning Service Standard: https://www.ogc.org/publications/standard/sps/
2. OGC, EO Satellite Tasking Extension for SPS 2.0: https://portal.ogc.org/files/?artifact_id=40185
3. ESA EO Framework Specifications: https://eof.esa.int/document/esa-eo-framework-eof-csc-specifications/
4. NASA SmallSat Institute, Ground Data Systems and Mission Operations: https://www.nasa.gov/smallsat-institute/sst-soa/ground-data-systems-and-mission-operations
5. 한국항공우주연구원, 위성 운영기술 연구: https://www.kari.re.kr/kor/contents/62
6. Satrec Initiative, Ground Systems: https://www.satreci.com/page/22
7. SI Imaging Services, 바로 주문하기: https://www.si-imaging.com/kr/page/32

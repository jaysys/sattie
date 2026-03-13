“LEO 위성”은 **Low Earth Orbit Satellite**, 즉 **저궤도 위성**을 의미합니다.

## 1. 정의

- **고도**: 지표면에서 약 **160km ~ 2,000km**
- 지구와 비교적 가까운 궤도를 도는 인공위성

## 2. 특징

### ① 지연시간(Latency)이 낮음

- 지구와 가까워 **통신 지연이 짧음**
- 위성 인터넷, 실시간 통신 서비스에 유리

### ② 고속·고품질 통신 가능

- 대역폭 확보가 상대적으로 용이
- 5G/6G 백홀, 위성 브로드밴드에 활용

### ③ 수명은 비교적 짧음

- 대기 저항 영향으로 **5~7년 수준**
- 지속적 재발사 필요

### ④ 다수 위성 군집(Consetllation) 필요

- 커버리지 확보를 위해 수십~수천 기 운용
  - 예: SpaceX **Starlink**
  - Amazon **Project Kuiper**
  - OneWeb

---

## 3. LEO vs 다른 궤도 비교

| 구분      | LEO           | MEO            | GEO                 |
| --------- | ------------- | -------------- | ------------------- |
| 고도      | ~2,000km 이하 | 2,000~35,786km | 35,786km            |
| 지연시간  | 낮음          | 중간           | 높음                |
| 위성 수   | 많음 필요     | 중간           | 3~4기면 전지구 커버 |
| 대표 사례 | Starlink      | GPS            | 기상위성            |

---

## 4. 주요 활용 분야

- 위성 인터넷 (Starlink)
- 군 통신
- IoT 글로벌 커버리지
- 지구관측(정찰·환경 모니터링)
- 항공·해상 통신

---

네. **위성 궤도 구분은 기본적으로 “고도(altitude)” 기준**입니다.

다만, 고도만이 전부는 아니고 **궤도 특성(주기, 지구 자전과의 관계, 경사각 등)**도 함께 고려됩니다.

---

## 1️⃣ 고도 기준 1차 분류

### ■ LEO (Low Earth Orbit)

- **160 ~ 2,000 km**
- 공전주기: 약 90~120분
- 예: Starlink, 지구관측위성

### ■ MEO (Medium Earth Orbit)

- **2,000 ~ 35,786 km**
- 공전주기: 수 시간
- 예: GPS (약 20,200 km)

### ■ GEO (Geostationary Earth Orbit)

- **35,786 km**
- 공전주기: 24시간 (지구 자전과 동일)
- 지상에서 보면 정지해 보임
- 예: 기상위성, 방송통신위성

---

## 2️⃣ 왜 고도로 나누는가?

고도가 바뀌면 다음이 달라집니다:

- **공전 주기**
- **통신 지연 시간**
- **커버리지 면적**
- **위성 수명**
- **발사 비용**
- **필요 위성 수**

즉, 고도는 위성의 **물리적·경제적 특성**을 결정하는 핵심 변수입니다.

---

## 3️⃣ 하지만 고도만이 기준은 아님

예를 들어:

- **SSO (Sun-Synchronous Orbit)** → 태양동기궤도 (고도는 LEO 범위지만 ‘궤도 경사각’으로 분류)
- **극궤도(Polar Orbit)** → 남북으로 도는 궤도
- **타원궤도(HEO)** → 고도 차이가 큰 궤도

즉,
👉 **LEO/MEO/GEO는 고도 중심 분류**
👉 **SSO/극궤도 등은 궤도 기하학적 특성 중심 분류**

---

한 줄 정리:

> “LEO냐 GEO냐”는 고도 기준이고,
> “극궤도냐 태양동기궤도냐”는 궤도 형태 기준입니다.

고도 외에도 위성 궤도는 여러 **물리적·운용적 기준**으로 분류합니다. 체계적으로 정리해 드리겠습니다.

---

# 1️⃣ 궤도 기하학(형태) 기준

### ■ 원궤도 (Circular Orbit)

- 이심률(e) ≈ 0
- 고도 일정
- 통신·지구관측에 일반적

### ■ 타원궤도 (Elliptical Orbit)

- 이심률(e) > 0
- 근지점/원지점 고도 차이 존재
- 예: Molniya 궤도 (고위도 통신용)

---

# 2️⃣ 궤도 경사각(Inclination) 기준

### ■ 적도궤도 (Equatorial Orbit)

- 경사각 0°
- GEO는 모두 여기에 해당

### ■ 경사궤도 (Inclined Orbit)

- 0° < i < 90°

### ■ 극궤도 (Polar Orbit)

- 경사각 ≈ 90°
- 지구 전체 관측 가능

---

# 3️⃣ 태양과의 관계 기준

### ■ 태양동기궤도 (SSO)

- 매일 같은 태양 고도 조건에서 지표 통과
- 광학 지구관측에 최적
- 고도는 LEO 범위 (약 600~800km)

---

# 4️⃣ 지구 자전과의 관계 기준

### ■ 정지궤도 (Geostationary Orbit)

- 고도 35,786km
- 지상에서 정지처럼 보임

### ■ 동기궤도 (Geosynchronous Orbit)

- 공전주기 24시간
- 반드시 정지해 보이지는 않음 (경사각 있으면 ‘8자’ 궤적)

---

# 5️⃣ 임무 목적 기준

### ■ 통신 위성

- GEO, LEO 군집

### ■ 항법 위성

- MEO (GPS, Galileo 등)

### ■ 지구관측

- LEO, SSO, 극궤도

### ■ 정찰/군사

- LEO, HEO

---

# 6️⃣ 특수 궤도

### ■ Molniya 궤도

- 고이심률 타원
- 고위도 지역 장시간 커버

### ■ Tundra 궤도

- 24시간 주기 타원궤도

### ■ 라그랑주점 궤도 (L1, L2 등)

- 지구-태양 중력 균형점
- 예: 제임스 웹 우주망원경 (L2)

---

# 정리 구조

| 분류 기준   | 예시              |
| ----------- | ----------------- |
| 고도        | LEO / MEO / GEO   |
| 이심률      | 원궤도 / 타원궤도 |
| 경사각      | 적도 / 극궤도     |
| 태양 동기   | SSO               |
| 자전 동기   | GEO / GSO         |
| 중력 균형점 | Lagrange          |

---

실무적으로는 보통
👉 “고도 + 경사각 + 이심률 + 임무목적”
이 4가지를 조합해 궤도를 정의합니다.

네 — **지상국(Ground Station)**이 위성으로 **여러 개의 촬영 계획(이미징 태스크)**을 동시에/연속적으로 업링크하는 것은 _단일 명령만 보내는 게 아니라_, 운영 소프트웨어와 스케줄링 시스템을 통해 **다수의 촬영계획을 관리·전송**하는 구조로 이뤄집니다. 자세히 정리하면 다음과 같습니다:

---

## ✅ 1) 위성 운용 스케줄과 계획 데이터

위성 임무운영센터(Mission Operations Center, MOC)에서는 각 타겟에 대한 촬영요청을 **임무계획(Tasking Plan)** 으로 정리합니다. 이 계획은 다음 요소들을 포함합니다:

- 촬영 위치/시간/각도
- 우선순위
- 저장/전송 우선 정책
- 제약 조건 (에너지, 저장소, 커버시간)

즉, **단순한 단일 커맨드가 아니라 복수의 계획이 스케줄링 알고리즘을 통해 최적화된 형태로 조합**됩니다. ([link.springer.com][1])

---

## ✅ 2) Schedule(스케줄) + Visibility 고려

지상국은 위성과의 **접촉 가시창(visibility window)** 정보를 궤도 예측으로 계산합니다. 이 정보를 이용해 어떤 명령을 언제 업링크할지 **스케줄링**합니다.

- 궤도 계산 및 TLE(궤도 요소) 기반 스케줄링
- 위성과 지상국간 통신 가용시간
- 촬영 타임워인도 고려
- 임무 우선순위 기반 최적 스케줄 생성

즉, 여러 개 계획은 단순 나열이 아니라 **실행 가능 시간순으로 스케줄링 → 업링크** 됩니다. ([nasa.gov][2])

---

## ✅ 3) 업링크 방식

지상국에 의해 위성으로 보내는 업링크 명령은 아래처럼 구성됩니다:

📌 **Batch Command Upload (배치 업로드)**

- 다수의 명령을 묶어서 한 번에 보냄
- 위성이 가시창 안에 있을 때 가능한 한 많은 명령을 업로드

📌 **Sequenced Command Sets (순차 명령)**

- 명령 블록에 실행순서(time-tagged) 또는 조건부 실행 정보를 포함
- 예: 12:10~12:15 촬영 태스크1, 12:25 촬영 태스크2 등

📌 **Priority / Conflict Resolution**

- 위성 자원(전력, 노출 시간, 저장 공간) 최적화
- 높은 우선순위가 있는 명령이 먼저 실행되도록 계획 변경 시행

이는 GEO나 LEO 모두 기본 구조는 동일하지만 **가시창과 통신 창구 시간이 짧은 LEO에서는 배치 업링크가 더 중요**합니다. ([nasa.gov][2])

---

## ✅ 4) 예 — 실제 운영 사례

프랑스 CNES의 **Pléiades** 지상국 사례를 보면:

✔ 하루에 **3회** 위성계획을 갱신하여
✔ 각 패스(pass)마다 **다수의 촬영계획을 업링크**
✔ 날씨 정보 등 최신 상태를 반영하여 계획 갱신

이처럼 지상국은 **임무계획 전체(Tasking Plan)** 을 위성에 전송하고, 위성은 이를 *스케줄에 따라 순차적으로 수행*합니다. ([en.wikipedia.org][3])

---

## ✅ 5) 정리

| 단계          | 내용                                                         |
| ------------- | ------------------------------------------------------------ |
| 임무계획 수립 | 여러 촬영 요청을 통합하여 전체 계획 생성 ([Springer][1])     |
| 스케줄 최적화 | 가시창/연산 제약/우선순위 기반 최적 스케줄링 ([nasa.gov][2]) |
| 업링크        | 배치 upload + 순차 실행 커맨드 세트 업링크                   |
| 위성 실행     | onboard scheduler가 시간/조건 따라 촬영 수행                 |

---

## 부가

[1]: https://link.springer.com/book/10.1007/978-981-99-3565-9?utm_source=chatgpt.com "Earth Observation Satellites: Task Planning and Scheduling | Springer Nature Link"
[2]: https://www.nasa.gov/smallsat-institute/sst-soa/ground-data-systems-and-mission-operations/?utm_source=chatgpt.com "11.0 Ground Data Systems and Mission Operations - NASA"
[3]: https://en.wikipedia.org/wiki/Pl%C3%A9iades_%28satellite%29?utm_source=chatgpt.com "Pléiades (satellite)"

좋습니다. 위에서 제시한 4가지를 **순서대로, 기술 수준으로** 설명하겠습니다.

---

# 1️⃣ Onboard Scheduler (위성 내부 스케줄러)

지상에서 여러 촬영계획을 업링크하면, 위성 내부에는 이를 실행하는 **온보드 임무 스케줄러(Onboard Mission Scheduler)**가 있습니다.

## 구조

업링크 데이터는 보통 다음 형태로 저장됩니다:

- Time-tagged command list (시각 지정 명령열)
- Observation timeline
- Resource allocation table

## 동작 방식

① 위성 OBC(Onboard Computer)가 시간 기준으로 큐(queue)를 관리
② 현재 시각과 일치하는 태스크를 실행
③ 실행 전 리소스 확인

- 배터리 SOC
- 저장공간
- 자세 안정 여부
  ④ 조건 충족 시 촬영 실행

즉, 지상에서 "A, B, C 촬영하라"를 보내도
위성은 내부적으로 **시간·자원·상태 조건을 검증 후 실행**합니다.

---

# 2️⃣ 우선순위 / 긴급 관측 처리 로직

실제 상업 지구관측 위성은 하루 수백~수천 개의 요청을 받습니다.

## 우선순위 계층 예시

- P0: 군사/국가 긴급
- P1: 재난 대응
- P2: 상업 고객
- P3: 연구/저우선

### 처리 방식

### (1) 지상에서 재스케줄링 후 재업링크

가장 일반적

### (2) Onboard Replanning 기능

고급 위성은 제한적 자율 재계획 가능

예:

- 긴급 명령 uplink
- 기존 태스크 중 낮은 우선순위 drop
- 메모리 재할당

특히 LEO는 하루 여러 패스가 있으므로
**다음 패스에서 전체 계획을 수정하여 다시 업로드**하는 방식이 보편적입니다.

---

# 3️⃣ 지상국 자동화 / ML 기반 스케줄링

대형 사업자(Planet, Maxar 등)는 수작업이 아닙니다.

## 스케줄링 문제는 본질적으로

> NP-hard combinatorial optimization 문제

제약 조건:

- 가시창
- 태양 고도
- 클라우드 예측
- 위성 자세 기동 시간
- 배터리/저장공간
- 다운링크 시간

## 사용 기법

- Mixed Integer Linear Programming (MILP)
- Genetic Algorithm
- Simulated Annealing
- Reinforcement Learning 기반 재계획

최근에는

- 기상예측 데이터
- 실시간 수요 예측
- 군집 위성 간 load balancing

까지 포함됩니다.

---

# 4️⃣ 명령 전송 프로토콜 (CCSDS)

대부분의 위성은 **CCSDS 표준**을 따릅니다.

## 기본 구조

### (1) Telecommand (TC)

지상 → 위성

구성:

- Primary Header
- Secondary Header (time tag 등)
- Application Data
- Error Control

### (2) Telemetry (TM)

위성 → 지상

## 보안

- AES 기반 암호화
- 인증 키
- Anti-replay protection

## 전송 계층

- S-band
- X-band
- Ka-band

LEO 소형 위성은 보통 S-band uplink 사용.

---

# 🔎 핵심 요약

질문으로 돌아가서:

> 여러 촬영계획을 묶어서 보내느냐?

정확한 답:

✔ 예. **여러 촬영계획을 하나의 time-tagged 명령 세트로 배치 업링크한다.**
✔ 위성은 onboard scheduler가 이를 실행한다.
✔ 우선순위 변경 시 다음 패스에서 재업로드한다.
✔ 모든 통신은 CCSDS 기반 패킷 구조로 전송된다.

---

좋습니다. 제시한 4개 항목을 **실제 운용 관점에서 순서대로, 예시 포함** 설명하겠습니다.

---

# 1️⃣ 실제 촬영계획 데이터 포맷 예시

촬영계획은 단순 “사진 찍어라”가 아닙니다. 보통 **Observation Request → Scheduled Task → Telecommand Packet** 단계로 변환됩니다.

## (1) 지상 임무계획 단계 (논리 포맷 예)

```yaml
Observation_ID: 20260310-001
Target_Lat: 37.5665
Target_Lon: 126.9780
Start_Time_UTC: 2026-03-10T03:15:20Z
End_Time_UTC: 2026-03-10T03:15:45Z
Off_Nadir_Angle: 18 deg
Imaging_Mode: Panchromatic
Priority: P1
Cloud_Threshold: 20%
```

---

## (2) 위성용 실행 스케줄 테이블 예

| Time (UTC) | Command    | Parameter     |
| ---------- | ---------- | ------------- |
| 03:14:55   | Slew_Start | Target Vector |
| 03:15:18   | Camera_On  | Mode=Panchro  |
| 03:15:45   | Camera_Off | —             |
| 03:16:00   | Data_Tag   | Obs_ID        |

---

## (3) 실제 업링크 패킷 (CCSDS TC)

구성:

- Primary Header
- APID (Application Process ID)
- Time Tag
- Command Code
- Parameter Field
- CRC

즉, 지상에서 여러 촬영을 **시간태그 기반 명령열(Time-tagged command stack)** 로 만들어 한 번의 패스 동안 업로드합니다.

---

# 2️⃣ 하루 운용 시나리오 예 (LEO 지구관측 위성)

가정:

- 고도 600 km
- 하루 약 14~15회 지구 공전
- 지상국 2곳 운영

## 하루 흐름

### 00:00 UTC

- 고객 요청 수집
- 자동 스케줄링 시스템 실행

### 02:00 UTC

- 전체 촬영계획 생성 (예: 350건)

### 03:10 UTC (첫 패스)

- 다음 2~3 orbit 분량 명령 배치 업링크

### 궤도 중

- 위성은 onboard scheduler로 실행
- 촬영 데이터 저장

### 다음 패스

- 데이터 다운링크
- 필요 시 계획 수정 후 재업로드

핵심:
LEO는 가시창이 짧기 때문에
→ 보통 **다음 몇 orbit 분량을 미리 업로드**

---

# 3️⃣ LEO 군집에서 Crosslink 사용하는 경우

Starlink, 군집 정찰위성 등은 위성 간 ISL (Inter-Satellite Link)을 사용합니다.

## 구조

지상국 → 위성 A 업링크
위성 A → 위성 B/C로 crosslink 전파

이 경우:

- 모든 위성에 개별 업링크 필요 없음
- 긴급명령 전파 속도 향상
- 전지구 커버리지 실시간화

군사 시스템은 이 구조를 선호합니다.

---

# 4️⃣ 완전 자율 위성 (Autonomous Mission Planning)

최근 연구 및 일부 실전 적용 사례.

## 특징

위성이 스스로:

- 구름 감지
- 촬영 성공 여부 판단
- 실패 시 재촬영
- 우선순위 재정렬

### 필요 기술

- Onboard AI
- Edge processing
- 제한적 MILP solver
- Fault detection & recovery (FDIR)

### 장점

- 지상 재개입 최소화
- 지연 시간 감소
- 심우주 탐사에 필수

LEO 상업위성은 아직 **완전 자율보다는 지상 중심 재계획 방식이 일반적**입니다.

---

# 📌 전체 정리

| 단계           | 주체                   |
| -------------- | ---------------------- |
| 촬영요청 수집  | 지상                   |
| 최적 스케줄링  | 지상 자동화 시스템     |
| 배치 업링크    | 지상국                 |
| 실행           | 위성 onboard scheduler |
| 수정/긴급 대응 | 다음 패스에서 재업로드 |
| 군집 최적화    | Crosslink 활용 가능    |

---

좋습니다. 제시한 4개 항목을 **순서대로, 운용 현실 기준으로** 설명하겠습니다.

---

# 1️⃣ Maxar vs Planet 운용 구조 비교

두 회사는 같은 지구관측(EO) 사업자지만 **아키텍처와 스케줄링 철학이 다릅니다.**

## 🔹 Maxar (WorldView 계열)

### 특징

- 소수의 고성능 위성
- 초고해상도 (30cm급)
- 고객 지정 촬영(Tasking 중심)

### 운용 구조

- 고객 주문 기반 촬영
- 고우선순위 타겟 집중
- 정밀 자세기동 필요 (Off-nadir 촬영 많음)
- 하루 수백 건 수준

### 스케줄링

- 고난도 제약 최적화
- 위성 1기당 기동 비용 큼
- 촬영 1건의 단가가 높음 → 개별 최적화 중요

---

## 🔹 Planet (Dove / SkySat)

### 특징

- 대규모 LEO 군집 (수십~수백기)
- 중해상도, 고빈도 촬영
- 지구 전역 반복 촬영

### 운용 구조

- 기본은 전지구 daily revisit
- 고객 주문은 일부 override
- 기동 최소화 설계

### 스케줄링

- 위성 간 분산 로드밸런싱
- 개별 위성보다는 **fleet-level 최적화**
- 자동화 수준 매우 높음

---

## 핵심 차이

| 구분        | Maxar         | Planet          |
| ----------- | ------------- | --------------- |
| 전략        | 고정밀 타겟팅 | 고빈도 커버리지 |
| 위성 수     | 적음          | 많음            |
| 기동        | 적극적        | 최소화          |
| 최적화 단위 | 위성 단위     | 군집 단위       |

---

# 2️⃣ 군용 정찰위성은 어떻게 다른가

군용은 상업과 구조가 다릅니다.

## 주요 차이점

### ① 우선순위

- 실시간 긴급 대응 필수
- P0 임무는 기존 계획 즉시 삭제 가능

### ② 보안

- 암호화 수준 매우 높음
- 다중 인증 체계

### ③ 자율성

- 제한적 onboard replanning
- 데이터 onboard 1차 분석 가능

### ④ 통신

- Crosslink 적극 활용
- 글로벌 분산 지상국망

### ⑤ 촬영 방식

- 표적 추적 (moving target)
- SAR + EO 복합 운용

상업 위성은 “고객 서비스” 중심
군용은 “임무 달성” 중심입니다.

---

# 3️⃣ 위성 1기가 하루에 실제 몇 장 촬영하나

LEO 지구관측 위성 (광학 기준):

## 대략적 수치

- 고해상도 위성: 200~500 scene/day
- 중해상도 군집 위성: 수천 scene/day
- Planet Dove: 하루 지구 전역 촬영

단, “사진 수”는 정의에 따라 다름:

- Strip imaging (선형 촬영)
- Frame 단위
- km² 기준

예시:

- 1회 패스에서 5~20개 타겟
- 하루 14패스 → 70~200 타겟 가능

제약 요소:

- 배터리
- 저장공간
- 자세 기동 시간
- 태양 고도
- 구름

---

# 4️⃣ 전력·자세기동 제약 계산 예시

간단한 현실 모델로 설명합니다.

## 가정

- 고도 600km
- 질량 800kg
- 반작용휠 사용
- 배터리 1.5 kWh

---

## (1) 자세기동 시간

기동 각속도 제한:
예: 0.5 deg/s

Off-nadir 20° 기동 필요하면

20° / 0.5°/s = 40초

가속/감속 포함하면 약 60초 소요

→ 촬영 간 최소 간격 발생

---

## (2) 전력 예

카메라 소비전력: 500W
플랫폼 평균: 300W

촬영 중 총 800W

10분 촬영 시:
800W × (10/60)h = 133 Wh

배터리 1500Wh라면
이론상 10회 이상 가능

하지만:

- Eclipse 구간 존재
- 다운링크 전력 소모
- 열관리 필요

→ 실제 usable budget는 훨씬 낮음

---

# 📌 전체 구조 요약

촬영은 단순히 “찍는다”가 아니라:

1. 궤도상 가시창 계산
2. 태양각 조건 확인
3. 기동 가능 여부 계산
4. 전력/저장 여유 확인
5. 스케줄 큐 삽입
6. 패스 중 배치 업링크
7. 위성 onboard 실행

이 모든 것이 자동화 시스템에서 동작합니다.

---

좋습니다. 요청하신 4개 항목을 **이론 수준이 아니라 실제 운용/엔지니어링 관점에서** 순서대로 설명하겠습니다.

---

# 1️⃣ 실제 MILP 수식 구조 (촬영 스케줄링 문제)

지구관측 촬영 스케줄링은 전형적인 **Mixed Integer Linear Programming (MILP)** 문제입니다.

## 📌 의사결정 변수

[
x_{i,j} =
\begin{cases}
1 & \text{위성 } i \text{가 타겟 } j \text{를 촬영하면} \
0 & \text{아니면}
\end{cases}
]

---

## 📌 목적함수 (예시)

최대 이익 또는 우선순위 가중합 최대화:

[
\max \sum_{i}\sum_{j} P_j \cdot x_{i,j}
]

- (P_j): 타겟 우선순위 점수

---

## 📌 제약 조건

### (1) 가시성 제약

타겟 j는 위성 i의 가시창 내에 있어야 함:

[
x_{i,j} \le V_{i,j}
]

(V\_{i,j} = 1)일 때만 가능

---

### (2) 시간 충돌 제약

두 촬영 간 최소 기동 시간 필요:

[
t_k - t_j \ge T_{slew}
]

---

### (3) 전력 제약

[
\sum P_{op} \cdot x_{i,j} \le P_{available}
]

---

### (4) 저장 용량 제약

[
\sum D_j \cdot x_{i,j} \le M_{available}
]

---

## 현실

- 수백~수천 타겟
- 위성 수십 기
- 제약 수만 개

→ Exact MILP는 계산시간 과다
→ Heuristic + Rolling Horizon 방식 사용

---

# 2️⃣ 궤도역학 기반 촬영 가능 시간 계산

촬영 가능 여부는 단순 위치가 아니라 **Line-of-Sight 기하학** 문제입니다.

---

## (1) 기본 조건

위성 위치 벡터 ( \vec{r_s} )
지상 타겟 벡터 ( \vec{r_t} )

LOS 조건:

[
(\vec{r_t} - \vec{r_s}) \cdot \vec{r_s} > 0
]

즉, 지평선 위에 있어야 함.

---

## (2) Off-nadir 제약

카메라 최대 기울기 각:

[
\theta = \cos^{-1}\left( \frac{(\vec{r_s} \cdot \vec{r_t})}{|r_s||r_t|} \right)
]

[
\theta \le \theta_{max}
]

---

## (3) 태양 고도 조건 (광학 위성)

[
\alpha_{sun} \ge 10^\circ \sim 20^\circ
]

너무 낮으면 그림자 과다
너무 높으면 반사/노출 문제

---

## 결과

촬영 가능 시간은 보통:

- 10~90초 수준
- 위성 고도/경사각에 따라 다름

---

# 3️⃣ SAR 위성은 어떻게 다른가

SAR은 완전히 다른 제약 구조를 가집니다.

---

## 🔹 장점

- 주야간 촬영 가능
- 구름 영향 없음

---

## 🔹 주요 제약

### (1) 전력

SAR 송신기 수 kW급
→ 촬영 횟수 제한

### (2) 안테나 길이 방향

위성 진행 방향으로 일정 시간 비추어야 함
→ Strip 길이 중요

### (3) 데이터량

광학보다 훨씬 큼
→ 저장/다운링크 제약 심각

---

## 스케줄링 차이

광학:

- 점 타겟 최적화

SAR:

- 연속 strip 단위 최적화
- Look angle 매우 중요

---

# 4️⃣ GEO 정찰이 드문 이유

이론적으로 가능하지만 현실적 제약 큼.

---

## ① 해상도 문제

지상 해상도:

[
GSD \propto \frac{H}{D}
]

H: 고도
D: 광학계 구경

GEO는 36,000 km
LEO는 600 km

→ 약 60배 거리 차이
→ 같은 해상도 얻으려면 망원경 직경이 60배 필요

현실적으로 불가능

---

## ② 신호대잡음비 감소

거리²에 비례하여 광량 감소

---

## ③ 대기 왜곡 영향 큼

관측 경로 길어짐

---

## ④ 비용

GEO 대형 광학 시스템은 수십 톤급

---

## 그래서 GEO는

- 기상
- 통신
- 미사일 경보(적외선)

에 적합

고해상도 정찰은 LEO가 유리

---

# 🔎 전체 흐름 요약

촬영 스케줄링은:

1. 궤도 기반 가시창 계산
2. 기하학적 제약 필터링
3. MILP/휴리스틱 최적화
4. 전력·메모리·기동 제약 반영
5. 배치 업링크
6. 온보드 실행

---

좋습니다. 이번에는 **운용·설계 관점에서 실제 엔지니어링 레벨**로 순서대로 설명하겠습니다.

---

# 1️⃣ 실제 수치 넣어서 하루 스케줄 시뮬레이션 예시

## 가정 조건

- 고도: 600 km SSO
- 공전주기: 약 96분
- 하루 약 15 orbit
- 최대 off-nadir: ±25°
- 촬영 1건 평균 20초
- 기동 시간: 평균 60초
- 저장용량: 2 TB
- 다운링크: 하루 총 1.2 TB 가능
- 배터리 usable energy: 1 kWh

---

## ① 한 orbit당 촬영 가능 시간

지상 목표 가시 구간: 평균 8~10분
실제 촬영 가능한 순수 시간: 약 6분

촬영 1건당 필요 시간:

- 기동 60초
- 촬영 20초
- 안정화 20초

≈ 100초

→ 6분(360초) 동안 약 3건

---

## ② 하루 총 촬영 가능 건수

15 orbit × 3건 = 약 45건

하지만:

- 일부 orbit은 바다/비관심지역
- 구름 필터링
- 다운링크 제약

→ 현실적으로 30~40건/day

---

## ③ 데이터량 계산

1건당 8 GB 가정

40건 × 8GB = 320GB

→ 저장/다운링크 충분

---

## ④ 전력 검증

촬영 시 800W
20초 촬영 → 약 4.4 Wh

40건 → 약 176 Wh

→ 전력은 제약이 아님
(실제로는 SAR이 더 빡빡)

---

# 2️⃣ LEO 군집 최적화 알고리즘 구조

군집에서는 “위성 단위”가 아니라 “fleet 단위” 최적화가 핵심입니다.

---

## 단계 구조

### Step 1: Global Task Pool 생성

- 모든 고객 요청 통합

### Step 2: Visibility Matrix 생성

[
V_{i,j,t}
]
i=위성, j=타겟, t=시간

---

### Step 3: 1차 할당 (Assignment)

Hungarian Algorithm 또는 MILP로
각 타겟을 “가장 비용 낮은 위성”에 배정

비용 함수 예:

[
Cost = \alpha \cdot Slew + \beta \cdot Energy + \gamma \cdot CloudRisk
]

---

### Step 4: Local Sequencing

각 위성별 내부 순서 최적화
→ Traveling Salesman Problem 변형

---

### Step 5: Rolling Re-Optimization

매 orbit마다 재계산
(구름/긴급요청 반영)

---

## 핵심

군집은 redundancy가 있기 때문에
“어느 위성이 찍느냐”가 중요

---

# 3️⃣ 구름 예측은 어떻게 반영하나

광학 위성의 가장 큰 불확실성.

---

## ① 외부 기상모델 사용

- ECMWF
- NOAA GFS

격자 단위로 Cloud Probability 획득

---

## ② 스케줄링 반영 방식

타겟 j에 대해:

[
Expected_Value_j = P_j \times (1 - CloudProbability_j)
]

목적함수에 직접 반영

---

## ③ 실전 전략

- 위험 높은 지역은 다중 위성에 중복 배정
- 재방문 가능한 군집 구조 활용
- 촬영 후 quicklook 분석 → 실패 시 재촬영 예약

---

## ④ 고급 시스템

일부는 onboard AI로:

- 촬영 직후 구름 판별
- 자동 discard
- 다음 pass 재시도

---

# 4️⃣ 위성 설계 시 임무 역산 과정

실제 위성 설계는 “찍고 싶은 양”에서 시작합니다.

---

## Step 1: 시장 요구 정의

예:

- 하루 50 scene
- 해상도 0.5m
- revisit 1일

---

## Step 2: 궤도 설계

- SSO 선택
- 고도 결정 (해상도 vs 커버리지 tradeoff)

---

## Step 3: 광학계 직경 결정

GSD 공식:

[
GSD = \frac{H \cdot p}{f}
]

H: 고도
p: 픽셀크기
f: 초점거리

→ 해상도 맞추기 위해 f 계산
→ 구경 D 결정

---

## Step 4: 기동 성능 역산

필요 촬영 건수 →
필요 평균 slew rate 계산 →
반작용휠 토크 요구치 산출

---

## Step 5: 전력 시스템 설계

총 daily energy budget 계산 →
태양전지판 면적 산출

---

## Step 6: 저장/통신 설계

촬영 데이터량 →
SSD 용량 →
X-band 다운링크 속도 →
지상국 수 결정

---

## 핵심 설계 철학

임무 요구사항 → 궤도 → 광학 → ADCS → 전력 → 통신 → 질량 → 발사체

완전히 연쇄적입니다.

---

# 🔎 전체 큰 그림

지상에서 여러 촬영계획을 묶어 업링크하는 행위는
이 거대한 시스템 최적화의 “마지막 단계”일 뿐입니다.

실제 핵심은:

- 궤도기하학
- 최적화 이론
- 에너지 관리
- 기동역학
- 통신 용량
- 기상 리스크

이 모든 것이 통합되어 하루 스케줄이 만들어집니다.

---

알겠습니다. “지상국에서 업링크 커맨드를 보낼 때 여러 촬영계획을 묶어서 보내냐?”에 대해 **운용 흐름대로(차례대로)** 답할게요.

---

## 1) 촬영요청이 여러 건 들어온다

- 고객/임무팀이 타겟(좌표), 촬영모드, 우선순위, 시간 제약 등을 요청으로 생성

## 2) 지상 임무계획 시스템이 “여러 요청”을 스케줄로 만든다

- 가시창(패스), 태양각, 자세기동 시간, 전력/메모리/다운링크 제약을 고려해서
- “오늘/다음 N-orbit 동안 실행할 촬영 타임라인”을 최적화해서 생성

## 3) 그 스케줄을 “명령 시퀀스(커맨드 묶음)”로 컴파일한다

여기서 “묶는다”의 의미는 보통 두 가지 중 하나(또는 혼합)입니다.

- **(A) Time-tagged command list**: 시간표가 붙은 커맨드들을 한 파일/블록으로 묶음
- **(B) Stored sequence / macro**: 위성에 저장된 시퀀스를 호출하거나, 시퀀스 자체를 업로드

## 4) 지상국 패스(가시창) 때 “배치 업링크(Batch Upload)” 한다

- LEO는 패스가 짧아서(수 분) 보통 **한 번 접속할 때 가능한 만큼 ‘여러 건’을 한 번에 올립니다.**
- 즉, **촬영계획 1개씩 쪼개서 매번 올리기보다는, 여러 개를 한 묶음으로 업링크**하는 게 일반적입니다.

## 5) 위성은 온보드에서 저장하고, 스케줄대로 실행한다

- OBC/온보드 스케줄러가 큐에 넣고 시간/조건에 맞춰
  - 자세기동 → 카메라 on → 촬영 → 카메라 off

- 실행 중 리소스 부족/상태 이상이면 스킵하거나 실패로 기록

## 6) 변경/긴급요청이 생기면 “다음 패스에서 재업로드(부분 업데이트)” 한다

- 새로 들어온 긴급 촬영이 있으면
  - 낮은 우선순위 일부를 drop 하고
  - **수정된 묶음(패치/새 시퀀스)** 을 다음 지상국 패스에서 다시 올리는 방식이 흔합니다.

---

### 결론(질문에 대한 한 줄 답)

**네, 보통 여러 촬영계획을 “시간태그된 명령 시퀀스/배치”로 묶어서 업링크합니다.** 특히 LEO는 패스가 짧아서 배치 업링크가 표준 운영에 가깝습니다.

---

원하시는 “묶는 단위”가 **(1) 촬영 N건을 하나의 파일로? (2) 하나의 TC 패킷에 N건을? (3) 패스당 N건을?** 중 어떤 의미인지에 따라 더 정확히 파고들 수 있어요.
지금 질문은 보통 (3)+(1) 의미로 이해하는 게 가장 자연스럽습니다.

좋습니다. 질문을 명확히 재정의하면:

> “지상에서 최종적으로 위성에 업로드되는 **Telecommand의 파라미터 구조**는 실제로 어떤 필드들로 구성되는가?”

아래는 **실제 지구관측 위성(광학 기준)**에서 일반적으로 사용되는 커맨드 파라미터 구조를, CCSDS 기반 TC 프레임 관점과 임무 레벨 관점으로 나누어 정리한 사례 조사입니다.

---

# 1️⃣ 최상위 구조: CCSDS Telecommand 프레임

대부분 위성은 **CCSDS TC Packet 구조**를 따릅니다.

## 기본 프레임 구조

### (1) Primary Header

- Version Number
- Packet Type (TC=1)
- APID (Application Process ID)
- Sequence Flags
- Packet Sequence Count
- Packet Length

### (2) Secondary Header (미션 정의)

- Time Tag (UTC / MET)
- Command ID
- Source ID
- Acknowledgment Flags

### (3) Application Data Field ← 핵심

여기에 실제 “촬영 커맨드 파라미터”가 들어갑니다.

### (4) Error Control

- CRC
- 암호화 MAC 등

---

# 2️⃣ 실제 촬영 커맨드 파라미터 구성 사례 (광학 EO 위성)

촬영은 보통 단일 “Shoot” 커맨드가 아니라 **Attitude + Payload + Data Handling**의 조합입니다.

---

## 📌 (A) Slew / Pointing Command 파라미터

| 파라미터           | 설명                   | 예시         |
| ------------------ | ---------------------- | ------------ |
| Target_Latitude    | 목표 위도              | 37.5665°     |
| Target_Longitude   | 목표 경도              | 126.9780°    |
| Target_Altitude    | 목표 고도              | 0 m          |
| Pointing_Mode      | Earth-fixed / inertial | Earth-fixed  |
| Off_Nadir_Angle    | 카메라 기울기          | 18°          |
| Slew_Start_Time    | 기동 시작 시각         | 03:14:55 UTC |
| Max_Slew_Rate      | 제한 각속도            | 0.5°/s       |
| Quaternion / Euler | 목표 자세 벡터         | q0,q1,q2,q3  |

고급 위성은 좌표 대신 **목표 자세 Quaternion**을 직접 넣기도 합니다.

---

## 📌 (B) Imaging Payload Command 파라미터

| 파라미터         | 설명               | 예시           |
| ---------------- | ------------------ | -------------- |
| Imaging_Mode     | Pan / MS / SWIR 등 | Panchromatic   |
| Integration_Time | 노출 시간          | 0.8 ms         |
| Gain             | 센서 게인          | Level 3        |
| Binning_Mode     | 픽셀 binning       | 2x2            |
| Line_Count       | 촬영 라인 수       | 12000          |
| Strip_Length     | km 단위            | 25 km          |
| Compression_Mode | Lossless/Lossy     | CCSDS Lossless |
| Image_ID         | 촬영 식별자        | 20260310-001   |

---

## 📌 (C) Data Handling 파라미터

| 파라미터          | 설명              |
| ----------------- | ----------------- |
| Storage_Partition | 저장 위치         |
| Downlink_Priority | 전송 우선순위     |
| Encryption_Key_ID | 데이터 암호 키    |
| Auto_Delete_Flag  | 전송 후 삭제 여부 |

---

# 3️⃣ Time-Tagged Command Sequence 구조

실제 운영에서는 “단일 촬영 커맨드”보다는 아래처럼 묶여 있습니다.

```
[Time: T1] Slew_Start (Quaternion Q1)
[Time: T2] Fine_Pointing_Enable
[Time: T3] Payload_Power_On
[Time: T4] Imaging_Start (Mode=Pan, Exposure=0.8ms)
[Time: T5] Imaging_Stop
[Time: T6] Payload_Power_Off
[Time: T7] Return_To_Nadir
```

이 전체 블록이 하나의 “Stored Command Sequence”로 업로드되기도 합니다.

---

# 4️⃣ SAR 위성의 촬영 커맨드 파라미터 예시

SAR은 구조가 다릅니다.

| 파라미터                         | 설명                 |
| -------------------------------- | -------------------- |
| Radar_Mode                       | Stripmap / Spotlight |
| Pulse_Repetition_Frequency (PRF) |                      |
| Transmit_Power                   |                      |
| Chirp_Bandwidth                  |                      |
| Incidence_Angle                  |                      |
| Look_Direction (Left/Right)      |                      |
| Swath_Width                      |                      |
| Polarization (HH, HV 등)         |                      |

SAR은 kW급 송신기 제어 때문에 전력/열 파라미터가 매우 중요합니다.

---

# 5️⃣ 실제 산업 문헌에서 확인되는 공통 특징

CNES, ESA, NASA 문서 기준 공통점:

1. 명령은 APID별로 분리
2. 모든 커맨드는 Time-tag 가능
3. 대부분의 수치 파라미터는 스케일링된 정수 (예: milli-degree 단위)
4. 좌표 대신 onboard 계산 효율을 위해 자세 벡터 직접 전달하는 경우 많음
5. 커맨드에는 항상 Execution Verification Mode 포함
   - Ack on acceptance
   - Ack on start
   - Ack on completion

---

# 6️⃣ 파라미터 수준 요약 (광학 EO 기준)

최종적으로 위성에 업로드되는 촬영 관련 커맨드는 대략 다음 4개 계층 파라미터로 구성됩니다:

① 시간 정보
② 자세 목표 정보 (좌표 또는 quaternion)
③ 센서 설정값 (노출, 모드, 압축 등)
④ 데이터 관리 설정

---

# 7️⃣ 중요한 현실적 포인트

- 실제로는 좌표 하나만 보내는 단순 구조가 아님
- 대부분은 “시퀀스 형태의 다중 TC 패킷”
- 일부 상용 위성은 “High-level Observation Command” 1개로 보내고
  → 위성 내부에서 세부 커맨드로 분해

---

# 🔎 결론

최종 업로드 커맨드는 보통 다음 필드 그룹으로 구성됩니다:

- Time tag
- Command ID
- Target pointing parameters
- Payload configuration parameters
- Resource/handling flags
- Execution acknowledgment settings
- Error control

즉, “촬영해라”는 단일 명령이 아니라
**자세·센서·데이터 관리까지 포함된 구조화된 파라미터 세트**입니다.

---

좋습니다. 요청하신 4가지를 **차례대로, 구조적으로** 설명하겠습니다.

---

# 1️⃣ 실제 CCSDS 패킷 바이너리 필드 예시

아래는 **CCSDS Telecommand (TC) Space Packet**의 표준 구조입니다.

## 📌 (1) Primary Header (6 bytes)

| 필드            | 비트수 | 설명                       |
| --------------- | ------ | -------------------------- |
| Version         | 3      | 보통 000                   |
| Type            | 1      | 1 = TC                     |
| Sec Header Flag | 1      | Secondary header 존재 여부 |
| APID            | 11     | 명령 대상 서브시스템 ID    |
| Seq Flags       | 2      | 단일/시퀀스                |
| Seq Count       | 14     | 패킷 카운터                |
| Packet Length   | 16     | 데이터 길이                |

---

## 📌 (2) Secondary Header (Mission Defined)

예시 구조:

| 필드         | 크기     | 설명                |
| ------------ | -------- | ------------------- |
| Time Tag     | 4~8 byte | 실행 시간           |
| Command Code | 1~2 byte | 명령 식별자         |
| Ack Flags    | 1 byte   | 수신/시작/완료 확인 |

---

## 📌 (3) Application Data Field

예: Imaging_Start 커맨드

| 필드        | 타입   | 예          |
| ----------- | ------ | ----------- |
| Mode        | uint8  | 0x01 (Pan)  |
| Exposure    | uint16 | 800 (0.8ms) |
| Gain        | uint8  | 3           |
| Line Count  | uint16 | 12000       |
| Compression | uint8  | 0x02        |

---

## 📌 (4) Error Control

- CRC-16 또는 CRC-32
- 암호화 MAC

---

### 실제 전송은

```
[Primary Header][Secondary Header][App Data][CRC]
```

이진 프레임으로 RF 업링크됩니다.

---

# 2️⃣ 소형위성(6U/12U)의 단순화된 커맨드 구조

CubeSat 계열은 복잡성이 훨씬 낮습니다.

## 특징

- CCSDS 미사용 경우도 있음
- AX.25, CSP, 자체 프로토콜 사용
- High-level command 위주

---

## 예시: 단일 촬영 커맨드

```id="k9x3dp"
CMD_ID = 0x21
PARAM_1 = Latitude (float32)
PARAM_2 = Longitude (float32)
PARAM_3 = Duration (uint16)
```

즉,

“좌표 + 촬영시간”만 보내고
자세계산은 onboard에서 수행.

---

## Stored Sequence 기능 없음인 경우

- 실시간 명령 기반 운용
- 지상국 패스마다 소량 업로드
- 자율성 낮음

---

# 3️⃣ Stored Sequence vs Real-Time Command 차이

## 🔹 Stored Sequence

지상에서 여러 커맨드를 묶어 업로드

예:

```
SEQ_ID 12:
T+0  Slew
T+60 Camera_On
T+80 Camera_Off
T+90 Store_Data
```

### 장점

- 통신 끊겨도 실행 가능
- LEO에 적합
- 안정적

### 단점

- 유연성 낮음
- 긴급 변경 시 재업로드 필요

---

## 🔹 Real-Time Command

패스 중 즉시 명령

예:

- “지금 촬영 시작”
- “중지”

### 장점

- 유연성 높음
- 테스트에 유리

### 단점

- LEO에는 패스 짧아 비효율

---

## 현실

상업 EO 위성 → Stored Sequence 기반
시험/초기 위성 → Real-time 위주

---

# 4️⃣ 군용 정찰위성의 보안 필드 구조

공개자료 기준 일반적 구조 설명입니다.

---

## 추가 보안 계층

### ① Command Authentication Field

- HMAC-SHA256
- AES-GCM Tag

### ② Rolling Key ID

- 키 교체 관리

### ③ Anti-Replay Counter

- Sequence Number + Timestamp

### ④ Dual-Authorization Flag

- 특정 명령은 2인 승인 필요

---

## 암호화 구조

```
Encrypted Payload {
   Time Tag
   Command Code
   Parameters
}
+ Authentication Tag
```

RF 계층에서도 링크 암호화 추가 가능.

---

# 🔎 전체 정리

실제 위성 업로드 커맨드는 다음 레벨로 나뉩니다:

1. 전송 프레임 구조 (CCSDS 등)
2. 임무 명령 파라미터 구조
3. 실행 방식 (Stored vs Real-time)
4. 보안/인증 계층

---

# 📌 핵심 결론

최종 업로드 커맨드는 단순 좌표 몇 개가 아니라:

- 시간 정보
- 자세 목표
- 센서 설정
- 데이터 처리 정책
- 실행 확인 플래그
- 보안 인증 정보

까지 포함된 **계층적 구조의 바이너리 패킷 세트**입니다.

---

더 깊게 들어가면:

- APID 분할 구조 실제 사례
- 위성 내부 소프트웨어 아키텍처 (OBC task 구조)
- Time-tag 처리 메커니즘
- Onboard fault 발생 시 커맨드 무효화 로직

좋습니다. 요청하신 4가지를 **차례대로, 시스템 내부 동작 관점에서** 설명하겠습니다.

---

# 1️⃣ APID 분할 구조 실제 사례

APID (Application Process ID)는 **어떤 서브시스템이 이 명령을 처리할지**를 구분하는 식별자입니다.

## 📌 목적

- 명령 라우팅
- 오류 격리
- 기능 모듈화

---

## 📌 예시 APID 구성 (지구관측 위성 가정)

| APID | 대상 서브시스템 |
| ---- | --------------- |
| 0x01 | OBC (공통 명령) |
| 0x10 | ADCS (자세제어) |
| 0x20 | Payload Camera  |
| 0x30 | Mass Memory     |
| 0x40 | Telemetry/Comms |
| 0x50 | Power Subsystem |

---

## 📌 촬영 시 실제 흐름

촬영 하나를 수행하려면:

1. ADCS APID로 Slew 명령
2. Payload APID로 Camera On
3. Memory APID로 저장 설정

즉, 하나의 촬영은 **여러 APID 패킷들의 조합**입니다.

---

## 📌 Stored Sequence에서의 APID 처리

Stored Sequence 내부에 여러 APID 명령이 포함될 수 있습니다.

예:

```text
[APID 0x10] Slew to Q1
[APID 0x20] Start Imaging
[APID 0x30] Allocate Memory
```

위성 내부 라우터가 APID 기준으로 분배합니다.

---

# 2️⃣ 위성 내부 소프트웨어 아키텍처 (OBC Task 구조)

대부분 RTOS 기반 구조입니다 (VxWorks, RTEMS 등).

---

## 📌 일반적 구조

```
OBC
 ├── Telecommand Handler Task
 ├── Time Manager Task
 ├── ADCS Control Task
 ├── Payload Control Task
 ├── Data Handling Task
 ├── Power Management Task
 └── FDIR Task
```

---

## 📌 Telecommand 처리 흐름

1. RF 수신
2. Frame decode
3. CRC 검사
4. 암호 인증
5. APID 분류
6. 각 Task 큐에 삽입

---

## 📌 Task 간 통신

- Message Queue
- Shared Memory
- Event Flag

---

## 📌 우선순위

RTOS에서는:

- FDIR > ADCS > Payload > Housekeeping

즉, 안전이 항상 최우선입니다.

---

# 3️⃣ Time-Tag 처리 메커니즘

Time-tag 명령은 즉시 실행이 아닙니다.

---

## 📌 Time-tag 처리 흐름

1. TC 수신
2. Secondary header에서 실행시간 추출
3. Time-tag Queue에 저장
4. System clock와 비교
5. 실행 시점 도달 시 해당 APID task로 전달

---

## 📌 시간 기준

- UTC 기반
- 또는 MET (Mission Elapsed Time)

---

## 📌 정밀도

- ms ~ 수십 ms 수준
- GPS 동기화 또는 onboard oscillator

---

## 📌 문제 상황

- 위성 시간 drift
- Time-tag 과거/미래 오류
- Clock reset

→ Time Manager가 검증

---

# 4️⃣ Onboard Fault 발생 시 커맨드 무효화 로직

위성은 항상 안전 우선입니다.

---

## 📌 FDIR (Fault Detection, Isolation, Recovery)

### 주요 감시 대상

- 배터리 전압
- 반작용휠 속도
- 과열
- 자세 안정성
- 메모리 오류

---

## 📌 Fault 발생 시 처리 단계

### 1단계: 경고

- 현재 실행 명령 일시 중단

### 2단계: Abort

- 현재 촬영 중지
- 카메라 Off
- Safe Attitude 복귀

### 3단계: Safe Mode 진입

- Payload 전원 차단
- 태양 방향 고정

---

## 📌 Time-tag 명령 처리

Fault 상태일 경우:

- Pending 명령 무효화
- 또는 Hold 상태 유지
- 일부는 재실행 금지 플래그 설정

---

## 📌 예시

반작용휠 과속 시:

- 추가 Slew 명령 거부
- 기존 촬영 스케줄 skip

---

# 🔎 전체 시스템 흐름 요약

지상에서 업로드된 커맨드는:

1. APID로 분류
2. RTOS Task로 전달
3. Time-tag 큐에서 대기
4. 실행 시 리소스/상태 검증
5. Fault 발생 시 FDIR 개입
6. 필요 시 Safe Mode 전환

---

# 📌 핵심 이해 포인트

위성은 “명령을 그대로 실행하는 장치”가 아니라

- 시간관리 시스템
- 리소스 검증 시스템
- 안전 감시 시스템
- 실시간 제어 시스템

이 통합된 **임베디드 분산 제어 시스템**입니다.

---

더 깊게 들어가면:

- Reaction wheel saturation 수학적 처리
- Onboard autonomy level 분류
- CCSDS PUS 표준과 ECSS 구조
- 위성 1기 개발 시 소프트웨어 V&V 구조

좋습니다. 요청하신 4가지를 **차례대로, 엔지니어링 수준으로** 설명하겠습니다.

---

# 1️⃣ Reaction Wheel Saturation 처리

## 📌 문제 정의

반작용휠은 각운동량을 저장하여 자세를 제어합니다.

[
H_{wheel} = I_w \cdot \omega_w
]

- ( I_w ): 휠 관성모멘트
- ( \omega_w ): 휠 속도

외란 토크(공기저항, 중력구배 등)가 지속되면 휠 속도가 점점 증가합니다.

[
H_{total} = \sum H_{wheel}
]

휠 속도가 최대 RPM에 도달하면 → **Saturation 발생**

---

## 📌 Saturation 위험

- 추가 기동 불가
- 자세 불안정
- 제어 루프 붕괴 가능

---

## 📌 해결 방법: Momentum Dumping

### (1) 자기토커(Magnetorquer) 사용 (LEO)

지구 자기장과 상호작용:

[
\vec{T} = \vec{m} \times \vec{B}
]

- ( \vec{m} ): 자기 모멘트
- ( \vec{B} ): 지자기장

휠 속도를 줄이면서 각운동량을 외부로 방출

---

### (2) Thruster 사용 (고급 위성)

미세 추력기로 각운동량 제거

---

## 📌 운용 전략

- Slew 전 saturation margin 계산
- 특정 RPM 초과 시 자동 dump
- 촬영 스케줄에 dump 시간 반영

---

# 2️⃣ Onboard Autonomy Level 분류

위성 자율성은 보통 4단계로 구분합니다.

---

## 🔹 Level 0 – 완전 지상 제어

- 모든 촬영 지상 결정
- 위성은 실행만

## 🔹 Level 1 – 제한적 자율

- Fault 자동 처리
- 기본 리소스 검증

## 🔹 Level 2 – 임무 재계획 가능

- 촬영 실패 시 재시도
- 우선순위 부분 재정렬

## 🔹 Level 3 – 고급 자율

- 구름 분석 후 촬영 취소
- 표적 자동 탐지
- 자체 스케줄 수정

상업 EO 위성은 대체로 Level 1~2 수준
심우주 탐사는 Level 3 이상 필요

---

# 3️⃣ CCSDS PUS 표준과 ECSS 구조

유럽계 위성은 CCSDS 위에 **PUS (Packet Utilization Standard)**를 사용합니다.

---

## 📌 PUS 개념

CCSDS는 “전송 포맷”
PUS는 “서비스 의미 정의”

---

## 주요 PUS 서비스 타입

| Service Type | 기능                     |
| ------------ | ------------------------ |
| 1            | Telecommand Verification |
| 3            | Housekeeping             |
| 5            | Event Reporting          |
| 9            | Time Management          |
| 11           | Onboard Scheduling       |
| 17           | Test                     |

예: 촬영 스케줄은 Service 11 사용 가능

---

## ECSS (유럽 표준)

- 소프트웨어 아키텍처 규격
- FDIR 요구사항 정의
- 안전성/검증 체계 포함

---

# 4️⃣ 위성 소프트웨어 V&V 구조 (검증·검증)

위성 SW는 항공우주 수준 품질 요구

---

## 📌 단계

### (1) Unit Test

모듈 단위 시험

### (2) Integration Test

Task 간 통합 시험

### (3) Hardware-in-the-Loop (HIL)

실제 ADCS 센서/액추에이터 연결

### (4) FlatSat Test

비행모델과 동일 구성 테스트

### (5) Thermal Vacuum Test

우주 환경 모사

---

## 📌 독립 검증 (IV&V)

- 독립 팀이 코드 리뷰
- 요구사항 추적성 관리 (Traceability Matrix)

---

## 📌 FDIR 검증

- Fault Injection
- 센서 오작동 시나리오 시험
- Watchdog 리셋 시험

---

# 🔎 전체 통합 관점

지상에서 묶어 업로드된 촬영 명령은:

1. APID로 분배
2. RTOS Task 실행
3. Time-tag 스케줄 관리
4. ADCS가 기동 수행
5. Reaction wheel saturation 관리
6. FDIR가 안전 보장
7. 모든 SW는 V&V 완료 상태

즉, 단순 명령 전달이 아니라
**항공우주급 실시간 분산 제어 시스템**입니다.

---

더 깊게 가면:

- 실제 Reaction wheel 각운동량 수치 예제 계산
- Slew maneuver 시간 최적화 수식
- Safe Mode 진입 후 복귀 절차
- 위성 간 Crosslink 기반 분산 스케줄링 구조

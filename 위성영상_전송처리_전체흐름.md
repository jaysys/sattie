<br><br>
위성이 촬영한 영상이 지상에서 최종 이미지 파일(예: GeoTIFF, NITF 등)로 생성되기까지의 전체 흐름을 탑재체 → 통신 → 지상국 → 영상처리 → 파일 생성 순으로 계층적으로 정리
<br><br>

---

## 1. 위성 탑재체 내부 (On-Board Segment)

## 1.1. 영상 획득 (Image Acquisition)

- 광학계 + FPA(Focal Plane Array, CCD/CMOS)가 지표면을 촬영
- 각 픽셀은 **아날로그 전압 신호** 생성

```
광자 → 센서 → 전하 → 전압 신호
```

---

## 1.2. 아날로그 → 디지털 변환 (A/D Conversion)

- ADC에서 디지털 값으로 변환
- 예: 12bit / 14bit / 16bit DN (Digital Number)

```
Pixel DN = 0 ~ 4095 (12bit 예시)
```

이 단계에서 이미 “디지털 영상 데이터”가 됨.

---

## 1.3. 온보드 영상 처리 (선택적)

위성 설계에 따라 다음 처리가 수행될 수 있음:

- Line reordering
- Radiometric correction (기본 보정)
- Compression (JPEG2000, CCSDS Image Compression 등)
- Packetization

---

## 1.4. 패킷화 (CCSDS Packetization)

영상 데이터는 그대로 보내지 않음.

다음 구조로 변환됨:

```
[Source Packet]
 ↓
[TM Transfer Frame]
 ↓
[CADU]
```

### 1.4.1. Source Packet

- 영상 데이터 + APID
- CCSDS Packet

### 1.4.2. TM Transfer Frame

- 일정 길이 프레임화
- 순번, 상태정보 포함

### 1.4.3. CADU (Channel Access Data Unit)

구조:

```
| ASM | TM Frame | RS Parity |
```

- ASM → 동기
- RS → 오류정정

---

## 2. 위성 → 지상 전송 (Space Link)

## 2.1. 채널 코딩 & 변조

- Reed-Solomon
- Convolutional / LDPC
- Interleaving

변조 방식 예:

- QPSK
- OQPSK
- 8PSK
- X-band downlink

```
CADU → 변조 → RF 신호
```

---

## 2.2. RF Downlink

- 안테나 통해 지상국으로 송신
- X-band (영상), S-band (Telemetry) 사용

---

## 3. 지상국 수신 (Ground Station Front-End)

## 3.1. RF 수신

지상국 구성:

```
Tracking Antenna
 ↓
LNA
 ↓
Down Converter
 ↓
Demodulator
```

- 반송파 복조
- 비트스트림 복원

---

## 3.2. 채널 복호 (Channel Decoding)

복조 후 처리:

- De-interleaving
- LDPC/Convolutional decoding
- Reed-Solomon decoding
- 오류 정정

출력:

```
복구된 CADU
```

---

## 4. 지상 데이터 처리 (Ground Data Processing)

## 4.1. 프레임 동기 및 CADU 분해

```
CADU → TM Frame → Source Packet
```

- ASM 탐색
- Frame 순서 재정렬
- 누락 여부 체크

---

## 4.2. 영상 데이터 재조립

- Packet sequence 기반 재조립
- Line 단위 정렬
- Strip 구성

---

## 4.3. 압축 해제 (필요 시)

온보드 압축이 있었다면:

- JPEG2000 decoding
- CCSDS image decompression

---

## 5. 영상 처리 (Level 0 → Level 1 → Level 2)

이 단계가 “이미지 파일 생성”의 핵심.

---

## 5.1. Level 0 (Raw)

- 센서 DN 값 그대로
- 기하보정 없음
- 시간 순 데이터

---

## 5.2. Level 1 (Radiometric / Geometric Correction)

### 5.2.1. 수행 작업:

1. Dark signal 제거
2. Gain 보정
3. Line alignment
4. Geometric correction
5. RPC 생성

출력:

- Geo-referenced image

---

## 5.3. Level 2 (정사보정)

- DEM 사용
- Ortho-rectification
- 지도 좌표계 보정

출력:

- Ortho Image

---

## 6. 최종 이미지 파일 생성

이제 실제 파일 포맷으로 생성됨.

대표 포맷:

| 포맷     | 특징     |
| -------- | -------- |
| GeoTIFF  | GIS 표준 |
| NITF     | 군사용   |
| JPEG2000 | 압축     |
| HDF5     | 다중밴드 |

파일 내부 포함 정보:

- 픽셀 데이터
- 좌표계
- RPC
- 촬영 시간
- 위성 자세 정보

---

## 7. 전체 흐름 요약 다이어그램

```
[위성 촬영]
 ↓
ADC
 ↓
온보드 처리
 ↓
Source Packet
 ↓
TM Frame
 ↓
CADU
 ↓
변조/RF 송신
 ↓
지상국 수신
 ↓
복조/복호
 ↓
CADU 분해
 ↓
Packet 재조립
 ↓
영상 복원
 ↓
보정(Level1/2)
 ↓
GeoTIFF 등 파일 생성
```

---

## 8. 핵심 포인트 정리

| 구분        | 핵심                   |
| ----------- | ---------------------- |
| 우주 구간   | CADU 단위 전송         |
| 지상 수신   | 오류정정 + 프레임 복원 |
| 데이터 처리 | Packet 재조립          |
| 영상 처리   | 보정 및 정사영상 생성  |
| 최종 산출물 | GeoTIFF 등 파일        |

---

<br><br>
위성 영상 생성 체계는 CADU 기반 전송과 지상 복원·보정을 거쳐 최종 산출물(GeoTIFF/NITF)로 완성된다.
<br><br>

---

## 1. CCSDS 계층 구조 상세 설명

CCSDS는 우주 통신 표준 프로토콜 스택입니다.
OSI 모델과 유사하게 계층 구조를 가집니다.

## 1.1. Physical Layer

- RF 송수신
- 변조/복조 (QPSK, OQPSK, 8PSK 등)
- 주파수 대역 (S-band, X-band)

출력: **Raw bitstream**

---

## 1.2. Channel Coding Layer

전송 오류 보정 목적.

- Convolutional / Turbo / LDPC
- Reed-Solomon (RS)
- Interleaving

출력: 오류정정된 CADU

---

## 1.3. Synchronization & Channel Coding Sublayer

### 1.3.1. CADU 구조:

```
| ASM | TM Transfer Frame | RS Parity |
```

- ASM: 동기 마커 (0x1ACFFC1D)
- RS: 오류정정 코드

---

## 1.4. Data Link Layer

### 1.4.1. TM Transfer Frame

- Master Channel ID
- Virtual Channel ID
- Frame Counter
- Data Field

프레임 단위 전송 관리.

---

## 1.5. Space Packet Layer

### 1.5.1. Source Packet

- APID
- Sequence Count
- Length
- Payload (영상 데이터)

여기서부터 “실제 데이터” 의미가 명확해짐.

---

## 1.6. Application Layer

- 영상 데이터
- Telemetry
- 명령 데이터

---

### 1.6.1. 계층 흐름 요약

```
Application Data
 ↓
Source Packet
 ↓
TM Frame
 ↓
CADU
 ↓
Channel Coding
 ↓
Modulation
 ↓
RF
```

---

## 2. 위성 영상 Level 정의 (0 ~ 4)

위성 영상은 처리 단계에 따라 Level이 구분됩니다.

---

## 2.1. Level 0 (Raw Data)

- 센서 DN 값
- 보정 없음
- 압축만 해제된 상태
- 지리정보 없음

엔지니어링/재처리용

---

## 2.2. Level 1A

- Radiometric calibration 적용
- 센서 왜곡 일부 보정
- Still geometry distortion 존재

---

## 2.3. Level 1B

- 기하 보정 수행
- 센서 모델 기반 좌표 계산
- RPC 생성

일반 분석 시작 가능

---

## 2.4. Level 2 (Ortho)

- DEM 적용
- 지형 왜곡 제거
- 지도 좌표계에 정확히 정렬

GIS 사용 가능

---

## 2.5. Level 3 / 4 (고급 산출물)

- Mosaic
- Time-series stack
- NDVI 등 지표 산출
- 분석 제품

---

## 3. LEO 실시간 수신 vs 저장 후 다운로드 차이

LEO 위성은 가시 시간(pass)이 짧습니다 (약 5~12분).

## 3.1. 실시간 Direct Downlink

촬영과 동시에 지상으로 송신

### 3.1.1. 특징:

- 대용량 실시간 링크 필요
- 가시권 내에서만 가능
- 지상국 네트워크 필요

### 3.1.2. 장점:

- 빠른 데이터 획득
- 재난 대응 유리

### 3.1.3. 단점:

- 가시 시간 제한
- 데이터 손실 위험

---

## 3.2. On-board Recording 후 Dump

위성 내부 Solid State Recorder (SSR)에 저장 후
지상국 통과 시 다운로드

### 3.2.1. 특징:

- Store & Forward 방식
- 고속 Burst 전송

### 3.2.2. 장점:

- 전세계 촬영 가능
- 가시권 의존도 낮음

### 3.2.3. 단점:

- 지연 발생
- 저장 용량 제한

---

## 3.3. 비교

| 항목          | 실시간 | 저장 후 다운로드 |
| ------------- | ------ | ---------------- |
| 지연          | 낮음   | 있음             |
| 링크 요구     | 지속적 | 통과 시만        |
| 시스템 복잡도 | 낮음   | 높음             |

실제 운용은 대부분 **혼합 방식** 사용.

---

## 4. 영상 데이터 레이트 계산 예시

실제 시스템 설계 시 가장 중요한 계산입니다.

---

## 4.1. 가정

- 해상도: 1 m GSD
- Swath width: 20 km
- Bit depth: 12 bit
- Spectral bands: 4 band
- Line rate: 10,000 line/sec

---

## 4.2. 한 라인 픽셀 수

```
20 km / 1 m = 20,000 pixel
```

---

## 4.3. 1 line 데이터량

```
20,000 pixel × 12 bit × 4 band
= 960,000 bit
≈ 0.96 Mbit
```

---

## 4.4. 초당 데이터율

```
0.96 Mbit × 10,000 line/sec
= 9,600 Mbit/sec
= 9.6 Gbps
```

---

## 4.5. 압축 적용 (예: 4:1)

```
9.6 Gbps / 4 = 2.4 Gbps
```

---

## 4.6. 채널 코딩 오버헤드 (약 20%)

```
2.4 Gbps × 1.2 ≈ 2.88 Gbps
```

실제 RF Downlink 요구 속도 ≈ **3 Gbps**

---

## 5. 최종 핵심 정리

1. **CCSDS는 계층형 구조이며 CADU는 채널 계층 단위**
2. **영상 Level은 보정 정도에 따라 구분**
3. **LEO는 실시간/저장 후 다운로드 혼합 운용**
4. **고해상도 영상은 수 Gbps급 링크 필요**

---

<br><br>
영상 레벨 체계와 수신 운용 방식의 차이를 바탕으로 실제 링크 요구 데이터율을 산정할 수 있다.
<br><br>

---

## 1. X-band 링크 예산 (Link Budget) 계산 개요

링크 예산은 **위성에서 송신한 신호가 지상 수신기에 도달했을 때 SNR이 충분한지**를 계산하는 절차입니다.

## 1.1. 기본 방정식

수신 전력:

[
P_r = P_t + G_t + G_r - L_s - L_{atm} - L_{other}
]

(dB 단위)

---

## 1.2. 주요 항목

### 1.2.1. 송신 전력 (P_t)

- 예: 20 W → 43 dBm

### 1.2.2. 송신 안테나 이득 (G_t)

- 예: 35 dBi

### 1.2.3. 경로 손실 (L_s) (Free Space Path Loss)

[
L_s = 92.45 + 20\log f(GHz) + 20\log R(km)
]

예시:

- f = 8 GHz (X-band)
- R = 700 km

→ 약 **165 ~ 170 dB**

---

### 1.2.4. 수신 안테나 이득 (G_r)

- 5 m 안테나 → 약 50 dBi

---

## 1.3. 수신 전력 계산 예시

[
P_r = 43 + 35 + 50 - 168 - 2
]

≈ -42 dBm

---

## 1.4. Noise Power

[
N = kTB
]

dB 형태:

[
N(dBm) = -228.6 + 10\log T + 10\log B
]

예:

- T = 500 K
- B = 1 GHz

→ 약 -82 dBm

---

## 1.5. SNR

[
SNR = P_r - N
]

≈ 40 dB

→ 실제로는 시스템 손실 포함해 더 낮아짐.

---

### 1.5.1. 목표

SNR이 변조 방식 + 코딩 방식이 요구하는 값 이상이어야 함.

---

## 2. LDPC 적용 시 Eb/N0 요구치

LDPC는 고성능 Forward Error Correction(FEC) 코드입니다.

---

## 2.1. Eb/N0 정의

[
Eb/N0 = \frac{SNR}{Spectral Efficiency}
]

단위: dB

---

## 2.2. 변조별 요구 Eb/N0 (대략값)

| 변조   | LDPC 적용 시 필요 Eb/N0 |
| ------ | ----------------------- |
| QPSK   | 1 ~ 2 dB                |
| 8PSK   | 4 ~ 6 dB                |
| 16APSK | 7 ~ 9 dB                |

---

## 2.3. 왜 LDPC가 중요한가?

과거:

- RS + Convolutional → Eb/N0 ≈ 5~6 dB 필요

현재:

- LDPC → 1~2 dB 근접 (Shannon limit 근접)

동일 전력으로 더 높은 데이터율 가능
또는 같은 데이터율에서 송신 전력 감소 가능

---

## 2.4. 시스템 설계 흐름

1. 목표 데이터율 설정
2. 변조 방식 선택
3. 코딩률 선택 (예: 1/2, 3/4)
4. 요구 Eb/N0 계산
5. Link Budget로 달성 가능 여부 판단

---

## 3. SSR (Solid State Recorder) 용량 산정

LEO 위성은 촬영 데이터를 내부 저장 후 dump합니다.

---

## 3.1. 기본 공식

[
필요용량 = 데이터율 × 촬영시간
]

---

## 3.2. 예시

데이터율 = 2.5 Gbps
촬영 시간 = 600초 (10분)

[
2.5 × 10^9 × 600
= 1.5 × 10^{12} bit
]

= 1.5 Tb
≈ 187.5 GB

---

## 3.3. 하루 촬영 5회라면?

[
187.5 × 5 = 937.5 GB
]

→ 최소 1 TB 이상 필요

---

## 3.4. 실제 설계 시 고려사항

- ECC redundancy
- Memory wear leveling
- Radiation tolerance
- Dump window 길이
- Downlink rate > Imaging rate 필요

---

## 3.5. 설계 조건

[
Downlink rate × 가시시간 ≥ 저장된 총 데이터
]

이 조건 만족 못하면 데이터 backlog 발생

---

## 4. 상용 위성 사례 비교

## 4.1. PlanetScope

- 해상도: 3~5 m
- 데이터율 낮음
- S-band/X-band 혼용
- 다수 소형 위성

전략: 다수 위성 + 낮은 해상도

---

## 4.2. WorldView-3

- 해상도: 0.3 m
- 데이터율 매우 높음
- X-band 고속 downlink
- 대형 안테나 사용

전략: 고해상도 + 고속 링크

---

## 4.3. ICEYE (SAR)

- X-band SAR
- 매우 높은 데이터율
- On-board processing 후 일부 데이터만 전송

전략: 원시데이터 일부 압축/처리

---

## 4.4. 최신 추세

- Optical inter-satellite link (Laser)
- 10 Gbps 이상
- 위성 → 중계위성 → 지상

---

## 5. 종합 구조

```text
Imaging Data Rate
 ↓
SSR 저장
 ↓
LDPC 코딩
 ↓
X-band Downlink
 ↓
지상국 수신
 ↓
Eb/N0 만족 여부 판단
 ↓
파일 생성
```

---

## 6. 핵심 요약

1. Link Budget는 SNR 확보 계산
2. LDPC는 Eb/N0를 크게 낮춰주는 핵심 기술
3. SSR 용량은 촬영량과 dump 능력의 함수
4. 상용 위성은 해상도와 링크 전략이 다름

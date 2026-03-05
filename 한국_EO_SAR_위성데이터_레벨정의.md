**대한민국에서 운영·서비스되는 EO/SAR 위성 데이터의 Level 정의(L0~L4)**를
국제 공통(Level 개념) + 한국 위성(특히 _KOMPSAT-5 SAR_) 사례를 기반으로 정리

---

## 1) 위성 데이터 Level 체계 (국제 공통 정의)

먼저 EO/SAR 모두에 적용되는 **일반적인 Level 개념**입니다.
이 정의는 CEOS 및 NASA 데이터를 기준으로 합니다: ([Cal/Val Portal][1])

| Level | 의미 | |
| ------ | ----------------------------------------------------------- | --------------------- |
| **L0** | 원시 데이터 재구성 (Full resolution, 통신 아티팩트 제거) | |
| **L1** | 기하/방사 보정 정보가 계산·첨부된 상태 (보정 적용은 선택적) | |
| **L2** | 물리/지구 물리 변수로 변환된 데이터 | |
| **L3** | 공간/시간 정규화 및 재샘플링된 시계열 또는 모자이크 | |
| **L4** | 분석 또는 모델 결과로부터 파생된 제품(지표정보, 인덱스 등) | ([Cal/Val Portal][1]) |

> 이 개념은 EO, SAR, 대기, 해양 등 모든 위성 관측 플랫폼에서 기본적으로 적용됩니다. ([Cal/Val Portal][1])

---

## 2) 한국 **SAR 위성(KOMPSAT-5)**의 Level 사례

한국의 대표적인 SAR 위성 **KOMPSAT-5 (Arirang-5)** 자료를 기준으로 한 실제 Level 정의입니다. ([geodata.kr][2])

### ▪ **L0 – Raw SAR Signal Data**

- SAR 레이더 _에코 신호(소스 복소수 형태)_
- 레이더 펄스 순서대로 수신된 복소수 데이터
- 노이즈 및 내부 검보정 신호 포함
 SAR 이미징 알고리즘 전단계의 원시 신호 데이터입니다. ([geodata.kr][2])

---

### ▪ **L1A – Focused Complex Image**

- L0 신호를 Range/Azimuth로 *압축 처리*한 신호
- 복소수 형태 유지(Amplitude + Phase)
- SAR 좌표계(Range-Doppler) 기반
 SAR Interferometry나 고급 분석에 유용한 초기 영상 데이터입니다. ([geodata.kr][2])

---

### ▪ **L1C – Projected Intensity Image (Topo Projection)**

- L1A에서 **Amplitude(강도)만 추출**
- 지구 타원체 모델 기준으로 투영
- DEM 없이도 기복 효과를 최소화한 영상
 분석·시각화 목적의 SAR 초기 제품입니다. ([geodata.kr][2])

---

### ▪ **L1D – Geocoded Intensity Image**

- L1C에서 **정확한 지리정보 적용 (Geocoding)**
- DEM 적용으로 기하 왜곡 보정
- GIS 및 지도 연동 가능
 일반 사용자/분석가가 가장 많이 사용하는 SAR 영상 제품입니다. ([geodata.kr][2])

---

## 3) 한국 **EO 위성** Level 체계 (일반 EO 정의)

한국 EO 위성(예: **KOMPSAT-3, KOMPSAT-3A** 등)도 국제 EO Level 정의를 따릅니다.

### EO Level 설명 (일반적 정의)

| Level | 처리 상태 | |
| ------- | -------------------------------------------------------------------- | --------------------- |
| **L0** | 통신/전송 복원된 원시 디지털 센서 데이터 (Reconstructed unprocessed) | |
| **L1A** | 보정 파라미터 첨부, 보정 적용 전 | |
| **L1B** | Radiometric/Geometric calibration 적용 | |
| **L2** | 지형 교정/지리 공간 정렬된 영상 | |
| **L3** | 모자이크/타일/재샘플링 제품 | |
| **L4** | 지표 변수, 분류/지표지수 산출물 | ([Cal/Val Portal][1]) |

 KOMPSAT-3 EO 영상은 L1B/L2 제품으로 사용되는 경우가 가장 많습니다(GeoTIFF 포맷 등). ([eoportal.org][3])

---

## 4) EO vs SAR Level 비교 (한국 위성 기준)

| Level | EO 위성 (예: KOMPSAT-3) | SAR 위성 (KOMPSAT-5) |
| ------ | --------------------------------------- | -------------------------------------- |
| **L0** | 원시 디지털 영상 데이터 | SAR Raw Echo Signal |
| **L1** | Radiometric/Geometric 체크 및 보정 정보 | SAR Focused/Complex Image |
| **L2** | Geo-corrected orthorectified image | SAR Projected Intensity |
| **L3** | Mosaic/temporal composite | SAR Geocoded Intensity + tiling |
| **L4** | Vegetation index, land cover map | SAR analysis products (change maps 등) |

_SAR은 신호 복소수/위상 정보가 포함되므로 EO와 동일 Level 명칭이지만 처리 의미가 달라집니다._ ([geodata.kr][2])

---

## 5) 한국 위성 Level 자료 저장 형식 (예)

| Level | Typical Format | Notes |
| ----- | ----------------------------------- | -------------- |
| L0 | Raw binary/CEOS, Complex IQ streams | 원시 신호/영상 |
| L1 | GeoTIFF, CEOS L1A archival | 보정 정보 포함 |
| L2 | GeoTIFF with RPC/DEM | 지형/지리 정렬 |
| L3 | Tiled GeoTIFF, Mosaic | 공개 지도용 |
| L4 | Classified raster, index maps | 분석 목적 |

위 표는 국제 EO/SAR 관행과 한국 사례를 반영한 일반적인 데이터 구조입니다. ([Cal/Val Portal][1])

---

## 요약

- **L0**: 복원된 원시 데이터 (EO는 DN, SAR은 Echo Complex)
- **L1**: 보정 파라미터 포함 또는 처리된 초기 산출물
- **L2**: 지형·지리 정렬 적용된 과학/영상 데이터
- **L3**: 공간/시간 통합된 모자이크/정규화 제품
- **L4**: 분석·모델 기반 파생 제품

한국 SAR(예: KOMPSAT-5)에서는 SAR 특성을 고려한 **L1A/1C/1D** 세분 레벨 체계가 사용됩니다. ([geodata.kr][2])

---

[1]: https://calvalportal.ceos.org/cal/val-wiki/-/wiki/CEOS_Terms_and_Definitions/EO%2Bdata%2Bproducts%2B%28L0%2C%20L1%2C%20L2%2C%20L3%2C%20L4%29%20across%2Bagencies%2Band%2Bcompanies/pop_up?utm_source=chatgpt.com "EO data products (L0, L1, L2, L3, L4) across agencies and companies"
[2]: https://www.geodata.kr/journal/view.php?number=73&viewtype=pubreader&utm_source=chatgpt.com "다목적실용위성 5호 영상 데이터 제공 및 품질관리 - GEO DATA"
[3]: https://www.eoportal.org/satellite-missions/kompsat-3?utm_source=chatgpt.com "KOMPSAT-3 (Korea Multi-Purpose Satellite-3) / Arirang-3"

# API 응답 명세 부록

## 문서 정보
- 최종 갱신일자: `03/14 15:46`

## 1. 문서 목적
- 마크다운 뷰어 웹서비스의 API 응답 형식을 구현 가능한 수준으로 고정한다.
- `2-설계자/마크다운_뷰어_웹서비스_설계서.md`의 API 설계를 상세 명세로 보완한다.

## 2. 공통 응답 원칙

### 2.1 성공 응답
- JSON API는 기본적으로 `ok: true`를 포함한다.
- 목록 응답은 `items`, `count`를 포함한다.
- 상세 응답은 `document` 또는 의미가 분명한 단수 필드를 사용한다.

### 2.2 실패 응답
```json
{
  "ok": false,
  "error_code": "DOC_NOT_FOUND",
  "message": "Document not found"
}
```

### 2.3 공통 필드 규칙
- `slug`
  - 외부 공개 식별자
- `source_path`
  - 내부 원본 파일 경로
- `public_url`
  - 브라우저 접근 URL
- `updated_at`
  - ISO 8601 UTC 문자열

## 3. 공통 객체 스키마

### 3.1 DocumentSummary
```json
{
  "slug": "system-design",
  "title": "SATTIE 시스템 설계 문서",
  "summary": "서비스 구조와 API 계층을 설명하는 설계 문서",
  "category": "architecture",
  "tags": ["fastapi", "api", "system-design"],
  "public_url": "/p/system-design",
  "updated_at": "2025-03-14T00:00:00Z",
  "word_count": 3200,
  "reading_minutes": 13
}
```

### 3.2 DocumentHeading
```json
{
  "level": 2,
  "text": "시스템 목표와 범위",
  "anchor": "system-goals"
}
```

### 3.3 DocumentDetail
```json
{
  "slug": "system-design",
  "title": "SATTIE 시스템 설계 문서",
  "summary": "서비스 구조와 API 계층을 설명하는 설계 문서",
  "source_path": "docs/SYSTEM_DESIGN.md",
  "public_url": "/p/system-design",
  "category": "architecture",
  "tags": ["fastapi", "api", "system-design"],
  "is_public": true,
  "updated_at": "2025-03-14T00:00:00Z",
  "word_count": 3200,
  "reading_minutes": 13,
  "headings": [
    {
      "level": 2,
      "text": "시스템 목표와 범위",
      "anchor": "system-goals"
    }
  ],
  "html": "<h1>...</h1>",
  "raw_markdown_url": "/api/docs/system-design/raw"
}
```

## 4. API 명세

### 4.1 `GET /api/docs/list`
- 목적
  - 문서 허브 목록 조회
- query
  - `category`: 카테고리 필터
  - `tag`: 태그 필터
  - `sort`: `latest | popular | title`
  - `page`: 기본 `1`
  - `page_size`: 기본 `20`

#### 성공 응답
```json
{
  "ok": true,
  "count": 2,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "slug": "system-design",
      "title": "SATTIE 시스템 설계 문서",
      "summary": "서비스 구조와 API 계층을 설명하는 설계 문서",
      "category": "architecture",
      "tags": ["fastapi", "api", "system-design"],
      "public_url": "/p/system-design",
      "updated_at": "2025-03-14T00:00:00Z",
      "word_count": 3200,
      "reading_minutes": 13
    }
  ]
}
```

#### 실패 응답
  - `400 INVALID_SORT`
- `500 INDEX_READ_FAILED`

#### 정렬 규칙
- `latest`
  - `updated_at` 내림차순
- `title`
  - 제목 오름차순
- `popular`
  - MVP에서는 조회수 집계가 없으므로 `order` 오름차순, 동률이면 `updated_at` 내림차순으로 처리한다.

### 4.2 `GET /api/docs/search?q=`
- 목적
  - 제목, 요약, 본문 일부 기반 검색
- query
  - `q`: 검색어
  - `category`: 선택 필터
  - `tag`: 선택 필터
  - `page`
  - `page_size`

#### 성공 응답
```json
{
  "ok": true,
  "query": "fastapi",
  "count": 1,
  "items": [
    {
      "slug": "system-design",
      "title": "SATTIE 시스템 설계 문서",
      "summary": "서비스 구조와 API 계층을 설명하는 설계 문서",
      "category": "architecture",
      "tags": ["fastapi", "api", "system-design"],
      "public_url": "/p/system-design",
      "updated_at": "2025-03-14T00:00:00Z",
      "word_count": 3200,
      "reading_minutes": 13,
      "score": 24.5,
      "matched_fields": ["title", "tags", "summary"]
    }
  ]
}
```

#### 실패 응답
- `400 EMPTY_QUERY`
- `500 SEARCH_INDEX_FAILED`

### 4.3 `GET /api/docs/{slug}`
- 목적
  - 문서 상세 조회

#### 성공 응답
```json
{
  "ok": true,
  "document": {
    "slug": "system-design",
    "title": "SATTIE 시스템 설계 문서",
    "summary": "서비스 구조와 API 계층을 설명하는 설계 문서",
    "source_path": "docs/SYSTEM_DESIGN.md",
    "public_url": "/p/system-design",
    "category": "architecture",
    "tags": ["fastapi", "api", "system-design"],
    "is_public": true,
    "updated_at": "2025-03-14T00:00:00Z",
    "word_count": 3200,
    "reading_minutes": 13,
    "headings": [
      {
        "level": 2,
        "text": "시스템 목표와 범위",
        "anchor": "system-goals"
      }
    ],
    "html": "<h1>...</h1>",
    "raw_markdown_url": "/api/docs/system-design/raw"
  },
  "related_documents": [
    {
      "slug": "maintainer-guide",
      "title": "유지보수 가이드",
      "summary": "운영과 장애대응을 위한 문서",
      "category": "operations",
      "tags": ["guide"],
      "public_url": "/p/maintainer-guide",
      "updated_at": "2025-03-14T00:00:00Z",
      "word_count": 1800,
      "reading_minutes": 8
    }
  ]
}
```

#### 실패 응답
- `404 DOC_NOT_FOUND`
- `500 RENDER_FAILED`

### 4.4 `GET /api/docs/{slug}/raw`
- 목적
  - 원본 Markdown 다운로드 또는 원문 조회

#### 성공 응답
- `text/markdown; charset=utf-8`
- 본문은 원본 Markdown 그대로 반환

#### 실패 응답
- `404 DOC_NOT_FOUND`

### 4.5 `POST /api/docs/upload`
- 목적
  - Markdown 파일 업로드 후 저장 및 재색인
- body
```json
{
  "filename": "system-design.md",
  "content": "# title\n\nbody"
}
```

#### 성공 응답
```json
{
  "ok": true,
  "file_name": "system-design.md",
  "size_bytes": 1280,
  "slug": "system-design",
  "public_url": "/p/system-design",
  "reindexed": true
}
```

#### 실패 응답
- `400 INVALID_UPLOAD_TYPE`
- `400 INVALID_FILENAME`
- `413 FILE_TOO_LARGE`
- `500 INDEX_BUILD_FAILED`

#### 검증 규칙
- `filename`은 `.md`로 끝나야 한다.
- 파일명은 sanitize 후 저장하며 경로 구분자와 상대경로 패턴은 허용하지 않는다.
- 본문은 UTF-8 문자열이어야 한다.

### 4.6 `POST /api/docs/reindex`
- 목적
  - 전체 인덱스 재생성

#### 성공 응답
```json
{
  "ok": true,
  "documents_count": 12,
  "reindexed_at": "2025-03-14T00:00:00Z"
}
```

#### 실패 응답
- `500 INDEX_BUILD_FAILED`

## 5. HTTP 상태코드 기준
- `200`
  - 정상 조회, 검색, 재색인
- `201`
  - 신규 업로드가 생성됨
- `400`
  - 잘못된 요청 파라미터
- `404`
  - 문서 또는 slug 없음
- `413`
  - 업로드 용량 초과
- `500`
  - 렌더링 실패, 인덱스 생성 실패

## 6. 정렬 및 검색 점수 규칙
- 제목 일치: 가장 높은 가중치
- 태그 일치: 높음
- summary 일치: 중간
- heading 일치: 중간
- 본문 일치: 기본
- 최신 문서는 동점일 때 우선 노출

## 7. 구현 메모
- HTML 페이지 라우트와 JSON API는 응답 형식을 섞지 않는다.
- 상세 API의 `html`은 서버 렌더 결과를 그대로 사용한다.
- 관련 문서 추천은 초기에는 동일 카테고리 우선 방식으로 단순화한다.
- `reading_minutes`는 `word_count / 250` 기준 올림으로 계산한다.

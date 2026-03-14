#!/usr/bin/env python3
"""
Sattie (virtual) - FastAPI service for serving satellite imagery images.

Run:
  uvicorn app.sattie_api:app --reload --host 0.0.0.0 --port 6001
"""

from __future__ import annotations

import html
import os
import re
import unicodedata
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.core as core_module
from app.doc_service import DocumentService
from app.md_viwer import render_guide_page
from app.core import *  # noqa: F403

app = FastAPI(
    title="Sattie (Virtual Satellite Imagery Service)",
    description=(
        "EO/SAR 위성영상 샘플 데이터를 L0~L4 레벨로 제공하는 Mock API입니다.\n\n"
        "주요 목적:\n"
        "- 이미지 목록/상세 조회 및 파일 다운로드\n"
        "- 브라우저 미리보기를 위한 콘텐츠 엔드포인트 제공\n"
        "- OpenStreetMap 타일 기반 위성촬영 모사(L0~L4) 생성\n"
        "- mock_store 샘플 데이터 생성/삭제/상태 확인\n\n"
        "OSM 모사 API 빠른 시작:\n"
        "1) 생성: `POST /osm/images/generate?lat=37.5665&lon=126.9780&zoom=14&sensor=eo`\n"
        "2) 결과조회: `GET /osm/images/{request_id}`\n"
        "3) 다운로드: `GET /osm/images/{request_id}/{level}/download`\n"
        "4) 미리보기: `GET /osm/images/{request_id}/{level}/content`\n\n"
        "OSM 레벨 해석:\n"
        "- L0: OSM 원본 타일 바이트 + 메타를 포장한 CEOS 유사 raw bin\n"
        "- L1: 기초 보정 영상(GeoTIFF)\n"
        "- L2: 정사 보정 영상(GeoTIFF)\n"
        "- L3: 타일/모자이크 서비스 영상(Tiled GeoTIFF + tile 파생 파일)\n"
        "- L4: EO=index-map, SAR=classified-raster 성격의 분석 산출물\n\n"
        "참고:\n"
        "- L0~L4는 동일 소스 장면을 기반으로 포맷/가공 수준만 달라집니다.\n"
        "- OSM 타일은 네트워크 상태/외부 서버 응답에 영향을 받습니다.\n"
        "- 샘플 전체 삭제 후 재생성 시 새로운 랜덤 시드가 적용되어 다른 장면이 만들어집니다."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "service", "description": "서비스 상태 점검 및 기본 헬스체크 API"},
        {"name": "images", "description": "이미지 목록/상세 조회, 원본 다운로드, 미리보기 콘텐츠 API"},
        {"name": "mock-store", "description": "mock_store 샘플 생성/삭제/상태 관리 API"},
        {
            "name": "osm-images",
            "description": (
                "OpenStreetMap 타일을 이용해 위성촬영 파이프라인(L0~L4)을 모사하는 API. "
                "요청별 request_id를 발급하며, 레벨별 다운로드/미리보기 URL을 제공합니다."
            ),
        },
        {"name": "ui", "description": "브라우저용 관리/탐색 UI API"},
        {"name": "guide", "description": "guide 폴더의 Markdown 문서를 HTML로 탐색/렌더링하는 API"},
        {"name": "docs-hub", "description": "문서 허브, 문서 상세, 업로드, 운영자 문서 관리 API"},
    ],
)
app.mount("/ui/static", StaticFiles(directory=str(UI_HTML_PATH.parent), html=False), name="ui-static")

CATALOG: list[ImageItem] = []
OSM_SIM_CATALOG: dict[str, list[ImageItem]] = {}
OSM_SIM_SOURCE: dict[str, dict] = {}
core_module.OSM_SIM_CATALOG = OSM_SIM_CATALOG
core_module.OSM_SIM_SOURCE = OSM_SIM_SOURCE
GUIDE_DIR = PROJECT_DIR / "guide"
DOC_SERVICE = DocumentService(PROJECT_DIR)


class GuideUploadPayload(BaseModel):
    filename: str
    content: str


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/hub", "/upload", "/admin/docs", "/ui", "/docs", "/redoc", "/openapi.json", "/guide"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def startup_event() -> None:
    global CATALOG
    CATALOG = _load_catalog()
    _ensure_osm_catalog_loaded()
    DOC_SERVICE.reindex()


def _is_local_client(request: Request) -> bool:
    client_host = (request.client.host if request.client else "") or ""
    return client_host in {"127.0.0.1", "::1", "localhost"}


def _require_docs_admin(request: Request) -> None:
    token = os.getenv("DOCS_ADMIN_TOKEN", "").strip()
    if token:
        supplied = request.headers.get("x-docs-admin-token", "").strip()
        if supplied != token:
            raise HTTPException(status_code=403, detail={"ok": False, "error_code": "FORBIDDEN", "message": "Admin token required"})
        return
    if not _is_local_client(request):
        raise HTTPException(status_code=403, detail={"ok": False, "error_code": "FORBIDDEN", "message": "Admin route is restricted to local access"})


def _docs_json_response(action):
    try:
        return action()
    except HTTPException as exc:
        if isinstance(exc.detail, dict):
            return JSONResponse(content=exc.detail, status_code=exc.status_code)
        raise


def _guide_markdown_files() -> list[Path]:
    if not GUIDE_DIR.exists():
        return []
    files = [p for p in GUIDE_DIR.rglob("*.md") if p.is_file()]

    def norm_rel(p: Path) -> str:
        return unicodedata.normalize("NFC", p.relative_to(GUIDE_DIR).as_posix())

    return sorted(
        files,
        key=lambda p: (
            0 if norm_rel(p).lower() == "readme.md" else 1,
            norm_rel(p).lower(),
        ),
    )


def _sanitize_md_filename(name: str) -> str:
    base = unicodedata.normalize("NFC", Path(name).name).strip()
    if not base:
        raise HTTPException(status_code=400, detail="Filename is empty")
    if not base.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md upload is allowed")
    safe = re.sub(r"[^A-Za-z0-9._\-\u3131-\u318E\uAC00-\uD7A3 ]+", "_", base)
    safe = safe.strip()
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe


def _resolve_guide_file(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        raise HTTPException(status_code=400, detail="Absolute path is not allowed")
    candidate = (GUIDE_DIR / p).resolve()
    guide_root = GUIDE_DIR.resolve()
    if guide_root != candidate and guide_root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed")
    if candidate.suffix.lower() != ".md":
        raise HTTPException(status_code=400, detail="Only .md file is allowed")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Guide file not found")
    return candidate


def _guide_upload_widget_html() -> str:
    return (
        "<div style='margin:12px 0;'>"
        "<input id='guideFileInput' type='file' accept='.md,text/markdown' style='display:none;' "
        "onchange=\"(function(){var el=document.getElementById('guideFileInput');var n=document.getElementById('guideFileName');if(n){n.textContent=(el&&el.files&&el.files[0])?el.files[0].name:'선택된 파일 없음';}})()\" /> "
        "<button type='button' onclick=\"document.getElementById('guideFileInput').click()\">추가할 파일 선택</button> "
        "<span id='guideFileName' style='margin-left:8px;color:#555;'>선택된 파일 없음</span> "
        "<button type='button' onclick='uploadGuideMd()'>.md 업로드</button> "
        "<span id='guideUploadMsg' style='margin-left:8px;color:#555;'></span>"
        "</div>"
        "<script>"
        "async function uploadGuideMd(){"
        "  const el=document.getElementById('guideFileInput');"
        "  const msg=document.getElementById('guideUploadMsg');"
        "  if(!el||!el.files||!el.files.length){ if(msg) msg.textContent='파일을 선택하세요.'; return; }"
        "  const f=el.files[0];"
        "  const n=(f.name||'').toLowerCase();"
        "  if(!n.endsWith('.md')){ if(msg) msg.textContent='.md 파일만 업로드 가능합니다.'; return; }"
        "  const text=await f.text();"
        "  if(msg) msg.textContent='업로드 중...';"
        "  const res=await fetch('/guide/upload',{"
        "    method:'POST',"
        "    headers:{'Content-Type':'application/json'},"
        "    body:JSON.stringify({filename:f.name, content:text})"
        "  });"
        "  if(!res.ok){"
        "    let detail='업로드 실패';"
        "    try{ const j=await res.json(); detail=j.detail||detail; }catch(e){}"
        "    if(detail==='업로드 실패'){ detail='업로드 실패 ('+res.status+')'; }"
        "    if(msg) msg.textContent=String(detail);"
        "    return;"
        "  }"
        "  const j=await res.json();"
        "  if(msg) msg.textContent='업로드 완료';"
        "  window.location.reload();"
        "}"
        "</script>"
    )


@app.get(
    "/health",
    tags=["service"],
    summary="서비스 상태 확인",
    description="API 서버 기동 상태와 현재 메모리에 로드된 샘플 이미지 개수를 반환합니다.",
    response_description="서비스 상태 정보(status/service/images)",
)
def health() -> dict:
    return {"status": "ok", "service": "sattie", "images": len(CATALOG)}


@app.get(
    "/images",
    tags=["images"],
    summary="샘플 이미지 목록 조회",
    description=(
        "샘플 이미지 목록을 조회합니다.\n\n"
        "필터:\n"
        "- sensor: eo 또는 sar\n"
        "- level: L0~L4\n"
        "- fmt: 포맷 문자열 정확 일치\n"
        "- q: image_id/summary/satellite 부분 검색(대소문자 무시)"
    ),
    response_description="필터 결과 개수(count)와 이미지 배열(items)",
)
def list_images(
    sensor: Optional[str] = Query(
        default=None,
        pattern="^(eo|sar)$",
        description="센서 필터(eo 또는 sar). 미지정 시 전체.",
        examples=["eo", "sar"],
    ),
    level: Optional[str] = Query(
        default=None,
        pattern="^L[0-4]$",
        description="처리 레벨 필터(L0~L4). 미지정 시 전체.",
        examples=["L0", "L2", "L4"],
    ),
    fmt: Optional[str] = Query(
        default=None,
        description="파일 포맷 정확 일치 필터(예: geotiff, ceos, tiled-geotiff).",
        examples=["geotiff", "ceos"],
    ),
    q: Optional[str] = Query(
        default=None,
        description="image_id, summary, satellite에서 부분 일치 검색(대소문자 무시).",
        examples=["kompsat", "mosaic"],
    ),
) -> dict:
    items = CATALOG
    if sensor:
        items = [x for x in items if x.sensor == sensor]
    if level:
        items = [x for x in items if x.level == level]
    if fmt:
        items = [x for x in items if x.fmt == fmt]
    if q:
        qq = q.lower()
        items = [
            x
            for x in items
            if qq in x.image_id.lower() or qq in x.summary.lower() or qq in x.satellite.lower()
        ]

    return {
        "count": len(items),
        "items": [asdict(x) for x in items],
    }


@app.post(
    "/images/generate",
    tags=["images"],
    summary="파라미터 기반 샘플 이미지 즉시 생성",
    description=(
        "요청한 `sensor`, `level`, `fmt` 조합으로 샘플 파일을 즉시 생성한 뒤 파일 본문을 바로 응답합니다.\n\n"
        "동작 방식:\n"
        "- 요청마다 새로운 랜덤 시드를 사용해 파일 내용을 생성합니다.\n"
        "- 생성된 파일은 `mock_store/random/generated/{sensor}/{level}` 경로에 저장됩니다.\n"
        "- 응답은 `Content-Disposition: attachment`로 내려가며, 호출 즉시 다운로드할 수 있습니다.\n\n"
        "포맷 규칙:\n"
        "- EO: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=index-map\n"
        "- SAR: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=classified-raster\n"
        "- 위 조합과 맞지 않으면 `400 Bad Request`를 반환합니다.\n"
        "- 확장자 매핑: ceos=.bin, 그 외 지원 포맷=.tif\n\n"
        "호출 예시:\n"
        "- `POST /images/generate?sensor=eo&level=L0&fmt=ceos`\n"
        "- `POST /images/generate?sensor=eo&level=L2&fmt=geotiff`\n"
        "- `POST /images/generate?sensor=eo&level=L3&fmt=tiled-geotiff`\n"
        "- `POST /images/generate?sensor=sar&level=L4&fmt=classified-raster`\n"
        "- `POST /images/generate?sensor=eo&level=L4&fmt=index-map`\n\n"
        "참고:\n"
        "- 본 API는 카탈로그(`CATALOG`)에 항목을 추가하지 않습니다.\n"
        "- 저장된 생성 파일 목록은 `/admin/mock-store/info`에서 확인할 수 있습니다."
    ),
    response_description="생성된 샘플 파일(binary stream)",
    responses={
        400: {"description": "fmt 미지원 또는 sensor/level/fmt 조합 불일치"},
    },
)
def generate_image(
    sensor: str = Query(
        ...,
        pattern="^(eo|sar)$",
        description="생성할 센서 타입(eo 또는 sar).",
        examples=["eo", "sar"],
    ),
    level: str = Query(
        ...,
        pattern="^L[0-4]$",
        description="생성할 처리 레벨(L0~L4).",
        examples=["L0", "L2", "L4"],
    ),
    fmt: str = Query(
        ...,
        description=(
            "생성 포맷. sensor/level 매핑 규칙을 따라야 합니다. "
            "EO: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=index-map / "
            "SAR: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=classified-raster"
        ),
        examples=["geotiff", "ceos", "classified-raster"],
    ),
) -> FileResponse:
    fmt_norm = _validate_generate_combo(sensor=sensor, level=level, fmt=fmt)
    suffix = _generated_suffix_for_format(fmt_norm)
    seed = random.SystemRandom().randrange(1, 2**31)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = random.SystemRandom().randrange(1000, 10000)
    filename = f"{sensor}_{level}_{fmt_norm.replace('-', '_')}_{ts}_{rid}{suffix}"

    out_dir = RANDOM_GENERATED_DIR / sensor / level
    out_path = out_dir / filename
    out_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".tif":
        _make_dummy_image(out_path, width=512, height=512, seed=seed, sensor=sensor)
        media_type = "image/tiff"
    else:
        _make_dummy_bin(out_path, size_bytes=1024 * 1024, seed=seed)
        media_type = "application/octet-stream"

    return FileResponse(
        path=out_path,
        media_type=media_type,
        filename=filename,
    )


@app.post(
    "/osm/images/generate-all",
    tags=["osm-images"],
    summary="OSM 타일 기반 L0~L4 일괄 생성",
    description=(
        "요청한 `lat`, `lon`, `zoom`, `sensor` 조합으로 OSM 기반 L0~L4 샘플을 즉시 생성한 뒤 "
        "`request_id`와 결과 메타데이터를 바로 응답합니다.\n\n"
        "동작 방식:\n"
        "- 좌표/줌으로 OSM 타일 x/y를 계산합니다.\n"
        "- `https://tile.openstreetmap.org/{z}/{x}/{y}.png`에서 타일을 다운로드합니다.\n"
        "- 타일 기반 seed로 L0~L4 파일을 생성합니다.\n"
        "- 생성된 파일은 `mock_store/osm/{request_id}` 경로에 저장됩니다.\n"
        "- 응답은 `request_id`, `source`, `items`(JSON)로 반환됩니다.\n\n"
        "- `level`/`fmt`를 함께 지정하면 해당 단일 조합만 생성합니다.\n"
        "- `level`/`fmt`를 함께 지정하면 `/images/generate`와 동일하게 파일 본문을 `Content-Disposition: attachment`로 즉시 응답합니다.\n"
        "- `level`/`fmt`를 생략하면 L0~L4 전체를 생성하고 JSON 메타데이터를 응답합니다.\n\n"
        "포맷 규칙:\n"
        "- EO: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=index-map\n"
        "- SAR: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=classified-raster\n"
        "- SAR L1~L4는 흑백(강도/분류 톤)으로 모사됩니다.\n"
        "- 확장자 매핑: ceos=.bin, 그 외 지원 포맷=.tif\n\n"
        "예외 상황:\n"
        "- `level` 또는 `fmt` 중 하나만 지정하면 `400 Bad Request`를 반환합니다.\n"
        "- `sensor/level/fmt` 조합이 규칙과 다르면 `400 Bad Request`를 반환합니다.\n"
        "- `lat/lon/zoom` 범위가 잘못되면 `422 Unprocessable Entity`를 반환합니다.\n"
        "- OSM 타일 다운로드 실패 시 `502 Bad Gateway`를 반환합니다.\n\n"
        "호출 예시:\n"
        "- `POST /osm/images/generate?lat=37.5665&lon=126.9780&zoom=14&sensor=eo`\n"
        "- `POST /osm/images/generate?lat=35.1796&lon=129.0756&zoom=13&sensor=sar`\n"
        "- `POST /osm/images/generate?lat=37.5665&lon=126.9780&zoom=14&sensor=eo&level=L2&fmt=geotiff`\n\n"
        "참고:\n"
        "- 본 API는 OSM 요청 카탈로그(request_id 단위)에 결과를 유지합니다.\n"
        "- 생성된 결과 조회: `/osm/images`, `/osm/images/{request_id}`, `/osm/images/items`"
    ),
    response_description="request_id와 레벨별 파일 메타데이터/다운로드 URL",
    responses={
        200: {"description": "생성 성공(JSON 메타데이터 또는 파일 본문 응답)"},
        400: {"description": "level/fmt 누락(한쪽만 지정) 또는 sensor/level/fmt 조합 불일치"},
        422: {"description": "lat/lon/zoom 파라미터 검증 실패"},
        502: {"description": "OSM 타일 수신 실패(네트워크/외부 응답 오류)"},
    },
)
def generate_osm_images(
    lat: float = Query(..., ge=-85.05112878, le=85.05112878, description="중심 위도(WGS84)", examples=[37.5665]),
    lon: float = Query(..., ge=-180.0, le=180.0, description="중심 경도(WGS84)", examples=[126.978]),
    zoom: int = Query(14, ge=0, le=19, description="OSM 줌 레벨(0=전지구, 숫자 클수록 상세)", examples=[14]),
    sensor: str = Query("eo", pattern="^(eo|sar)$", description="모사 센서 타입(eo/sar)", examples=["eo", "sar"]),
    level: Optional[str] = Query(None, pattern="^L[0-4]$", description="생성 레벨 지정(선택). fmt와 함께 지정"),
    fmt: Optional[str] = Query(
        None,
        description=(
            "생성 포맷 지정(선택). `level`과 함께 지정해야 합니다. "
            "허용 조합: EO(L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=index-map), "
            "SAR(L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=classified-raster)."
        ),
        examples=["ceos", "geotiff", "tiled-geotiff", "index-map", "classified-raster"],
    ),
) -> dict:
    sensor_norm = sensor.lower()
    single_mode = (level is not None) or (fmt is not None)
    selected_fmt: Optional[str] = None
    if single_mode:
        if not level or not fmt:
            raise HTTPException(status_code=400, detail="Both level and fmt are required when selecting single level generation")
        selected_fmt = _validate_generate_combo(sensor=sensor_norm, level=level, fmt=fmt)

    tile = _fetch_osm_tile(lon=lon, lat=lat, zoom=zoom)
    tile_hash = hashlib.sha256(tile["bytes"]).hexdigest()

    ts = datetime.now(timezone.utc)
    ts_text = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    req_id = f"osm-{sensor_norm}-{ts.strftime('%Y%m%dT%H%M%SZ')}-{random.SystemRandom().randrange(1000, 10000)}"
    out_dir = OSM_STORE_DIR / req_id
    out_dir.mkdir(parents=True, exist_ok=True)

    l0_meta = {
        "source": "openstreetmap",
        "tile_url": tile["url"],
        "tile_x": tile["x"],
        "tile_y": tile["y"],
        "zoom": zoom,
        "lat": lat,
        "lon": lon,
        "sensor": sensor_norm,
        "captured_at_utc": ts_text,
    }

    level_defs: list[tuple[str, str, str]] = [
        ("L0", "ceos", "원시 패킷(CEOS 유사)"),
        ("L1", "geotiff", "기초 보정 영상"),
        ("L2", "geotiff", "정사 보정 영상"),
        ("L3", "tiled-geotiff", "타일/모자이크 서비스 영상"),
        ("L4", "index-map" if sensor_norm == "eo" else "classified-raster", "분석 산출물"),
    ]
    level_notes = {
        "L0": "원시 타일 바이트와 메타를 포장한 CEOS 유사 단계",
        "L1": "OSM 장면 원본을 TIFF로 표현한 기준 단계",
        "L2": "L1에 명암/감마 보정을 적용한 개선 단계",
        "L3": "인접 타일을 결합한 2x2 모자이크 단계",
        "L4": "주제도(분류) 형태로 재표현한 분석 단계",
    }

    tile_rgb = _png_bytes_to_rgb8(tile["bytes"], out_w=512, out_h=512)
    l1_rgb = tile_rgb
    l2_rgb = _rgb_adjust_linear(tile_rgb, gain=1.08, bias=3, gamma=0.92)

    n = 2**zoom
    nx = (tile["x"] + 1) % n
    ny = min(n - 1, tile["y"] + 1)
    t00 = _png_bytes_to_rgb8(_fetch_osm_tile_by_xyz(zoom, tile["x"], tile["y"]), out_w=256, out_h=256)
    t10 = _png_bytes_to_rgb8(_fetch_osm_tile_by_xyz(zoom, nx, tile["y"]), out_w=256, out_h=256)
    t01 = _png_bytes_to_rgb8(_fetch_osm_tile_by_xyz(zoom, tile["x"], ny), out_w=256, out_h=256)
    t11 = _png_bytes_to_rgb8(_fetch_osm_tile_by_xyz(zoom, nx, ny), out_w=256, out_h=256)
    l3_tiles = [t00, t10, t01, t11]
    l3_rgb = _stitch_2x2_rgb(l3_tiles, tile_w=256, tile_h=256)

    l4_rgb = _rgb_to_classified(l2_rgb, sensor=sensor_norm)
    l1_gray = _rgb_to_gray_u16(l1_rgb) if sensor_norm == "sar" else None
    l2_gray = _rgb_to_gray_u16(l2_rgb) if sensor_norm == "sar" else None
    l3_gray = _rgb_to_gray_u16(l3_rgb) if sensor_norm == "sar" else None
    l4_gray = _rgb_to_classified_gray_u16(l2_rgb, sensor=sensor_norm) if sensor_norm == "sar" else None

    target_defs = level_defs
    if single_mode and level and selected_fmt:
        target_defs = [x for x in level_defs if x[0] == level and x[1] == selected_fmt]
    items: list[ImageItem] = []
    for lvl, f, summary in target_defs:
        suffix = ".bin" if lvl == "L0" else ".tif"
        file_name = f"{req_id}_{lvl}_{f.replace('-', '_')}{suffix}"
        fp = out_dir / file_name

        if lvl == "L0":
            _write_osm_l0_bin(path=fp, tile_bytes=tile["bytes"], meta=l0_meta)
        elif lvl == "L1":
            if sensor_norm == "sar" and l1_gray is not None:
                _write_tiff_gray_u16(fp, 512, 512, l1_gray)
            else:
                _write_tiff_rgb_u8(fp, 512, 512, l1_rgb)
        elif lvl == "L2":
            if sensor_norm == "sar" and l2_gray is not None:
                _write_tiff_gray_u16(fp, 512, 512, l2_gray)
            else:
                _write_tiff_rgb_u8(fp, 512, 512, l2_rgb)
        elif lvl == "L3":
            if sensor_norm == "sar" and l3_gray is not None:
                _write_tiff_gray_u16(fp, 512, 512, l3_gray)
                tiles_dir = out_dir / "tiles"
                tiles_dir.mkdir(parents=True, exist_ok=True)
                for idx, rgb in enumerate(l3_tiles):
                    ty = idx // 2
                    tx = idx % 2
                    tfp = tiles_dir / f"tile_{ty:03d}_{tx:03d}.tif"
                    _write_tiff_gray_u16(tfp, 256, 256, _rgb_to_gray_u16(rgb))
            else:
                _write_tiff_rgb_u8(fp, 512, 512, l3_rgb)
                _make_l3_tiles_from_osm_under(
                    base_dir=out_dir / "tiles",
                    tile_rgbs=l3_tiles,
                    tile_w=256,
                    tile_h=256,
                )
        else:
            if sensor_norm == "sar" and l4_gray is not None:
                _write_tiff_gray_u16(fp, 512, 512, l4_gray)
            else:
                _write_tiff_rgb_u8(fp, 512, 512, l4_rgb)

        items.append(
            ImageItem(
                image_id=f"{req_id}-{lvl.lower()}",
                sensor=sensor_norm,
                level=lvl,
                fmt=f,
                satellite="OSM-simulated",
                acquired_at_utc=ts_text,
                file_name=file_name,
                file_size_bytes=fp.stat().st_size,
                path=str(fp),
                summary=(
                    f"{summary} / OSM z{zoom} ({tile['x']},{tile['y']}) 기반 모사 / "
                    f"{level_notes.get(lvl, '레벨 처리 단계')} / "
                    "학습용 시뮬레이션(물리기반 정밀처리 아님)"
                ),
            )
        )

    OSM_SIM_CATALOG[req_id] = items
    OSM_SIM_SOURCE[req_id] = {
        "provider": "openstreetmap",
        "tile_url": tile["url"],
        "tile_x": tile["x"],
        "tile_y": tile["y"],
        "zoom": zoom,
        "lat": lat,
        "lon": lon,
        "tile_size_bytes": len(tile["bytes"]),
        "tile_sha256": tile_hash,
    }

    payload = {
        "request_id": req_id,
        "source": OSM_SIM_SOURCE[req_id],
        "count": len(items),
        "items": [
            {
                **asdict(x),
                "download_url": f"/osm/images/{req_id}/{x.level}/download",
                "content_url": f"/osm/images/{req_id}/{x.level}/content",
            }
            for x in items
        ],
    }

    if single_mode:
        selected = items[0] if items else None
        if not selected:
            raise HTTPException(status_code=404, detail="Generated file not found")
        fp = Path(selected.path)
        media_type = "image/tiff" if fp.suffix.lower() in {".tif", ".tiff"} else "application/octet-stream"
        return FileResponse(
            path=fp,
            media_type=media_type,
            filename=selected.file_name,
            headers={"X-OSM-Request-ID": req_id},
        )

    return payload


@app.post(
    "/osm/images/generate",
    tags=["osm-images"],
    summary="OSM 단건 샘플 이미지 즉시 생성",
    description=(
        "요청한 `lat`, `lon`, `zoom`, `sensor`, `level`, `fmt` 조합으로 OSM 기반 샘플 파일을 즉시 생성한 뒤 "
        "파일 본문을 바로 응답합니다.\n\n"
        "동작 방식:\n"
        "- 요청마다 OSM 타일을 조회해 단일 레벨 파일을 생성합니다.\n"
        "- 생성된 파일은 `mock_store/osm/{request_id}` 경로에 저장됩니다.\n"
        "- 응답은 `Content-Disposition: attachment`로 내려가며 호출 즉시 다운로드할 수 있습니다.\n\n"
        "포맷 규칙:\n"
        "- EO: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=index-map\n"
        "- SAR: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=classified-raster\n"
        "- 위 조합과 맞지 않으면 `400 Bad Request`를 반환합니다.\n"
        "- 확장자 매핑: ceos=.bin, 그 외 지원 포맷=.tif\n\n"
        "호출 예시:\n"
        "- `POST /osm/images/generate?lat=37.5665&lon=126.9780&zoom=14&sensor=eo&level=L2&fmt=geotiff`\n"
        "- `POST /osm/images/generate?lat=35.1796&lon=129.0756&zoom=13&sensor=sar&level=L4&fmt=classified-raster`\n\n"
        "참고:\n"
        "- 본 API는 `/images/generate`와 동일한 단건 파일응답 패턴입니다.\n"
        "- 응답 헤더 `X-OSM-Request-ID`에 생성 request_id를 함께 제공합니다."
    ),
    response_description="생성된 OSM 샘플 파일(binary stream)",
    responses={
        400: {"description": "sensor/level/fmt 조합 불일치"},
        422: {"description": "lat/lon/zoom 파라미터 검증 실패"},
        502: {"description": "OSM 타일 수신 실패(네트워크/외부 응답 오류)"},
    },
)
def generate_osm_image_file(
    lat: float = Query(..., ge=-85.05112878, le=85.05112878, description="중심 위도(WGS84)", examples=[37.5665]),
    lon: float = Query(..., ge=-180.0, le=180.0, description="중심 경도(WGS84)", examples=[126.978]),
    zoom: int = Query(14, ge=0, le=19, description="OSM 줌 레벨(0=전지구, 숫자 클수록 상세)", examples=[14]),
    sensor: str = Query(..., pattern="^(eo|sar)$", description="모사 센서 타입(eo/sar)", examples=["eo", "sar"]),
    level: str = Query(..., pattern="^L[0-4]$", description="생성 레벨(L0~L4)"),
    fmt: str = Query(
        ...,
        description=(
            "생성 포맷. `sensor`/`level` 조합 규칙을 따라야 합니다. "
            "EO: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=index-map / "
            "SAR: L0=ceos, L1=geotiff, L2=geotiff, L3=tiled-geotiff, L4=classified-raster"
        ),
        examples=["ceos", "geotiff", "tiled-geotiff", "index-map", "classified-raster"],
    ),
) -> FileResponse:
    result = generate_osm_images(lat=lat, lon=lon, zoom=zoom, sensor=sensor, level=level, fmt=fmt)
    if isinstance(result, FileResponse):
        return result
    raise HTTPException(status_code=500, detail="Unexpected response type from OSM single generation")


@app.get(
    "/osm/images",
    tags=["osm-images"],
    summary="OSM 모사 생성 요청 목록 조회",
    description=(
        "서버 메모리에 유지 중인 OSM 생성 요청 목록과 최신 요청의 L0~L4 결과를 반환합니다.\n"
        "UI 새로고침 시 최신 결과를 다시 표시할 때 사용합니다."
    ),
    response_description="요청 목록(requests)과 최신 요청(latest) 정보",
)
def list_osm_images() -> dict:
    _ensure_osm_catalog_loaded()
    request_ids = sorted(OSM_SIM_CATALOG.keys(), reverse=True)
    requests: list[dict] = []
    for rid in request_ids:
        items = OSM_SIM_CATALOG.get(rid, [])
        src = OSM_SIM_SOURCE.get(rid, {})
        requests.append(
            {
                "request_id": rid,
                "count": len(items),
                "acquired_at_utc": items[0].acquired_at_utc if items else None,
                "sensor": items[0].sensor if items else None,
                "source": src,
            }
        )

    latest = None
    if request_ids:
        rid = request_ids[0]
        items = OSM_SIM_CATALOG.get(rid, [])
        latest = {
            "request_id": rid,
            "count": len(items),
            "source": OSM_SIM_SOURCE.get(rid, {}),
            "items": [
                {
                    **asdict(x),
                    "download_url": f"/osm/images/{rid}/{x.level}/download",
                    "content_url": f"/osm/images/{rid}/{x.level}/content",
                }
                for x in items
            ],
        }

    return {
        "request_count": len(request_ids),
        "requests": requests,
        "latest": latest,
    }


@app.get(
    "/osm/images/items",
    tags=["osm-images"],
    summary="OSM 생성 이미지 전체 목록 조회",
    description=(
        "OSM 요청 전체(request_id 단위)에서 생성된 이미지 아이템(L0~L4)을 평탄화하여 조회합니다.\n"
        "랜덤 샘플 `/images`와 유사하게 sensor/level/fmt/q 필터를 지원합니다."
    ),
    response_description="필터 결과 개수(count)와 이미지 배열(items)",
)
def list_osm_image_items(
    sensor: Optional[str] = Query(default=None, pattern="^(eo|sar)$", description="센서 필터(eo/sar)"),
    level: Optional[str] = Query(default=None, pattern="^L[0-4]$", description="레벨 필터(L0~L4)"),
    fmt: Optional[str] = Query(default=None, description="포맷 정확 일치 필터"),
    q: Optional[str] = Query(default=None, description="request_id/image_id/summary/satellite 부분 검색"),
    request_id: Optional[str] = Query(default=None, description="특정 request_id만 조회"),
) -> dict:
    _ensure_osm_catalog_loaded()
    request_ids = sorted(OSM_SIM_CATALOG.keys(), reverse=True)
    items: list[dict] = []
    for rid in request_ids:
        if request_id and rid != request_id:
            continue
        for x in OSM_SIM_CATALOG.get(rid, []):
            item = {
                **asdict(x),
                "request_id": rid,
                "download_url": f"/osm/images/{rid}/{x.level}/download",
                "content_url": f"/osm/images/{rid}/{x.level}/content",
            }
            items.append(item)

    if sensor:
        items = [x for x in items if x.get("sensor") == sensor]
    if level:
        items = [x for x in items if x.get("level") == level]
    if fmt:
        items = [x for x in items if x.get("fmt") == fmt]
    if q:
        qq = q.lower()
        items = [
            x for x in items
            if qq in str(x.get("request_id", "")).lower()
            or qq in str(x.get("image_id", "")).lower()
            or qq in str(x.get("summary", "")).lower()
            or qq in str(x.get("satellite", "")).lower()
        ]

    return {"count": len(items), "items": items}


@app.get(
    "/osm/images/{request_id}",
    tags=["osm-images"],
    summary="OSM 모사 생성 결과 조회",
    description=(
        "request_id로 OSM 모사 결과(L0~L4) 메타데이터를 조회합니다.\n"
        "- 생성 직후 응답을 다시 확인하거나, 다운로드 링크를 재획득할 때 사용합니다.\n"
        "- 서버 재시작 시 메모리 카탈로그는 초기화될 수 있습니다."
    ),
    response_description="레벨별 산출물 메타데이터 목록",
    responses={404: {"description": "request_id 없음/만료"}},
)
def get_osm_images(request_id: str) -> dict:
    _ensure_osm_catalog_loaded()
    items = OSM_SIM_CATALOG.get(request_id)
    if not items:
        raise HTTPException(status_code=404, detail="OSM simulated request not found")
    return {
        "request_id": request_id,
        "count": len(items),
        "items": [
            {
                **asdict(x),
                "download_url": f"/osm/images/{request_id}/{x.level}/download",
                "content_url": f"/osm/images/{request_id}/{x.level}/content",
            }
            for x in items
        ],
    }


@app.get(
    "/osm/images/{request_id}/{level}/download",
    tags=["osm-images"],
    summary="OSM 모사 산출물 다운로드",
    description=(
        "request_id와 level(L0~L4)로 생성된 파일을 다운로드합니다.\n"
        "- L0는 `.bin`(application/octet-stream)\n"
        "- L1~L4는 `.tif`(image/tiff)"
    ),
    response_description="산출물 원본 파일(binary stream)",
    responses={404: {"description": "request_id 또는 level 결과 없음"}},
)
def download_osm_image(
    request_id: str = ApiPath(..., description="생성 요청 식별자"),
    level: str = ApiPath(..., pattern="^L[0-4]$", description="다운로드할 레벨"),
) -> FileResponse:
    _ensure_osm_catalog_loaded()
    items = OSM_SIM_CATALOG.get(request_id)
    if not items:
        raise HTTPException(status_code=404, detail="OSM simulated request not found")
    for p in items:
        if p.level != level:
            continue
        fp = Path(p.path)
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Generated file missing")
        media_type, _ = mimetypes.guess_type(str(fp))
        return FileResponse(path=fp, media_type=media_type, filename=p.file_name)
    raise HTTPException(status_code=404, detail="Level output not found")


@app.get(
    "/osm/images/{request_id}/{level}/content",
    tags=["osm-images"],
    summary="OSM 모사 산출물 콘텐츠(미리보기) 조회",
    description=(
        "request_id와 level로 생성된 산출물의 브라우저 미리보기 콘텐츠를 반환합니다.\n"
        "- TIFF: BMP로 변환 후 반환\n"
        "- BIN: 원본 반환\n"
        "- UI 이미지 태그에서 바로 확인하려면 이 엔드포인트를 사용하세요."
    ),
    response_description="브라우저 미리보기용 콘텐츠",
    responses={
        404: {"description": "request_id 또는 level 결과 없음"},
        415: {"description": "TIFF 미리보기 변환 실패"},
    },
)
def view_osm_image_content(
    request_id: str = ApiPath(..., description="생성 요청 식별자"),
    level: str = ApiPath(..., pattern="^L[0-4]$", description="조회할 레벨"),
):
    _ensure_osm_catalog_loaded()
    items = OSM_SIM_CATALOG.get(request_id)
    if not items:
        raise HTTPException(status_code=404, detail="OSM simulated request not found")
    for p in items:
        if p.level != level:
            continue
        fp = Path(p.path)
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Generated file missing")
        if fp.suffix.lower() in {".tif", ".tiff"}:
            try:
                bmp = _tiff_to_bmp_bytes(fp)
                return Response(content=bmp, media_type="image/bmp")
            except Exception as exc:
                raise HTTPException(status_code=415, detail=f"TIFF preview conversion failed: {exc}") from exc
        media_type, _ = mimetypes.guess_type(str(fp))
        return FileResponse(path=fp, media_type=media_type or "application/octet-stream", filename=p.file_name)
    raise HTTPException(status_code=404, detail="Level output not found")


@app.get(
    "/images/{image_id}",
    tags=["images"],
    summary="샘플 이미지 상세 조회",
    description="image_id로 단일 이미지 메타데이터를 조회합니다.",
    response_description="요청한 image_id의 이미지 메타데이터",
    responses={404: {"description": "image_id에 해당하는 샘플이 없음"}},
)
def get_image(
    image_id: str = ApiPath(
        ...,
        description="조회할 이미지 식별자(image_id). 예: eo-kompsat3-l1-scene001",
    )
) -> dict:
    for p in CATALOG:
        if p.image_id == image_id:
            return asdict(p)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get(
    "/images/{image_id}/download",
    tags=["images"],
    summary="샘플 이미지 원본 다운로드",
    description=(
        "image_id에 해당하는 실제 파일을 내려받습니다.\n"
        "응답은 application/octet-stream으로 반환됩니다."
    ),
    response_description="이미지 원본 파일(binary stream)",
    responses={
        404: {"description": "image_id 없음 또는 물리 파일 누락"},
    },
)
def download_image(
    image_id: str = ApiPath(
        ...,
        description="다운로드할 이미지 식별자(image_id).",
    )
) -> FileResponse:
    for p in CATALOG:
        if p.image_id != image_id:
            continue
        fp = Path(p.path)
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Image file missing")
        return FileResponse(
            path=fp,
            media_type="application/octet-stream",
            filename=p.file_name,
        )
    raise HTTPException(status_code=404, detail="Image not found")


@app.get(
    "/images/{image_id}/content",
    tags=["images"],
    summary="샘플 이미지 콘텐츠(미리보기) 조회",
    description=(
        "브라우저 표시를 위한 콘텐츠를 반환합니다.\n"
        "- TIFF 파일: 서버에서 BMP로 변환 후 반환\n"
        "- 그 외 확장자: 원본 파일을 적절한 media type으로 반환"
    ),
    response_description="브라우저 미리보기용 이미지 콘텐츠",
    responses={
        404: {"description": "image_id 없음 또는 물리 파일 누락"},
        415: {"description": "TIFF 미리보기 변환 실패"},
    },
)
def view_image_content(
    image_id: str = ApiPath(
        ...,
        description="미리보기할 이미지 식별자(image_id).",
    )
) -> FileResponse:
    for p in CATALOG:
        if p.image_id != image_id:
            continue
        fp = Path(p.path)
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Image file missing")
        if fp.suffix.lower() in {".tif", ".tiff"}:
            try:
                bmp = _tiff_to_bmp_bytes(fp)
                return Response(content=bmp, media_type="image/bmp")
            except Exception as exc:
                raise HTTPException(status_code=415, detail=f"TIFF preview conversion failed: {exc}") from exc
        media_type, _ = mimetypes.guess_type(str(fp))
        return FileResponse(
            path=fp,
            media_type=media_type or "application/octet-stream",
            filename=p.file_name,
        )
    raise HTTPException(status_code=404, detail="Image not found")


@app.get(
    "/admin/mock-store/info",
    tags=["mock-store"],
    summary="mock_store 상태 조회",
    description=(
        "mock_store 디렉터리 상태, 카탈로그 경로, 생성된 폴더/파일 목록을 반환합니다.\n"
        "UI에서 샘플 생성 상태를 동기화할 때 사용합니다."
    ),
    response_description="mock_store 상태 및 생성 파일/폴더 목록",
)
def mock_store_info() -> dict:
    catalog_folders = _collect_image_folders(CATALOG)
    catalog_files = _collect_image_files(CATALOG)
    store_folders = _collect_store_folders()
    store_files = _collect_store_files()
    return {
        "store_dir": str(STORE_DIR),
        "catalog_path": str(CATALOG_PATH),
        "disabled": MOCK_STORE_DISABLED_FLAG.exists(),
        "exists": STORE_DIR.exists(),
        "image_count": len(CATALOG),
        "image_folders": catalog_folders,
        "image_files": catalog_files,
        "catalog_folders": catalog_folders,
        "catalog_files": catalog_files,
        "store_folders": store_folders,
        "store_files": store_files,
    }


@app.get(
    "/admin/osm-store/info",
    tags=["osm-images"],
    summary="OSM 저장소 상태 조회",
    description=(
        "OSM 생성 결과 저장소(mock_store/osm) 상태와 최신 요청 요약을 반환합니다.\n"
        "UI에서 OSM 샘플 상태/파일 목록을 표시할 때 사용합니다."
    ),
    response_description="OSM 저장소 상태 및 최신 요청 통계/파일 목록",
)
def osm_store_info() -> dict:
    _ensure_osm_catalog_loaded()
    folders: list[str] = []
    files: list[str] = []
    if OSM_STORE_DIR.exists():
        folder_set: set[str] = set()
        for fp in OSM_STORE_DIR.rglob("*"):
            if not fp.is_file():
                continue
            try:
                files.append(str(fp.relative_to(STORE_DIR)))
                folder_set.add(str(fp.parent.relative_to(STORE_DIR)))
            except ValueError:
                files.append(str(fp))
                folder_set.add(str(fp.parent))
        folders = sorted(folder_set)
        files = sorted(files)

    request_ids = sorted(OSM_SIM_CATALOG.keys(), reverse=True)
    latest_items = OSM_SIM_CATALOG.get(request_ids[0], []) if request_ids else []
    by_level = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for it in latest_items:
        if it.level in by_level:
            by_level[it.level] += 1

    return {
        "store_dir": str(OSM_STORE_DIR),
        "exists": OSM_STORE_DIR.exists(),
        "request_count": len(request_ids),
        "latest_request_id": request_ids[0] if request_ids else None,
        "latest_image_count": len(latest_items),
        "latest_level_counts": by_level,
        "image_folders": folders,
        "image_files": files,
    }


@app.post(
    "/admin/osm-store/delete",
    tags=["osm-images"],
    summary="OSM 생성 데이터 전체 삭제",
    description="mock_store/osm 아래 OSM 생성 결과를 전체 삭제합니다.",
    response_description="삭제 결과(ok/message/store_dir)",
)
def delete_osm_store() -> dict:
    OSM_SIM_CATALOG.clear()
    OSM_SIM_SOURCE.clear()
    if OSM_STORE_DIR.exists():
        shutil.rmtree(OSM_STORE_DIR)
    return {
        "ok": True,
        "message": "osm store deleted",
        "store_dir": str(OSM_STORE_DIR),
    }


@app.post(
    "/admin/mock-store/rebuild",
    tags=["mock-store"],
    summary="샘플 데이터 재생성",
    description=(
        "샘플 위성 데이터를 새로 생성합니다.\n"
        "- 기존 파일이 남아 있으면 409를 반환합니다.\n"
        "- 삭제 후 재생성할 때 새로운 랜덤 시드를 사용해 다른 장면을 만듭니다."
    ),
    response_description="재생성 결과(ok/message/image_count/image_folders)",
    responses={409: {"description": "기존 샘플이 남아 있어 재생성 불가"}},
)
def rebuild_mock_store() -> dict:
    global CATALOG
    if _collect_image_files(CATALOG):
        raise HTTPException(status_code=409, detail="샘플이 이미 존재합니다. 샘플 모두 삭제 후 다시 시도해주세요.")
    if MOCK_STORE_DISABLED_FLAG.exists():
        MOCK_STORE_DISABLED_FLAG.unlink()
    CATALOG = _build_mock_store(force_rebuild=True)
    return {
        "ok": True,
        "message": "mock store rebuilt",
        "image_count": len(CATALOG),
        "image_folders": _collect_image_folders(CATALOG),
    }


@app.post(
    "/admin/mock-store/delete",
    tags=["mock-store"],
    summary="샘플 데이터 전체 삭제",
    description=(
        "mock_store 디렉터리의 샘플 파일을 전체 삭제하고, 자동 재생성을 막는 disabled 플래그를 생성합니다."
    ),
    response_description="삭제 결과(ok/message/store_dir)",
)
def delete_mock_store() -> dict:
    global CATALOG
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    MOCK_STORE_DISABLED_FLAG.write_text("disabled\n", encoding="utf-8")
    CATALOG = []
    return {
        "ok": True,
        "message": "mock store deleted",
        "store_dir": str(STORE_DIR),
    }


@app.get(
    "/hub",
    response_class=HTMLResponse,
    tags=["docs-hub"],
    summary="문서 허브",
    description="문서 허브 메인 화면을 반환합니다.",
)
def docs_hub(
    category: Optional[str] = Query(default=None, description="카테고리 필터"),
    tag: Optional[str] = Query(default=None, description="태그 필터"),
    sort: str = Query(default="latest", description="latest | popular | title"),
    q: str = Query(default="", description="검색어"),
) -> str:
    return DOC_SERVICE.render_hub_page(category=category, tag=tag, sort=sort, q=q)


@app.get(
    "/hub/category/{category}",
    response_class=HTMLResponse,
    tags=["docs-hub"],
    summary="카테고리별 문서 허브",
    description="카테고리별 문서 허브 화면을 반환합니다.",
)
def docs_hub_by_category(
    category: str,
    tag: Optional[str] = Query(default=None, description="태그 필터"),
    sort: str = Query(default="latest", description="latest | popular | title"),
    q: str = Query(default="", description="검색어"),
) -> str:
    return DOC_SERVICE.render_hub_page(category=category, tag=tag, sort=sort, q=q)


@app.get(
    "/p/{slug}",
    response_class=HTMLResponse,
    tags=["docs-hub"],
    summary="문서 상세 페이지",
    description="문서 상세 HTML 화면을 반환합니다.",
)
def docs_detail_page(slug: str) -> str:
    return DOC_SERVICE.render_detail_page(slug)


@app.get(
    "/upload",
    response_class=HTMLResponse,
    tags=["docs-hub"],
    summary="문서 업로드 페이지",
    description="Markdown 업로드 화면을 반환합니다.",
)
def docs_upload_page() -> str:
    return DOC_SERVICE.render_upload_page()


@app.get(
    "/admin/docs",
    response_class=HTMLResponse,
    tags=["docs-hub"],
    summary="운영자 문서 관리 화면",
    description="문서 인덱스 상태와 재색인 기능을 제공하는 운영자 화면입니다.",
)
def docs_admin_page(request: Request) -> str:
    _require_docs_admin(request)
    return DOC_SERVICE.render_admin_page()


@app.get(
    "/api/docs/list",
    response_class=JSONResponse,
    tags=["docs-hub"],
    summary="문서 목록 조회",
    description="문서 허브 목록을 조회합니다.",
)
def api_docs_list(
    category: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    sort: str = Query(default="latest"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    return _docs_json_response(
        lambda: DOC_SERVICE.list_documents(category=category, tag=tag, sort=sort, page=page, page_size=page_size)
    )


@app.get(
    "/api/docs/search",
    response_class=JSONResponse,
    tags=["docs-hub"],
    summary="문서 검색",
    description="제목, 태그, 요약, headings, 본문 일부를 검색합니다.",
)
def api_docs_search(
    q: str = Query(..., description="검색어"),
    category: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    return _docs_json_response(
        lambda: DOC_SERVICE.search_documents(query=q, category=category, tag=tag, page=page, page_size=page_size)
    )


@app.get(
    "/api/docs/{slug}",
    response_class=JSONResponse,
    tags=["docs-hub"],
    summary="문서 상세 조회",
    description="문서 메타데이터와 본문 HTML을 반환합니다.",
)
def api_docs_detail(slug: str) -> dict:
    return _docs_json_response(lambda: DOC_SERVICE.get_document_detail(slug))


@app.get(
    "/api/docs/{slug}/raw",
    response_class=Response,
    tags=["docs-hub"],
    summary="원본 Markdown 조회",
    description="문서 원본 Markdown을 반환합니다.",
)
def api_docs_raw(slug: str) -> Response:
    raw = DOC_SERVICE.get_document_raw(slug)
    return Response(content=raw, media_type="text/markdown; charset=utf-8")


@app.post(
    "/api/docs/preview",
    response_class=JSONResponse,
    tags=["docs-hub"],
    summary="문서 미리보기",
    description="Markdown 본문을 서버 렌더링하여 업로드 전 HTML 미리보기를 반환합니다.",
)
async def api_docs_preview(payload: GuideUploadPayload) -> JSONResponse:
    result = _docs_json_response(lambda: DOC_SERVICE.preview_document(filename=payload.filename, content=payload.content))
    if isinstance(result, dict):
        return JSONResponse(content=result)
    return result


@app.post(
    "/api/docs/upload",
    response_class=JSONResponse,
    tags=["docs-hub"],
    summary="문서 업로드",
    description="Markdown 파일을 업로드하고 인덱스 반영을 시도합니다.",
)
async def api_docs_upload(payload: GuideUploadPayload) -> JSONResponse:
    result = _docs_json_response(lambda: DOC_SERVICE.upload_document(filename=payload.filename, content=payload.content))
    if isinstance(result, tuple):
        status_code, payload = result
        return JSONResponse(content=payload, status_code=status_code)
    return result


@app.post(
    "/api/docs/reindex",
    response_class=JSONResponse,
    tags=["docs-hub"],
    summary="문서 인덱스 재생성",
    description="운영자용 전체 인덱스 재생성 API입니다.",
)
def api_docs_reindex(request: Request) -> dict:
    _require_docs_admin(request)
    return _docs_json_response(DOC_SERVICE.reindex)


@app.get(
    "/guide",
    response_class=HTMLResponse,
    tags=["guide"],
    summary="Guide 문서 목록",
    description="프로젝트 루트 `guide` 폴더의 Markdown(.md) 목록을 HTML 페이지로 보여줍니다.",
    response_description="HTML 문서 목록 페이지",
)
def guide_index() -> str:
    guide_rel = "guide/"
    GUIDE_DIR.mkdir(parents=True, exist_ok=True)
    upload_widget = _guide_upload_widget_html()
    files = _guide_markdown_files()
    if not files:
        return (
            "<html><body style='font-family:sans-serif;padding:20px;'>"
            "<p><a href='/'>&larr; 메인 홈페이지</a></p>"
            "<h2>Guide 문서</h2>"
            f"<p>기준 폴더: <code>{html.escape(guide_rel)}</code> (상대경로)</p>"
            f"{upload_widget}"
            "<p>.md 파일이 없습니다.</p>"
            "</body></html>"
        )
    items: list[str] = []
    for fp in files:
        rel = fp.relative_to(GUIDE_DIR).as_posix()
        display_rel = unicodedata.normalize("NFC", rel)
        href = "/guide/" + quote(rel, safe="/")
        items.append(f"<li><a href='{href}'>{html.escape(display_rel)}</a></li>")
    return (
        "<html><body style='font-family:sans-serif;padding:20px;line-height:1.6;'>"
        "<p><a href='/'>&larr; 메인 홈페이지</a></p>"
        "<h2>Guide 문서 목록</h2>"
        f"<p>기준 폴더: <code>{html.escape(guide_rel)}</code> (상대경로)</p>"
        f"{upload_widget}"
        "<ul>"
        + "".join(items)
        + "</ul>"
        "</body></html>"
    )


@app.get(
    "/guide/{guide_path:path}",
    response_class=HTMLResponse,
    tags=["guide"],
    summary="Guide 문서 렌더링",
    description="선택한 Markdown(.md) 파일을 읽어 HTML로 렌더링합니다.",
    response_description="렌더링된 HTML 문서",
)
def guide_view(guide_path: str) -> str:
    fp = _resolve_guide_file(guide_path)
    md_text = fp.read_text(encoding="utf-8")
    rel = fp.relative_to(GUIDE_DIR).as_posix()
    return render_guide_page(rel=rel, md_text=md_text)


@app.get(
    "/guide/upload",
    response_class=HTMLResponse,
    tags=["guide"],
    summary="Guide 업로드 안내",
    description="업로드는 POST `/guide/upload`를 사용합니다. 이 경로는 `/guide`로 리다이렉트합니다.",
)
def guide_upload_info() -> RedirectResponse:
    return RedirectResponse(url="/guide", status_code=303)


@app.post(
    "/guide/upload",
    response_class=JSONResponse,
    tags=["guide"],
    summary="Guide Markdown 업로드",
    description="웹에서 업로드한 `.md` 파일 내용을 JSON으로 받아 `guide/` 폴더에 저장합니다.",
    response_description="업로드 결과(JSON) 및 리다이렉트 URL",
)
async def guide_upload(payload: GuideUploadPayload) -> dict:
    GUIDE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_md_filename(payload.filename or "")
    target = (GUIDE_DIR / safe_name).resolve()
    guide_root = GUIDE_DIR.resolve()
    if guide_root != target.parent and guide_root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid upload path")
    data = payload.content.encode("utf-8")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    target.write_bytes(data)
    return {
        "ok": True,
        "file_name": safe_name,
        "size_bytes": len(data),
        "redirect_url": f"/guide/{quote(safe_name)}",
    }


@app.get(
    "/ui",
    response_class=HTMLResponse,
    tags=["ui"],
    summary="웹 UI 페이지",
    description="샘플 생성/조회/다운로드/미리보기를 위한 단일 HTML UI 페이지를 반환합니다.",
    response_description="HTML 문서",
)
def ui() -> str:
    try:
        return UI_HTML_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"UI template not found: {UI_HTML_PATH}") from exc


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["ui"],
    summary="루트 안내 페이지",
    description="`/ui`와 `/docs`로 이동할 수 있는 간단한 시작 페이지를 반환합니다.",
    response_description="HTML 문서",
)
def root() -> str:
    return DOC_SERVICE.render_landing_page()

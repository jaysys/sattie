#!/usr/bin/env python3
"""
Satti (virtual) - FastAPI service for serving satellite imagery products.

Run:
  uvicorn app.satti_api:app --reload --host 0.0.0.0 --port 6001
"""

from __future__ import annotations

import json
import math
import mimetypes
import random
import shutil
import struct
import io
from array import array
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STORE_DIR = PROJECT_DIR / "mock_store"
CATALOG_PATH = STORE_DIR / "catalog.json"
MOCK_STORE_DISABLED_FLAG = PROJECT_DIR / ".mock_store_disabled"
MOCK_STORE_VERSION = 3


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Product:
    product_id: str
    sensor: str          # eo | sar
    level: str           # L0~L4
    fmt: str             # geotiff | ceos | tiled-geotiff | index-map | classified-raster
    satellite: str       # e.g., KOMPSAT-3, KOMPSAT-5
    acquired_at_utc: str
    file_name: str
    file_size_bytes: int
    path: str
    summary: str


def _write_tiff_gray_u16(path: Path, width: int, height: int, data_u16: list[int]) -> None:
    """
    Minimal uncompressed grayscale TIFF writer (uint16, little-endian).
    """
    if len(data_u16) != width * height:
        raise ValueError("TIFF data length mismatch.")

    path.parent.mkdir(parents=True, exist_ok=True)
    img = array("H", data_u16)
    img_bytes = img.tobytes()

    header = bytearray()
    header += b"II"
    header += struct.pack("<H", 42)
    ifd_offset = 8 + len(img_bytes)
    header += struct.pack("<I", ifd_offset)

    entries = [
        (256, 4, 1, width),          # ImageWidth
        (257, 4, 1, height),         # ImageLength
        (258, 3, 1, 16),             # BitsPerSample
        (259, 3, 1, 1),              # Compression none
        (262, 3, 1, 1),              # BlackIsZero
        (273, 4, 1, 8),              # Strip offset
        (277, 3, 1, 1),              # SamplesPerPixel
        (278, 4, 1, height),         # RowsPerStrip
        (279, 4, 1, len(img_bytes)), # StripByteCounts
        (284, 3, 1, 1),              # PlanarConfiguration
    ]

    ifd = bytearray()
    ifd += struct.pack("<H", len(entries))
    for tag, typ, count, value in entries:
        ifd += struct.pack("<HHII", tag, typ, count, value)
    ifd += struct.pack("<I", 0)

    with path.open("wb") as f:
        f.write(header)
        f.write(img_bytes)
        f.write(ifd)


def _write_tiff_rgb_u8(path: Path, width: int, height: int, data_rgb: list[tuple[int, int, int]]) -> None:
    """
    Minimal uncompressed RGB TIFF writer (uint8, little-endian).
    """
    if len(data_rgb) != width * height:
        raise ValueError("TIFF RGB data length mismatch.")

    path.parent.mkdir(parents=True, exist_ok=True)
    img_bytes = bytearray()
    for r, g, b in data_rgb:
        img_bytes.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    header = bytearray()
    header += b"II"
    header += struct.pack("<H", 42)
    ifd_offset = 8 + len(img_bytes)
    header += struct.pack("<I", ifd_offset)

    entries = [
        (256, 4, 1, width),           # ImageWidth
        (257, 4, 1, height),          # ImageLength
        (258, 3, 3, 0),               # BitsPerSample (offset patched later)
        (259, 3, 1, 1),               # Compression none
        (262, 3, 1, 2),               # Photometric RGB
        (273, 4, 1, 8),               # Strip offset
        (277, 3, 1, 3),               # SamplesPerPixel
        (278, 4, 1, height),          # RowsPerStrip
        (279, 4, 1, len(img_bytes)),  # StripByteCounts
        (284, 3, 1, 1),               # PlanarConfiguration chunky
    ]

    ifd_size = 2 + (len(entries) * 12) + 4
    bits_offset = ifd_offset + ifd_size
    entries[2] = (258, 3, 3, bits_offset)

    ifd = bytearray()
    ifd += struct.pack("<H", len(entries))
    for tag, typ, count, value in entries:
        ifd += struct.pack("<HHII", tag, typ, count, value)
    ifd += struct.pack("<I", 0)
    extra = struct.pack("<HHH", 8, 8, 8)

    with path.open("wb") as f:
        f.write(header)
        f.write(img_bytes)
        f.write(ifd)
        f.write(extra)


def _make_dummy_image(path: Path, width: int = 512, height: int = 512, seed: int = 42, sensor: str = "eo") -> None:
    rng = random.Random(seed)
    if sensor == "eo":
        data_rgb: list[tuple[int, int, int]] = []
        for y in range(height):
            yn = y / max(1, height - 1)
            for x in range(width):
                xn = x / max(1, width - 1)

                # Procedural "continent" mask + terrain variation.
                n1 = math.sin((xn * 7.0) + (seed * 0.01)) + 0.7 * math.sin((yn * 5.2) - (seed * 0.02))
                n2 = 0.6 * math.sin(((xn + yn) * 10.5) + (seed * 0.03))
                n3 = 0.35 * math.sin((xn * 19.0) - (yn * 13.0))
                land_score = n1 + n2 + n3
                coast = -0.18 <= land_score <= 0.22
                is_land = land_score > 0.02

                lat = abs((yn - 0.5) * 2.0)
                elev = max(0.0, min(1.0, 0.5 + 0.5 * math.sin((xn * 16.0) + (yn * 8.0))))

                if is_land:
                    veg = max(0.0, 1.0 - lat * 1.1)
                    dry = max(0.0, lat - 0.35)
                    r = int(52 + 55 * dry + 38 * elev)
                    g = int(78 + 105 * veg + 22 * elev)
                    b = int(42 + 30 * (1.0 - veg))
                    if coast:
                        r, g, b = int(r * 0.95), int(g * 1.03), int(b * 0.9)
                else:
                    depth = max(0.0, min(1.0, (0.2 - land_score) * 0.8))
                    r = int(8 + 18 * (1.0 - depth))
                    g = int(55 + 70 * (1.0 - depth))
                    b = int(112 + 115 * (1.0 - depth))
                    if coast:
                        r, g, b = int(r + 10), int(g + 22), int(b + 20)

                # Cloud layer.
                cloud = (
                    0.55 * math.sin((xn * 21.0) + (seed * 0.07))
                    + 0.45 * math.sin((yn * 24.0) - (seed * 0.05))
                    + 0.35 * math.sin(((xn + yn) * 33.0) + (seed * 0.11))
                )
                cloud = max(0.0, min(1.0, cloud - 0.35))
                if rng.random() < 0.015:
                    cloud = min(1.0, cloud + 0.35)

                # Snow near poles/high elevation.
                snow = max(0.0, (lat - 0.76) * 2.6)
                if is_land:
                    snow = max(snow, max(0.0, (elev - 0.86) * 3.5))

                overlay = min(1.0, cloud * 0.78 + snow * 0.65)
                r = int(r * (1.0 - overlay) + 245 * overlay)
                g = int(g * (1.0 - overlay) + 247 * overlay)
                b = int(b * (1.0 - overlay) + 250 * overlay)

                # Sensor-like subtle grain.
                grain = rng.randint(-5, 5)
                r = max(0, min(255, r + grain))
                g = max(0, min(255, g + grain))
                b = max(0, min(255, b + grain))
                data_rgb.append((r, g, b))

        _write_tiff_rgb_u8(path, width, height, data_rgb)
        return

    data_gray = []
    for y in range(height):
        for x in range(width):
            xn = x / max(1, width - 1)
            yn = y / max(1, height - 1)
            texture = (
                0.55 * math.sin(xn * 28.0)
                + 0.35 * math.sin((xn + yn) * 17.0)
                + 0.25 * math.sin((xn * 49.0) - (yn * 31.0))
            )
            speckle = int(rng.random() * 3500)
            v = int(24000 + 16000 * texture + speckle)
            data_gray.append(max(0, min(65535, v)))
    _write_tiff_gray_u16(path, width, height, data_gray)


def _make_dummy_bin(path: Path, size_bytes: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        chunk = bytearray(4096)
        remaining = size_bytes
        while remaining > 0:
            n = min(remaining, len(chunk))
            for i in range(n):
                chunk[i] = rng.randrange(0, 256)
            f.write(chunk[:n])
            remaining -= n


def _make_l3_tiles(sensor: str, product_slug: str, seed: int = 42) -> list[Path]:
    """
    Generate simple 2x2 tile set for L3 preview/use-case demonstration.
    """
    tiles_dir = STORE_DIR / sensor / "L3" / "tiles" / product_slug
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for ty in range(2):
        for tx in range(2):
            tile_path = tiles_dir / f"tile_{ty:03d}_{tx:03d}.tif"
            _make_dummy_image(
                tile_path,
                width=256,
                height=256,
                seed=(seed + ty * 101 + tx * 211),
                sensor=sensor,
            )
            out.append(tile_path)
    return out


def _u16_to_u8(v: int) -> int:
    if v <= 0:
        return 0
    if v >= 65535:
        return 255
    return v >> 8


def _write_bmp_bytes_rgb8(width: int, height: int, rgb: list[tuple[int, int, int]]) -> bytes:
    if len(rgb) != width * height:
        raise ValueError("BMP RGB length mismatch.")
    buf = io.BytesIO()
    row_bytes = width * 3
    pad = (4 - (row_bytes % 4)) % 4
    pixel_array_size = (row_bytes + pad) * height
    file_size = 14 + 40 + pixel_array_size
    buf.write(b"BM")
    buf.write(struct.pack("<I", file_size))
    buf.write(struct.pack("<HH", 0, 0))
    buf.write(struct.pack("<I", 14 + 40))
    buf.write(struct.pack("<I", 40))
    buf.write(struct.pack("<i", width))
    buf.write(struct.pack("<i", height))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 24))
    buf.write(struct.pack("<I", 0))
    buf.write(struct.pack("<I", pixel_array_size))
    buf.write(struct.pack("<i", 2835))
    buf.write(struct.pack("<i", 2835))
    buf.write(struct.pack("<I", 0))
    buf.write(struct.pack("<I", 0))
    for y in range(height - 1, -1, -1):
        row = y * width
        for x in range(width):
            r, g, b = rgb[row + x]
            buf.write(struct.pack("BBB", b, g, r))
        if pad:
            buf.write(b"\x00" * pad)
    return buf.getvalue()


def _read_u16_le(raw: bytes, off: int) -> int:
    return int.from_bytes(raw[off : off + 2], "little", signed=False)


def _read_u32_le(raw: bytes, off: int) -> int:
    return int.from_bytes(raw[off : off + 4], "little", signed=False)


def _tiff_to_bmp_bytes(path: Path) -> bytes:
    """
    Converts simple uncompressed TIFF (single strip, generated by this service)
    to 24-bit BMP bytes for browser preview.
    """
    raw = path.read_bytes()
    if len(raw) < 8 or raw[0:2] != b"II" or _read_u16_le(raw, 2) != 42:
        raise ValueError("Unsupported TIFF header")
    ifd_off = _read_u32_le(raw, 4)
    n = _read_u16_le(raw, ifd_off)
    tags: dict[int, tuple[int, int, int]] = {}
    base = ifd_off + 2
    for i in range(n):
        eoff = base + i * 12
        tag = _read_u16_le(raw, eoff)
        typ = _read_u16_le(raw, eoff + 2)
        cnt = _read_u32_le(raw, eoff + 4)
        val = _read_u32_le(raw, eoff + 8)
        tags[tag] = (typ, cnt, val)

    width = tags.get(256, (0, 0, 0))[2]
    height = tags.get(257, (0, 0, 0))[2]
    photometric = tags.get(262, (0, 0, 1))[2]
    strip_off = tags.get(273, (0, 0, 0))[2]
    strip_len = tags.get(279, (0, 0, 0))[2]
    spp = tags.get(277, (0, 0, 1))[2]
    bits = tags.get(258, (0, 1, 8))
    if width <= 0 or height <= 0 or strip_off <= 0 or strip_len <= 0:
        raise ValueError("Unsupported TIFF tags")

    pixel = raw[strip_off : strip_off + strip_len]
    rgb: list[tuple[int, int, int]] = []

    # RGB uint8
    if photometric == 2 and spp == 3:
        if bits[1] == 3:
            bits_off = bits[2]
            b0 = _read_u16_le(raw, bits_off)
            b1 = _read_u16_le(raw, bits_off + 2)
            b2 = _read_u16_le(raw, bits_off + 4)
            if b0 != 8 or b1 != 8 or b2 != 8:
                raise ValueError("Unsupported RGB bit depth")
        idx = 0
        px_count = width * height
        for _ in range(px_count):
            r = pixel[idx]
            g = pixel[idx + 1]
            b = pixel[idx + 2]
            rgb.append((r, g, b))
            idx += 3
        return _write_bmp_bytes_rgb8(width, height, rgb)

    # Grayscale uint16
    if photometric == 1 and spp == 1:
        if bits[2] != 16:
            raise ValueError("Unsupported grayscale bit depth")
        arr = array("H")
        arr.frombytes(pixel)
        for v in arr:
            g = _u16_to_u8(v)
            rgb.append((g, g, g))
        return _write_bmp_bytes_rgb8(width, height, rgb)

    raise ValueError("Unsupported TIFF format")


def _build_mock_store(force_rebuild: bool = False) -> list[Product]:
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Create mock files (idempotent).
    files = {
        "eo_l0": STORE_DIR / "eo" / "L0" / "K3_L0_raw_001.bin",
        "eo_l1": STORE_DIR / "eo" / "L1" / "K3_L1_scene_001.tif",
        "eo_l2": STORE_DIR / "eo" / "L2" / "K3_L2_scene_001.tif",
        "eo_l3": STORE_DIR / "eo" / "L3" / "K3_L3_mosaic_001.tif",
        "eo_l4": STORE_DIR / "eo" / "L4" / "K3_L4_index_001.tif",
        "sar_l0": STORE_DIR / "sar" / "L0" / "K5_L0_raw_001.bin",
        "sar_l1": STORE_DIR / "sar" / "L1" / "K5_L1_intensity_001.tif",
        "sar_l2": STORE_DIR / "sar" / "L2" / "K5_L2_geocoded_001.tif",
        "sar_l3": STORE_DIR / "sar" / "L3" / "K5_L3_mosaic_001.tif",
        "sar_l4": STORE_DIR / "sar" / "L4" / "K5_L4_change_001.tif",
    }

    for key, fp in files.items():
        if fp.exists() and not force_rebuild:
            continue
        if fp.suffix == ".tif":
            sensor = "eo" if key.startswith("eo_") else "sar"
            _make_dummy_image(fp, width=512, height=512, seed=hash(key) & 0xFFFF, sensor=sensor)
        else:
            _make_dummy_bin(fp, size_bytes=1024 * 1024, seed=hash(key) & 0xFFFF)

    # L3 service tiles (mosaic derivatives)
    _make_l3_tiles(sensor="eo", product_slug="eo-kompsat3-l3-mosaic001", seed=31001)
    _make_l3_tiles(sensor="sar", product_slug="sar-kompsat5-l3-mosaic001", seed=51001)

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    catalog = [
        Product(
            product_id="eo-kompsat3-l0-raw001",
            sensor="eo",
            level="L0",
            fmt="ceos",
            satellite="KOMPSAT-3",
            acquired_at_utc=ts,
            file_name=files["eo_l0"].name,
            file_size_bytes=files["eo_l0"].stat().st_size,
            path=str(files["eo_l0"]),
            summary="원시 EO 원격측정 기반 L0 샘플",
        ),
        Product(
            product_id="eo-kompsat3-l1-scene001",
            sensor="eo",
            level="L1",
            fmt="geotiff",
            satellite="KOMPSAT-3",
            acquired_at_utc=ts,
            file_name=files["eo_l1"].name,
            file_size_bytes=files["eo_l1"].stat().st_size,
            path=str(files["eo_l1"]),
            summary="Radiometric/geometric 기본 보정 단계 EO L1 샘플",
        ),
        Product(
            product_id="eo-kompsat3-l2-scene001",
            sensor="eo",
            level="L2",
            fmt="geotiff",
            satellite="KOMPSAT-3",
            acquired_at_utc=ts,
            file_name=files["eo_l2"].name,
            file_size_bytes=files["eo_l2"].stat().st_size,
            path=str(files["eo_l2"]),
            summary="정사보정 기반 EO L2 샘플",
        ),
        Product(
            product_id="eo-kompsat3-l3-mosaic001",
            sensor="eo",
            level="L3",
            fmt="tiled-geotiff",
            satellite="KOMPSAT-3",
            acquired_at_utc=ts,
            file_name=files["eo_l3"].name,
            file_size_bytes=files["eo_l3"].stat().st_size,
            path=str(files["eo_l3"]),
            summary="모자이크/서비스용 EO L3 샘플",
        ),
        Product(
            product_id="eo-kompsat3-l4-index001",
            sensor="eo",
            level="L4",
            fmt="index-map",
            satellite="KOMPSAT-3",
            acquired_at_utc=ts,
            file_name=files["eo_l4"].name,
            file_size_bytes=files["eo_l4"].stat().st_size,
            path=str(files["eo_l4"]),
            summary="분석 산출물 EO L4 샘플",
        ),
        Product(
            product_id="sar-kompsat5-l0-raw001",
            sensor="sar",
            level="L0",
            fmt="ceos",
            satellite="KOMPSAT-5",
            acquired_at_utc=ts,
            file_name=files["sar_l0"].name,
            file_size_bytes=files["sar_l0"].stat().st_size,
            path=str(files["sar_l0"]),
            summary="원시 SAR IQ 기반 L0 샘플",
        ),
        Product(
            product_id="sar-kompsat5-l1-intensity001",
            sensor="sar",
            level="L1",
            fmt="geotiff",
            satellite="KOMPSAT-5",
            acquired_at_utc=ts,
            file_name=files["sar_l1"].name,
            file_size_bytes=files["sar_l1"].stat().st_size,
            path=str(files["sar_l1"]),
            summary="강도 영상 SAR L1 샘플",
        ),
        Product(
            product_id="sar-kompsat5-l2-geocoded001",
            sensor="sar",
            level="L2",
            fmt="geotiff",
            satellite="KOMPSAT-5",
            acquired_at_utc=ts,
            file_name=files["sar_l2"].name,
            file_size_bytes=files["sar_l2"].stat().st_size,
            path=str(files["sar_l2"]),
            summary="지오코딩 기반 SAR L2 샘플",
        ),
        Product(
            product_id="sar-kompsat5-l3-mosaic001",
            sensor="sar",
            level="L3",
            fmt="tiled-geotiff",
            satellite="KOMPSAT-5",
            acquired_at_utc=ts,
            file_name=files["sar_l3"].name,
            file_size_bytes=files["sar_l3"].stat().st_size,
            path=str(files["sar_l3"]),
            summary="모자이크/서비스용 SAR L3 샘플",
        ),
        Product(
            product_id="sar-kompsat5-l4-change001",
            sensor="sar",
            level="L4",
            fmt="classified-raster",
            satellite="KOMPSAT-5",
            acquired_at_utc=ts,
            file_name=files["sar_l4"].name,
            file_size_bytes=files["sar_l4"].stat().st_size,
            path=str(files["sar_l4"]),
            summary="변화탐지형 SAR L4 샘플",
        ),
    ]

    write_obj = {
        "mock_store_version": MOCK_STORE_VERSION,
        "generated_at_utc": ts,
        "count": len(catalog),
        "products": [asdict(x) for x in catalog],
    }
    CATALOG_PATH.write_text(json.dumps(write_obj, indent=2, ensure_ascii=False))
    return catalog


def _load_catalog() -> list[Product]:
    # If disabled flag exists, never auto-rebuild; rebuild only via explicit admin action.
    if MOCK_STORE_DISABLED_FLAG.exists():
        return []
    if not CATALOG_PATH.exists():
        return _build_mock_store()
    obj = read_json(CATALOG_PATH)
    products = [Product(**x) for x in obj["products"]]

    # Keep catalog aligned with current mock model (EO L0 included).
    if obj.get("mock_store_version") != MOCK_STORE_VERSION:
        return _build_mock_store(force_rebuild=True)
    if not any(p.sensor == "eo" and p.level == "L0" for p in products):
        return _build_mock_store(force_rebuild=True)
    if not any(p.sensor == "sar" and p.level == "L3" for p in products):
        return _build_mock_store(force_rebuild=True)
    return products


def _collect_image_folders(products: list[Product]) -> list[str]:
    if not STORE_DIR.exists():
        return []
    folders: set[str] = set()
    for fp in STORE_DIR.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            folders.add(str(fp.parent.relative_to(STORE_DIR)))
        except ValueError:
            folders.add(str(fp.parent))
    return sorted(folders)


def _collect_image_files(products: list[Product]) -> list[str]:
    if not STORE_DIR.exists():
        return []
    files: list[str] = []
    for fp in STORE_DIR.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            files.append(str(fp.relative_to(STORE_DIR)))
        except ValueError:
            files.append(str(fp))
    return sorted(files)


app = FastAPI(
    title="Satti (Virtual Satellite Imagery Service)",
    description="FastAPI-based mock service serving EO/SAR products (L0~L4).",
    version="0.1.0",
)

CATALOG: list[Product] = []


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/ui", "/"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def startup_event() -> None:
    global CATALOG
    CATALOG = _load_catalog()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "satti", "products": len(CATALOG)}


@app.get("/products")
def list_products(
    sensor: Optional[str] = Query(default=None, pattern="^(eo|sar)$"),
    level: Optional[str] = Query(default=None, pattern="^L[0-4]$"),
    fmt: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, description="Search in product_id/summary"),
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
            if qq in x.product_id.lower() or qq in x.summary.lower() or qq in x.satellite.lower()
        ]

    return {
        "count": len(items),
        "items": [asdict(x) for x in items],
    }


@app.get("/products/{product_id}")
def get_product(product_id: str) -> dict:
    for p in CATALOG:
        if p.product_id == product_id:
            return asdict(p)
    raise HTTPException(status_code=404, detail="Product not found")


@app.get("/products/{product_id}/download")
def download_product(product_id: str) -> FileResponse:
    for p in CATALOG:
        if p.product_id != product_id:
            continue
        fp = Path(p.path)
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Product file missing")
        return FileResponse(
            path=fp,
            media_type="application/octet-stream",
            filename=p.file_name,
        )
    raise HTTPException(status_code=404, detail="Product not found")


@app.get("/products/{product_id}/content")
def view_product_content(product_id: str) -> FileResponse:
    for p in CATALOG:
        if p.product_id != product_id:
            continue
        fp = Path(p.path)
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Product file missing")
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
    raise HTTPException(status_code=404, detail="Product not found")


@app.get("/admin/mock-store/info")
def mock_store_info() -> dict:
    image_folders = _collect_image_folders(CATALOG)
    image_files = _collect_image_files(CATALOG)
    return {
        "store_dir": str(STORE_DIR),
        "catalog_path": str(CATALOG_PATH),
        "disabled": MOCK_STORE_DISABLED_FLAG.exists(),
        "exists": STORE_DIR.exists(),
        "product_count": len(CATALOG),
        "image_folders": image_folders,
        "image_files": image_files,
    }


@app.post("/admin/mock-store/rebuild")
def rebuild_mock_store() -> dict:
    global CATALOG
    if MOCK_STORE_DISABLED_FLAG.exists():
        MOCK_STORE_DISABLED_FLAG.unlink()
    CATALOG = _build_mock_store(force_rebuild=True)
    return {
        "ok": True,
        "message": "mock store rebuilt",
        "product_count": len(CATALOG),
        "image_folders": _collect_image_folders(CATALOG),
    }


@app.post("/admin/mock-store/delete")
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


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Satti Product Explorer</title>
  <style>
    :root {
      --bg: #f4f8fb;
      --panel: #ffffff;
      --line: #d6e1eb;
      --text: #17212b;
      --muted: #5d6b79;
      --accent: #0a7f5a;
      --accent-2: #0b5ad4;
      --warn: #bc4f00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 100% 0%, #dff1ef, #f4f8fb 35%);
      color: var(--text);
      font: 14px/1.5 "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", sans-serif;
    }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0; font-size: 28px; letter-spacing: -0.2px; }
    h2 { margin: 0 0 10px; font-size: 17px; color: var(--accent); }
    .intro { margin: 6px 0 16px; color: var(--muted); }
    .layout {
      display: grid;
      gap: 14px;
      grid-template-columns: 1.05fr 1fr;
      align-items: start;
    }
    @media (max-width: 1024px) { .layout { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 2px 10px rgba(18, 35, 57, 0.05);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    @media (max-width: 720px) { .stats { grid-template-columns: repeat(2, 1fr); } }
    .chip {
      border: 1px solid var(--line);
      background: #f8fcff;
      border-radius: 10px;
      padding: 8px;
    }
    .chip .k { color: var(--muted); font-size: 12px; }
    .chip .v { font-size: 19px; font-weight: 700; margin-top: 1px; }
    .admin-box {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #f9fcff;
    }
    .folder-list {
      margin-top: 8px;
      max-height: 140px;
      overflow: auto;
      border: 1px solid #dbe8f4;
      border-radius: 8px;
      background: #fff;
      padding: 8px 10px;
      font-size: 12px;
      color: #2a4662;
    }
    .folder-item { padding: 2px 0; }

    .row { display: flex; gap: 8px; flex-wrap: wrap; }
    label { color: var(--muted); font-size: 12px; display: block; margin-bottom: 4px; }
    input, select, button {
      background: #fff;
      color: var(--text);
      border: 1px solid #c8d6e4;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
    }
    input, select { min-width: 130px; }
    button {
      cursor: pointer;
      font-weight: 600;
      border-color: #95b8ec;
      background: linear-gradient(180deg, #f0f7ff, #dcecff);
      color: #0a3f93;
    }
    button.secondary {
      border-color: #d3dce6;
      background: linear-gradient(180deg, #fff, #f3f6f9);
      color: #2a3948;
    }
    button.ghost {
      background: transparent;
      border-color: #cdd9e5;
      color: #2b4d71;
    }

    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; min-width: 760px; }
    th, td { border-bottom: 1px solid #e4edf6; padding: 9px 10px; text-align: left; vertical-align: top; }
    th { background: #f6fbff; color: #2f4d6a; font-size: 12px; }
    td small { color: var(--muted); }

    .products-list {
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f9fcff;
      padding: 8px;
      margin-top: 8px;
    }
    .products-list button {
      width: 100%;
      text-align: left;
      margin-bottom: 6px;
      border: 1px solid #cfdeed;
      background: #fff;
      color: #17334e;
    }
    .products-list button:hover { border-color: #97b6d8; }

    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 10px;
      margin-top: 8px;
      font-size: 13px;
    }
    .kv .k { color: var(--muted); }
    .pill {
      display: inline-block;
      border: 1px solid #bdd6ef;
      background: #eef6ff;
      color: #244c76;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      margin-right: 6px;
      margin-bottom: 6px;
    }
    .preview-box {
      margin-top: 10px;
      border: 1px solid #dbe8f4;
      border-radius: 10px;
      background: #f9fcff;
      padding: 10px;
    }
    .preview-box img {
      max-width: 100%;
      max-height: 320px;
      display: block;
      border: 1px solid #cfdceb;
      border-radius: 8px;
      background: #fff;
    }
    .steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 8px; }
    .step {
      text-align: center;
      border: 1px solid #d4e2ef;
      border-radius: 8px;
      padding: 6px 4px;
      color: #476278;
      background: #f7fbff;
      font-size: 12px;
      font-weight: 600;
    }
    .step.on { border-color: #7eb3ea; color: #0e5db2; background: #e9f4ff; }
    .muted { color: var(--muted); }
    .warn { color: var(--warn); font-weight: 700; }
    pre {
      margin: 8px 0 0;
      padding: 10px;
      background: #f7fbff;
      border: 1px solid #dbe8f4;
      border-radius: 8px;
      max-height: 210px;
      overflow: auto;
      font-size: 12px;
      color: #27415b;
    }
    a { color: var(--accent-2); text-decoration: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Satti Product Explorer</h1>
    <p class="intro">레벨(L0~L4)별로 영상 정보가 어떻게 확장되는지와, 현재 제품에 포함된 부가정보(좌표, RPC/DEM, 타일, 분류 레이어)를 한 화면에서 확인할 수 있습니다.</p>

    <section class="card">
      <div class="row" style="justify-content:space-between; align-items:center;">
        <h2>시스템 상태</h2>
        <button class="ghost" onclick="callHealth()">상태 새로고침</button>
      </div>
      <div id="healthBanner" class="muted">연결 확인 중...</div>
      <div class="stats">
        <div class="chip"><div class="k">전체</div><div class="v" id="sTotal">-</div></div>
        <div class="chip"><div class="k">L0</div><div class="v" id="sL0">-</div></div>
        <div class="chip"><div class="k">L1</div><div class="v" id="sL1">-</div></div>
        <div class="chip"><div class="k">L2</div><div class="v" id="sL2">-</div></div>
        <div class="chip"><div class="k">L3/L4</div><div class="v" id="sL34">-</div></div>
      </div>
      <div class="admin-box">
        <div class="row">
          <button onclick="rebuildSamples()">샘플 생성</button>
          <button class="secondary" onclick="deleteSamples()">샘플 모두 삭제</button>
          <button class="ghost" onclick="loadMockStoreInfo()">폴더 목록 새로고침</button>
        </div>
        <div class="muted" id="adminStatus" style="margin-top:6px;">샘플 스토어 정보를 불러오는 중...</div>
        <div class="folder-list" id="folderList">-</div>
      </div>
    </section>

    <div class="layout" style="margin-top:14px;">
      <section class="card">
        <h2>위성 센서 및 영상레벨 선택</h2>
        <div class="row">
          <div>
            <label>sensor</label>
            <select id="sensor">
              <option value="">(all)</option>
              <option value="eo">eo</option>
              <option value="sar">sar</option>
            </select>
          </div>
          <div>
            <label>level</label>
            <select id="level">
              <option value="">(all)</option>
              <option value="L0">L0</option>
              <option value="L1">L1</option>
              <option value="L2">L2</option>
              <option value="L3">L3</option>
              <option value="L4">L4</option>
            </select>
          </div>
          <div>
            <label>fmt</label>
            <select id="fmt">
              <option value="">(all)</option>
              <option value="geotiff">geotiff</option>
              <option value="ceos">ceos</option>
              <option value="tiled-geotiff">tiled-geotiff</option>
              <option value="index-map">index-map</option>
              <option value="classified-raster">classified-raster</option>
            </select>
          </div>
          <div>
            <label>q</label>
            <input id="q" placeholder="satellite / summary 검색" />
          </div>
        </div>
        <div class="row" style="margin-top:8px;">
          <button onclick="callProducts()">/products 조회</button>
          <button class="secondary" onclick="resetFilters()">필터 초기화</button>
        </div>
        <div class="products-list" id="productsList"></div>

        <h2 style="margin-top:14px;">레벨별 정보 비교</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Level</th><th>대표 포맷</th><th>이미지/신호 상태</th><th>추가되는 부가정보</th><th>주요 활용</th>
              </tr>
            </thead>
            <tbody id="levelGuideBody"></tbody>
          </table>
        </div>
      </section>

      <section class="card">
        <h2>선택 제품 상세 해설</h2>
        <div class="row">
          <div style="flex:1; min-width:200px;">
            <label>product_id</label>
            <input id="pid" style="width:100%;" placeholder="목록에서 자동 선택됩니다." />
          </div>
          <div style="display:flex; align-items:flex-end;">
            <button onclick="callDetail()">상세 조회</button>
          </div>
        </div>

        <div class="steps" id="levelSteps">
          <div class="step" data-level="L0">L0</div>
          <div class="step" data-level="L1">L1</div>
          <div class="step" data-level="L2">L2</div>
          <div class="step" data-level="L3">L3</div>
          <div class="step" data-level="L4">L4</div>
        </div>

        <div class="kv" id="productKv"></div>
        <div style="margin-top:8px;">
          <div class="muted" style="font-size:12px; margin-bottom:6px;">이 레벨에서 포함되는 부가정보</div>
          <div id="extraInfo"></div>
        </div>
        <div style="margin-top:8px;">
          <div class="muted" style="font-size:12px; margin-bottom:6px;">권장 활용/뷰어</div>
          <div id="usageInfo"></div>
        </div>

        <div class="row" style="margin-top:10px;">
          <button class="secondary" onclick="callDownload()">파일 다운로드</button>
        </div>
        <div id="downloadHint" class="muted" style="margin-top:6px;">선택한 제품을 원본 파일로 내려받습니다.</div>

        <h2 style="margin-top:14px;">Raw API 응답</h2>
        <pre id="detailOut">ready</pre>
        <div class="preview-box">
          <div class="muted" style="font-size:12px; margin-bottom:6px;">웹 미리보기</div>
          <div id="previewWrap" class="muted">선택 제품 없음</div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"];
    const LEVEL_GUIDE = {
      L0: {
        formats: "Raw binary, CEOS",
        imageState: "원시 텔레메트리/신호, 사람 판독 어려움",
        extras: ["패킷/라인 헤더", "센서 수집 시각", "기초 획득 메타"],
        usage: "복원/재처리 입력"
      },
      L1: {
        formats: "GeoTIFF, CEOS L1A",
        imageState: "기초 보정된 영상(기하/방사 보정)",
        extras: ["밴드/비트심도", "기본 보정 파라미터", "센서/촬영 메타"],
        usage: "전문 분석 시작점"
      },
      L2: {
        formats: "GeoTIFF (+RPC/DEM)",
        imageState: "지리 좌표계 정합된 분석용 영상",
        extras: ["좌표계/해상도", "RPC 계수", "DEM 연계 정보"],
        usage: "측정/정밀 분석"
      },
      L3: {
        formats: "Mosaic GeoTIFF, Tiles",
        imageState: "서비스용 모자이크/타일 레이어",
        extras: ["타일 인덱스", "모자이크 경계", "표시 최적화 정보"],
        usage: "지도 서비스/웹 제공"
      },
      L4: {
        formats: "Classified raster, index map",
        imageState: "지수/분류 결과(해석 산출물)",
        extras: ["분류 코드 체계", "지수 값 범위", "판정 기준 정보"],
        usage: "의사결정/리포트"
      }
    };

    function pretty(obj) { return JSON.stringify(obj, null, 2); }
    function toNum(v) { return Number.isFinite(v) ? v : 0; }
    function bytesToLabel(n) {
      const b = Number(n || 0);
      if (b < 1024) return `${b} B`;
      if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
      return `${(b / (1024 * 1024)).toFixed(2)} MB`;
    }

    function buildQuery(params) {
      const q = new URLSearchParams();
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null && String(v).trim() !== "") q.set(k, String(v).trim());
      });
      return q.toString();
    }

    function renderLevelGuide() {
      const body = $("levelGuideBody");
      body.innerHTML = "";
      LEVEL_ORDER.forEach((lv) => {
        const g = LEVEL_GUIDE[lv];
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${lv}</strong></td>
          <td>${g.formats}</td>
          <td>${g.imageState}</td>
          <td>${g.extras.map((x) => `<small>${x}</small>`).join("<br/>")}</td>
          <td>${g.usage}</td>
        `;
        body.appendChild(tr);
      });
    }

    function setStep(level) {
      document.querySelectorAll(".step").forEach((x) => {
        if (x.dataset.level === level) x.classList.add("on");
        else x.classList.remove("on");
      });
    }

    function renderTags(elId, items) {
      const el = $(elId);
      if (!items || !items.length) {
        el.innerHTML = '<span class="warn">정보 없음</span>';
        return;
      }
      el.innerHTML = items.map((x) => `<span class="pill">${x}</span>`).join("");
    }

    function renderSelectedProduct(p) {
      if (!p) {
        $("productKv").innerHTML = '<div class="k">상태</div><div class="warn">선택된 제품이 없습니다.</div>';
        renderTags("extraInfo", []);
        renderTags("usageInfo", []);
        $("previewWrap").innerHTML = '<span class="warn">미리보기할 제품이 없습니다.</span>';
        return;
      }
      $("pid").value = p.product_id;
      $("downloadHint").innerHTML = `선택된 파일: <strong>${p.file_name}</strong> (${bytesToLabel(p.file_size_bytes)})`;
      setStep(p.level);
      const guide = LEVEL_GUIDE[p.level] || { extras: [], usage: "" };
      $("productKv").innerHTML = `
        <div class="k">Product ID</div><div>${p.product_id}</div>
        <div class="k">Sensor / Level</div><div>${p.sensor.toUpperCase()} / ${p.level}</div>
        <div class="k">Format</div><div>${p.fmt}</div>
        <div class="k">Satellite</div><div>${p.satellite}</div>
        <div class="k">Acquired(UTC)</div><div>${p.acquired_at_utc}</div>
        <div class="k">Summary</div><div>${p.summary}</div>
      `;
      const viewerHint = p.level === "L0"
        ? ["일반 이미지 뷰어 비권장", "신호/아카이브 전용 도구 필요"]
        : ["GIS 툴(QGIS 등) 권장", "웹 지도/타일 연계 가능"];
      renderTags("extraInfo", guide.extras);
      renderTags("usageInfo", [guide.usage, ...viewerHint]);

      const name = String(p.file_name || "").toLowerCase();
      const webExt = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".tif", ".tiff"];
      const displayable = webExt.some((ext) => name.endsWith(ext));
      if (!displayable) {
        $("previewWrap").innerHTML = `<span class="warn">이 파일은 브라우저 직접 미리보기 대상이 아닙니다.</span> <span class="muted">(${p.file_name})</span>`;
      } else {
        const src = `/products/${encodeURIComponent(p.product_id)}/content`;
        $("previewWrap").innerHTML = `<img src="${src}" alt="${p.product_id}" />`;
      }
    }

    function renderProductsList(items) {
      const wrap = $("productsList");
      if (!items.length) {
        wrap.innerHTML = '<span class="warn">검색 결과가 없습니다.</span>';
        renderSelectedProduct(null);
        return;
      }
      wrap.innerHTML = "";
      items.forEach((x) => {
        const b = document.createElement("button");
        b.className = "secondary";
        b.textContent = `${x.product_id} | ${x.sensor}/${x.level} | ${x.fmt}`;
        b.onclick = () => renderSelectedProduct(x);
        wrap.appendChild(b);
      });
      renderSelectedProduct(items[0]);
    }

    function updateStats(items) {
      const by = { L0: 0, L1: 0, L2: 0, L3: 0, L4: 0 };
      items.forEach((x) => { by[x.level] = toNum(by[x.level]) + 1; });
      $("sTotal").textContent = items.length;
      $("sL0").textContent = by.L0;
      $("sL1").textContent = by.L1;
      $("sL2").textContent = by.L2;
      $("sL34").textContent = by.L3 + by.L4;
    }

    async function getJson(url, options = {}) {
      const res = await fetch(url, options);
      const text = await res.text();
      let body;
      try { body = JSON.parse(text); } catch { body = text; }
      return { res, body };
    }

    async function callHealth() {
      try {
        const { body } = await getJson("/health");
        $("healthBanner").textContent = `서비스 상태: ${body.status} / 제품 ${body.products}건`;
      } catch (e) {
        $("healthBanner").innerHTML = `<span class="warn">상태 조회 실패: ${String(e)}</span>`;
      }
    }

    async function loadMockStoreInfo() {
      try {
        const { body } = await getJson("/admin/mock-store/info");
        const folders = body.image_folders || [];
        const files = body.image_files || [];
        const locked = body.disabled ? " / 삭제 잠금 상태" : "";
        $("adminStatus").textContent = `스토어: ${body.store_dir} / 이미지 폴더 ${folders.length}개 / 이미지 파일 ${files.length}개${locked}`;
        if (!files.length) {
          $("folderList").innerHTML = '<span class="warn">생성된 이미지 파일이 없습니다.</span>';
          return;
        }
        $("folderList").innerHTML = files.map((f) => `<div class="folder-item">${f}</div>`).join("");
      } catch (e) {
        $("adminStatus").innerHTML = `<span class="warn">스토어 정보 조회 실패: ${String(e)}</span>`;
      }
    }

    async function rebuildSamples() {
      try {
        const { res, body } = await getJson("/admin/mock-store/rebuild", { method: "POST" });
        if (!res.ok) {
          $("detailOut").textContent = pretty({ status: res.status, body });
          return;
        }
        $("detailOut").textContent = pretty({ status: res.status, body });
        await callHealth();
        await callProducts();
        await loadMockStoreInfo();
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e) });
      }
    }

    async function deleteSamples() {
      const ok = window.confirm("mock_store 아래 생성 샘플을 모두 삭제합니다. 계속할까요?");
      if (!ok) return;
      try {
        const { res, body } = await getJson("/admin/mock-store/delete", { method: "POST" });
        if (!res.ok) {
          $("detailOut").textContent = pretty({ status: res.status, body });
          return;
        }
        $("detailOut").textContent = pretty({ status: res.status, body });
        $("productsList").innerHTML = '<span class="warn">샘플이 삭제되었습니다.</span>';
        $("productKv").innerHTML = '<div class="k">상태</div><div class="warn">샘플 없음</div>';
        $("folderList").innerHTML = '<span class="warn">생성된 이미지 파일이 없습니다.</span>';
        await callHealth();
        await callProducts();
        await loadMockStoreInfo();
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e) });
      }
    }

    function resetFilters() {
      $("sensor").value = "";
      $("level").value = "";
      $("fmt").value = "";
      $("q").value = "";
      callProducts();
    }

    async function callProducts() {
      const qs = buildQuery({
        sensor: $("sensor").value,
        level: $("level").value,
        fmt: $("fmt").value,
        q: $("q").value
      });
      const url = "/products" + (qs ? ("?" + qs) : "");
      try {
        const { body } = await getJson(url);
        const items = body.items || [];
        updateStats(items);
        renderProductsList(items);
      } catch (e) {
        $("productsList").innerHTML = `<span class="warn">목록 조회 실패: ${String(e)}</span>`;
      }
    }

    async function callDetail() {
      const pid = $("pid").value.trim();
      if (!pid) {
        $("detailOut").textContent = pretty({ error: "product_id required" });
        return;
      }
      try {
        const { res, body } = await getJson(`/products/${encodeURIComponent(pid)}`);
        $("detailOut").textContent = pretty({ status: res.status, body });
        if (res.ok) renderSelectedProduct(body);
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e) });
      }
    }

    async function callDownload() {
      const pid = $("pid").value.trim();
      if (!pid) {
        $("detailOut").textContent = pretty({ error: "product_id required" });
        return;
      }
      const url = `/products/${encodeURIComponent(pid)}/download`;
      try {
        const res = await fetch(url);
        if (!res.ok) {
          const text = await res.text();
          $("detailOut").textContent = pretty({ status: res.status, error: text });
          return;
        }
        const blob = await res.blob();
        const link = document.createElement("a");
        const href = URL.createObjectURL(blob);
        link.href = href;
        const cd = res.headers.get("content-disposition") || "";
        const m = cd.match(/filename="?([^"]+)"?/);
        link.download = m ? m[1] : `${pid}.bin`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(href);
        $("detailOut").textContent = pretty({
          status: res.status,
          downloaded_bytes: blob.size,
          filename: link.download,
          endpoint: url
        });
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e), endpoint: url });
      }
    }

    renderLevelGuide();
    callHealth();
    callProducts();
    loadMockStoreInfo();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return '<html><body style="font-family:sans-serif;padding:20px;"><h2>Satti API</h2><p><a href="/ui">Open Test UI</a></p><p><a href="/docs">Open Swagger Docs</a></p></body></html>'

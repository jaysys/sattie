#!/usr/bin/env python3
"""
Satti (virtual) - FastAPI service for serving satellite imagery images.

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

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query
from fastapi.responses import FileResponse, HTMLResponse, Response


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STORE_DIR = PROJECT_DIR / "mock_store"
CATALOG_PATH = STORE_DIR / "catalog.json"
MOCK_STORE_DISABLED_FLAG = PROJECT_DIR / ".mock_store_disabled"
MOCK_STORE_VERSION = 3
GENERATED_DIR = STORE_DIR / "generated"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ImageItem:
    image_id: str
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


def _make_l3_tiles(sensor: str, image_slug: str, seed: int = 42) -> list[Path]:
    """
    Generate simple 2x2 tile set for L3 preview/use-case demonstration.
    """
    tiles_dir = STORE_DIR / sensor / "L3" / "tiles" / image_slug
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


def _build_mock_store(force_rebuild: bool = False) -> list[ImageItem]:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    seed_rng = random.SystemRandom()
    scene_seed = {
        "eo": seed_rng.randrange(1, 2**31),
        "sar": seed_rng.randrange(1, 2**31),
    }

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
        sensor = "eo" if key.startswith("eo_") else "sar"
        file_seed = scene_seed[sensor]
        if fp.suffix == ".tif":
            _make_dummy_image(fp, width=512, height=512, seed=file_seed, sensor=sensor)
        else:
            _make_dummy_bin(fp, size_bytes=1024 * 1024, seed=file_seed)

    # L3 service tiles (mosaic derivatives)
    _make_l3_tiles(
        sensor="eo",
        image_slug="eo-kompsat3-l3-mosaic001",
        seed=scene_seed["eo"],
    )
    _make_l3_tiles(
        sensor="sar",
        image_slug="sar-kompsat5-l3-mosaic001",
        seed=scene_seed["sar"],
    )

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    catalog = [
        ImageItem(
            image_id="eo-kompsat3-l0-raw001",
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
        ImageItem(
            image_id="eo-kompsat3-l1-scene001",
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
        ImageItem(
            image_id="eo-kompsat3-l2-scene001",
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
        ImageItem(
            image_id="eo-kompsat3-l3-mosaic001",
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
        ImageItem(
            image_id="eo-kompsat3-l4-index001",
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
        ImageItem(
            image_id="sar-kompsat5-l0-raw001",
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
        ImageItem(
            image_id="sar-kompsat5-l1-intensity001",
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
        ImageItem(
            image_id="sar-kompsat5-l2-geocoded001",
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
        ImageItem(
            image_id="sar-kompsat5-l3-mosaic001",
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
        ImageItem(
            image_id="sar-kompsat5-l4-change001",
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
        "images": [asdict(x) for x in catalog],
    }
    CATALOG_PATH.write_text(json.dumps(write_obj, indent=2, ensure_ascii=False))
    return catalog


def _load_catalog() -> list[ImageItem]:
    # If disabled flag exists, never auto-rebuild; rebuild only via explicit admin action.
    if MOCK_STORE_DISABLED_FLAG.exists():
        return []
    if not CATALOG_PATH.exists():
        return _build_mock_store()
    obj = read_json(CATALOG_PATH)
    raw_images = obj.get("images")
    if raw_images is None:
        # Backward compatibility with older catalog schema.
        raw_images = obj.get("products")
    if not isinstance(raw_images, list):
        return _build_mock_store(force_rebuild=True)

    images: list[ImageItem] = []
    for raw in raw_images:
        if not isinstance(raw, dict):
            return _build_mock_store(force_rebuild=True)
        item = dict(raw)
        if "image_id" not in item and "product_id" in item:
            item["image_id"] = item["product_id"]
        item.pop("product_id", None)
        try:
            images.append(ImageItem(**item))
        except TypeError:
            return _build_mock_store(force_rebuild=True)

    # Keep catalog aligned with current mock model (EO L0 included).
    if obj.get("mock_store_version") != MOCK_STORE_VERSION:
        return _build_mock_store(force_rebuild=True)
    if not any(p.sensor == "eo" and p.level == "L0" for p in images):
        return _build_mock_store(force_rebuild=True)
    if not any(p.sensor == "sar" and p.level == "L3" for p in images):
        return _build_mock_store(force_rebuild=True)
    return images


def _collect_image_folders(images: list[ImageItem]) -> list[str]:
    if not STORE_DIR.exists():
        return []
    folders: set[str] = set()
    for fp in STORE_DIR.rglob("*"):
        if not fp.is_file():
            continue
        try:
            folders.add(str(fp.parent.relative_to(STORE_DIR)))
        except ValueError:
            folders.add(str(fp.parent))
    return sorted(folders)


def _collect_image_files(images: list[ImageItem]) -> list[str]:
    if not STORE_DIR.exists():
        return []
    files: list[str] = []
    for fp in STORE_DIR.rglob("*"):
        if not fp.is_file():
            continue
        try:
            files.append(str(fp.relative_to(STORE_DIR)))
        except ValueError:
            files.append(str(fp))
    return sorted(files)


def _generated_suffix_for_format(fmt: str) -> str:
    normalized = fmt.strip().lower()
    if normalized in {"geotiff", "tiled-geotiff", "index-map", "classified-raster"}:
        return ".tif"
    if normalized == "ceos":
        return ".bin"
    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported fmt. allowed: ceos, geotiff, tiled-geotiff, "
            "index-map, classified-raster"
        ),
    )


def _validate_generate_combo(sensor: str, level: str, fmt: str) -> str:
    sensor_norm = sensor.strip().lower()
    level_norm = level.strip().upper()
    fmt_norm = fmt.strip().lower()
    allowed_map = {
        ("eo", "L0"): {"ceos"},
        ("eo", "L1"): {"geotiff"},
        ("eo", "L2"): {"geotiff"},
        ("eo", "L3"): {"tiled-geotiff"},
        ("eo", "L4"): {"index-map"},
        ("sar", "L0"): {"ceos"},
        ("sar", "L1"): {"geotiff"},
        ("sar", "L2"): {"geotiff"},
        ("sar", "L3"): {"tiled-geotiff"},
        ("sar", "L4"): {"classified-raster"},
    }
    allowed = allowed_map.get((sensor_norm, level_norm))
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported sensor/level: {sensor_norm}/{level_norm}")
    if fmt_norm not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid fmt for {sensor_norm}/{level_norm}. "
                f"allowed: {allowed_list}; requested: {fmt_norm}"
            ),
        )
    return fmt_norm


app = FastAPI(
    title="Satti (Virtual Satellite Imagery Service)",
    description=(
        "EO/SAR 위성영상 샘플 데이터를 L0~L4 레벨로 제공하는 Mock API입니다.\n\n"
        "주요 목적:\n"
        "- 이미지 목록/상세 조회 및 파일 다운로드\n"
        "- 브라우저 미리보기를 위한 콘텐츠 엔드포인트 제공\n"
        "- mock_store 샘플 데이터 생성/삭제/상태 확인\n\n"
        "참고:\n"
        "- L0~L4는 동일 소스 장면을 기반으로 포맷/가공 수준만 달라집니다.\n"
        "- 샘플 전체 삭제 후 재생성 시 새로운 랜덤 시드가 적용되어 다른 장면이 만들어집니다."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "service", "description": "서비스 상태 점검 및 기본 헬스체크 API"},
        {"name": "images", "description": "이미지 목록/상세 조회, 원본 다운로드, 미리보기 콘텐츠 API"},
        {"name": "mock-store", "description": "mock_store 샘플 생성/삭제/상태 관리 API"},
        {"name": "ui", "description": "브라우저용 관리/탐색 UI API"},
    ],
)

CATALOG: list[ImageItem] = []


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


@app.get(
    "/health",
    tags=["service"],
    summary="서비스 상태 확인",
    description="API 서버 기동 상태와 현재 메모리에 로드된 샘플 이미지 개수를 반환합니다.",
    response_description="서비스 상태 정보(status/service/images)",
)
def health() -> dict:
    return {"status": "ok", "service": "satti", "images": len(CATALOG)}


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
        "- 생성된 파일은 `mock_store/generated/{sensor}/{level}` 경로에 저장됩니다.\n"
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

    out_dir = GENERATED_DIR / sensor / level
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
    image_folders = _collect_image_folders(CATALOG)
    image_files = _collect_image_files(CATALOG)
    return {
        "store_dir": str(STORE_DIR),
        "catalog_path": str(CATALOG_PATH),
        "disabled": MOCK_STORE_DISABLED_FLAG.exists(),
        "exists": STORE_DIR.exists(),
        "image_count": len(CATALOG),
        "image_folders": image_folders,
        "image_files": image_files,
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
    "/ui",
    response_class=HTMLResponse,
    tags=["ui"],
    summary="웹 UI 페이지",
    description="샘플 생성/조회/다운로드/미리보기를 위한 단일 HTML UI 페이지를 반환합니다.",
    response_description="HTML 문서",
)
def ui() -> str:
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>위성영상 이미지포맷 L0~L4 Explorer</title>
  <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7611240025188124"
     crossorigin="anonymous"></script>
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
    html, body { max-width: 100%; overflow-x: hidden; }
    body {
      margin: 0;
      background: radial-gradient(circle at 100% 0%, #dff1ef, #f4f8fb 35%);
      color: var(--text);
      font: 14px/1.5 "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", sans-serif;
    }
    .wrap { width: 100%; max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0; font-size: 28px; letter-spacing: -0.2px; }
    h2 { margin: 0 0 10px; font-size: 17px; color: var(--accent); }
    .intro { margin: 6px 0 16px; color: var(--muted); }
    .layout {
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr 1fr;
      align-items: start;
    }
    @media (max-width: 1024px) { .layout { grid-template-columns: 1fr; } }
    @media (max-width: 900px) {
      .wrap { padding: 16px; }
      h1 { font-size: 24px; }
      .card { padding: 12px; }
      .row { gap: 6px; }
      .row > * { flex: 1 1 calc(50% - 6px); min-width: 0; }
      input, select { min-width: 0; width: 100%; }
      .filter-row { flex-wrap: wrap; }
      .filter-row > div { flex: 1 1 calc(50% - 6px); }
    }
    @media (max-width: 640px) {
      .wrap { padding: 10px; }
      h1 { font-size: 21px; }
      h2 { font-size: 15px; }
      .card { padding: 10px; border-radius: 10px; }
      .row > * { flex: 1 1 100%; min-width: 0; }
      input, select, button { width: 100%; min-width: 0; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .kv { grid-template-columns: 1fr; gap: 4px; }
      .kv .k { font-weight: 600; }
      .table-wrap { border-radius: 8px; }
      table { min-width: 0; width: 100%; table-layout: fixed; }
      th, td { padding: 7px 8px; }
      th, td { white-space: normal; overflow-wrap: anywhere; word-break: break-word; }
      .folder-list { max-height: 180px; }
      .job-panel { padding: 10px; }
      .job-time { grid-template-columns: 72px 1fr; }
    }
    @media (max-width: 480px) {
      .table-wrap { overflow: hidden; }
      table { min-width: 0; width: 100%; table-layout: fixed; }
      th, td {
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
        font-size: 11px;
      }
      .steps { grid-template-columns: repeat(3, 1fr); }
      .step { font-size: 11px; padding: 5px 3px; }
      .preview-box { padding: 8px; }
      #imageKv > div { overflow-wrap: anywhere; word-break: break-word; }
      #detailOut { font-size: 11px; }
    }
    @media (max-width: 360px) {
      .steps { grid-template-columns: repeat(2, 1fr); }
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 2px 10px rgba(18, 35, 57, 0.05);
      min-width: 0;
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
    .chip .k {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .chip .v { font-size: 19px; font-weight: 700; margin-top: 1px; }
    .admin-box {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #f9fcff;
    }
    .admin-layout {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 10px;
      align-items: start;
    }
    .admin-api {
      border: 1px solid #dbe8f4;
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }
    .admin-api h3 {
      margin: 0;
      font-size: 13px;
      color: #2f4d6a;
    }
    .admin-api pre {
      margin-top: 6px;
      max-height: 196px;
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
    .filter-row { flex-wrap: nowrap; align-items: flex-end; }
    .filter-item-short { flex: 0 0 110px; min-width: 0; }
    .filter-item-long { flex: 1 1 320px; min-width: 0; }
    .filter-row input, .filter-row select { width: 100%; min-width: 0; }
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

    .table-wrap { overflow: hidden; border: 1px solid var(--line); border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; min-width: 0; table-layout: fixed; }
    th, td { border-bottom: 1px solid #e4edf6; padding: 9px 10px; text-align: left; vertical-align: top; }
    th { background: #f6fbff; color: #2f4d6a; font-size: 12px; }
    th:nth-child(1), td:nth-child(1) { width: 8%; }
    th:nth-child(2), td:nth-child(2) { width: 16%; }
    th:nth-child(3), td:nth-child(3) { width: 18%; }
    th:nth-child(4), td:nth-child(4) { width: 22%; }
    th:nth-child(5), td:nth-child(5) { width: 14%; }
    th:nth-child(6), td:nth-child(6) { width: 22%; }
    td small {
      color: inherit;
      font-size: inherit;
      line-height: inherit;
      font-weight: inherit;
    }
    .fmt-tip {
      position: relative;
      display: inline-block;
      cursor: help;
    }
    .fmt-tip::after {
      content: attr(data-tip);
      position: absolute;
      left: 50%;
      top: calc(100% + 8px);
      transform: translateX(-50%);
      min-width: 220px;
      max-width: 320px;
      padding: 7px 9px;
      border: 1px solid #bcdcc7;
      border-radius: 10px;
      background: #f1fbf4;
      color: #214634;
      font-size: 12px;
      line-height: 1.35;
      box-shadow: 0 8px 16px rgba(24, 58, 38, 0.16);
      white-space: normal;
      z-index: 20;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.15s ease;
    }
    .fmt-tip:hover::after { opacity: 1; }

    .images-list {
      max-height: none;
      min-height: 0;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f9fcff;
      padding: 8px;
      margin-top: 8px;
    }
    .images-list button {
      width: 100%;
      text-align: left;
      margin-bottom: 6px;
      border: 1px solid #cfdeed;
      background: #fff;
      color: #17334e;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
      line-height: 1.35;
    }
    .images-list button:hover { border-color: #97b6d8; }

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
    .job-modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(10, 22, 38, 0.45);
      z-index: 9999;
      padding: 16px;
    }
    .job-modal.show { display: flex; }
    .job-panel {
      width: min(460px, 100%);
      border: 1px solid #c7d9ea;
      border-radius: 12px;
      background: #fff;
      box-shadow: 0 16px 36px rgba(8, 25, 44, 0.22);
      padding: 14px;
    }
    .job-head {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #1f456b;
      font-weight: 700;
    }
    .job-spin {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid #9fc5ed;
      border-top-color: #1b6fc3;
      animation: spin 0.9s linear infinite;
    }
    .job-modal.done .job-spin { animation: none; border-color: #6ab48d; border-top-color: #6ab48d; }
    .job-time {
      margin-top: 10px;
      display: grid;
      grid-template-columns: 80px 1fr;
      gap: 6px 10px;
      padding: 8px;
      background: #f6fbff;
      border: 1px solid #dbe8f4;
      border-radius: 8px;
      font-size: 13px;
    }
    .confirm-modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(10, 22, 38, 0.45);
      z-index: 9998;
      padding: 16px;
    }
    .confirm-modal.show { display: flex; }
    .confirm-panel {
      width: min(420px, 100%);
      border: 1px solid #c7d9ea;
      border-radius: 12px;
      background: #fff;
      box-shadow: 0 16px 36px rgba(8, 25, 44, 0.22);
      padding: 14px;
    }
    .confirm-title { color: #1f456b; font-weight: 700; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 900px) {
      .admin-layout { grid-template-columns: 1fr; }
    }
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
      max-width: 100%;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    a { color: var(--accent-2); text-decoration: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>위성영상 이미지포맷 L0~L4 Explorer</h1>
    <p class="intro">레벨(L0~L4)별로 영상이미지 정보가 어떻게 확장되는지와, 선택한 이미지포맷에 추가로 포함된 부가정보(좌표, RPC/DEM, 타일, 분류 레이어)가 뭐가 있는지 한 화면에서 한번에 확인할 수 있습니다.</p>

    <section class="card">
      <div class="row" style="justify-content:space-between; align-items:center;">
        <h2>샘플 LO~L4 파일 현황</h2>
        <button class="ghost" onclick="refreshStatus()">상태/폴더 새로고침</button>
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
          <div class="admin-layout">
            <div>
              <div class="row">
              <button id="rebuildBtn" onclick="rebuildSamples()">샘플위성영상생성</button>
              <button class="secondary" onclick="deleteSamples()">샘플 모두 삭제</button>
            </div>
            <div class="muted" id="adminStatus" style="margin-top:6px;">샘플 스토어 정보를 불러오는 중...</div>
            <div class="folder-list" id="folderList">-</div>
            </div>
            <aside class="admin-api">
              <h3>Raw API 응답</h3>
              <pre id="detailOut">ready</pre>
            </aside>
          </div>
        </div>
    </section>

    <div class="layout" style="margin-top:14px;">
      <section class="card">
        <h2>샘플 이미지 선택</h2>
        <div class="row filter-row">
          <div class="filter-item-short">
            <label>sensor</label>
            <select id="sensor">
              <option value="">(all)</option>
              <option value="eo">eo</option>
              <option value="sar">sar</option>
            </select>
          </div>
          <div class="filter-item-short">
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
          <div class="filter-item-short">
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
          <div class="filter-item-long">
            <label>q</label>
            <input id="q" placeholder="satellite / summary 검색" />
          </div>
        </div>
        <div class="row" style="margin-top:8px;">
          <button onclick="callImages()">선택한 이미지 목록 조회</button>
          <button class="secondary" onclick="resetFilters()">필터 초기화</button>
        </div>
        <div class="images-list" id="imagesList"></div>
      </section>

      <section class="card">
        <h2>선택한 샘플 이미지 상세 내역(설명)</h2>
        <div class="preview-box">
          <div class="muted" style="font-size:12px; margin-bottom:6px;">웹 미리보기</div>
          <div id="previewWrap" class="muted">선택 이미지 없음</div>
        </div>

        <div class="row">
          <div style="flex:1; min-width:0;">
            <label>image_id</label>
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

        <div class="kv" id="imageKv"></div>
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
        <div id="downloadHint" class="muted" style="margin-top:6px;">선택한 이미지를 원본 파일로 내려받습니다.</div>
      </section>
    </div>

    <section class="card" style="margin-top:14px;">
      <h2>이미지 레벨별 정보 비교표</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Level</th><th>대표 포맷</th><th>이미지/신호 상태</th><th>추가되는 부가정보</th><th>주요 활용</th><th>비고/예시</th>
            </tr>
          </thead>
          <tbody id="levelGuideBody"></tbody>
        </table>
      </div>
    </section>
  </div>
  <div id="jobModal" class="job-modal" role="dialog" aria-modal="true" aria-live="polite">
    <div class="job-panel">
      <div class="job-head"><span class="job-spin"></span><span id="jobTitle">백그라운드 작업 진행 중</span></div>
      <div id="jobMsg" class="muted" style="margin-top:6px;">작업이 완료될 때까지 잠시 기다려주세요.</div>
      <div class="job-time">
        <div class="muted">현재 시각</div><div id="jobNow">-</div>
        <div class="muted">경과 시간</div><div id="jobElapsed">00:00</div>
      </div>
      <div class="row" style="margin-top:10px; justify-content:flex-end;">
        <button id="jobCloseBtn" class="secondary" onclick="closeJobModal()" disabled>닫기</button>
      </div>
    </div>
  </div>
  <div id="confirmModal" class="confirm-modal" role="dialog" aria-modal="true">
    <div class="confirm-panel">
      <div class="confirm-title">샘플 삭제 확인</div>
      <div class="muted" style="margin-top:8px;">mock_store 아래 생성 샘플을 모두 삭제합니다. 계속할까요?</div>
      <div class="row" style="margin-top:12px; justify-content:flex-end;">
        <button class="secondary" onclick="resolveDeleteConfirm(false)">취소</button>
        <button onclick="resolveDeleteConfirm(true)">삭제 진행</button>
      </div>
    </div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"];
    let jobTimer = null;
    let jobStartedAt = 0;
    let jobRunning = false;
    let deleteConfirmResolver = null;
    const LEVEL_GUIDE = {
      L0: {
        formats: "Raw binary, CEOS",
        formatTooltip: "Raw binary: 센서 원시신호 / CEOS: 위성원시자료 교환 포맷",
        imageState: "원시 텔레메트리/신호, 사람 판독 어려움",
        extras: ["패킷/라인 헤더", "센서 수집 시각", "기초 획득 메타"],
        usage: "복원/재처리 입력",
        remarks: "다른 이미지 포맷 적용 전 단계"
      },
      L1: {
        formats: "GeoTIFF, CEOS L1A",
        formatTooltip: "GeoTIFF: 공간좌표 포함 래스터 / CEOS L1A: 기초 보정 단계 위성자료",
        imageState: "기초 보정된 영상(기하/방사 보정)",
        extras: ["밴드/비트심도", "기본 보정 파라미터", "센서/촬영 메타"],
        usage: "전문 분석 시작점",
        remarks: "기타 포맷: JPEG2000, NITF"
      },
      L2: {
        formats: "GeoTIFF (+RPC/DEM)",
        formatTooltip: "GeoTIFF + RPC/DEM: 정밀 지오리퍼런싱/정사보정에 필요한 보조정보 포함",
        imageState: "지리 좌표계 정합된 분석용 영상",
        extras: ["좌표계/해상도", "RPC 계수", "DEM 연계 정보"],
        usage: "측정/정밀 분석",
        remarks: "기타 포맷: COG, NetCDF, HDF5"
      },
      L3: {
        formats: "Mosaic GeoTIFF, Tiles",
        formatTooltip: "Mosaic: 여러 장면 결합 영상 / Tiles: 웹 지도용 타일 분할 포맷",
        imageState: "서비스용 모자이크/타일 레이어",
        extras: ["타일 인덱스", "모자이크 경계", "표시 최적화 정보"],
        usage: "지도 서비스/웹 제공",
        remarks: "기타 포맷: MBTiles, WMTS, XYZ Tiles"
      },
      L4: {
        formats: "Classified raster, index map",
        formatTooltip: "Classified raster: 분류코드 래스터 / index map: 지수값 기반 주제도",
        imageState: "지수/분류 결과(해석 산출물)",
        extras: ["분류 코드 체계", "지수 값 범위", "판정 기준 정보"],
        usage: "의사결정/리포트",
        remarks: "기타 포맷: GeoPackage, GeoJSON, CSV"
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

    function formatElapsed(ms) {
      const total = Math.max(0, Math.floor(ms / 1000));
      const mm = String(Math.floor(total / 60)).padStart(2, "0");
      const ss = String(total % 60).padStart(2, "0");
      return `${mm}:${ss}`;
    }

    function tickJobClock() {
      const now = new Date();
      $("jobNow").textContent = now.toLocaleTimeString("ko-KR", { hour12: false });
      $("jobElapsed").textContent = formatElapsed(Date.now() - jobStartedAt);
    }

    function openJobModal(title, msg) {
      jobRunning = true;
      jobStartedAt = Date.now();
      $("jobTitle").textContent = title;
      $("jobMsg").textContent = msg;
      $("jobCloseBtn").disabled = true;
      $("jobModal").classList.remove("done");
      $("jobModal").classList.add("show");
      tickJobClock();
      clearInterval(jobTimer);
      jobTimer = setInterval(tickJobClock, 1000);
      $("rebuildBtn").disabled = true;
    }

    function finishJobModal(ok, msg) {
      jobRunning = false;
      clearInterval(jobTimer);
      jobTimer = null;
      tickJobClock();
      $("jobTitle").textContent = ok ? "샘플 생성 완료" : "샘플 생성 실패";
      $("jobMsg").textContent = msg;
      $("jobCloseBtn").disabled = false;
      $("jobModal").classList.add("done");
      $("rebuildBtn").disabled = false;
    }

    function closeJobModal() {
      if (jobRunning) return;
      $("jobModal").classList.remove("show");
    }

    function openDeleteConfirm() {
      return new Promise((resolve) => {
        deleteConfirmResolver = resolve;
        $("confirmModal").classList.add("show");
      });
    }

    function resolveDeleteConfirm(ok) {
      $("confirmModal").classList.remove("show");
      if (deleteConfirmResolver) {
        const done = deleteConfirmResolver;
        deleteConfirmResolver = null;
        done(Boolean(ok));
      }
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
          <td><span class="fmt-tip" data-tip="${g.formatTooltip || g.formats}">${g.formats}</span></td>
          <td>${g.imageState}</td>
          <td>${g.extras.map((x) => `<small>${x}</small>`).join("<br/>")}</td>
          <td>${g.usage}</td>
          <td>${g.remarks || "-"}</td>
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

    function renderSelectedImage(p) {
      if (!p) {
        $("imageKv").innerHTML = '<div class="k">상태</div><div class="warn">선택된 이미지가 없습니다.</div>';
        renderTags("extraInfo", []);
        renderTags("usageInfo", []);
        $("previewWrap").innerHTML = '<span class="warn">미리보기할 이미지가 없습니다.</span>';
        return;
      }
      $("pid").value = p.image_id;
      $("downloadHint").innerHTML = `선택된 파일: <strong>${p.file_name}</strong> (${bytesToLabel(p.file_size_bytes)})`;
      setStep(p.level);
      const guide = LEVEL_GUIDE[p.level] || { extras: [], usage: "" };
      $("imageKv").innerHTML = `
        <div class="k">Image ID</div><div>${p.image_id}</div>
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
        $("previewWrap").innerHTML = `<span class="warn">이 파일은 브라우저에서 직접 미리보기가 안됩니다.</span> <span class="muted">(${p.file_name})</span>`;
      } else {
        const src = `/images/${encodeURIComponent(p.image_id)}/content`;
        $("previewWrap").innerHTML = `<img src="${src}" alt="${p.image_id}" />`;
      }
    }

    function renderImagesList(items) {
      const wrap = $("imagesList");
      if (!items.length) {
        wrap.innerHTML = '<span class="warn">검색 결과가 없습니다.</span>';
        wrap.style.maxHeight = "none";
        renderSelectedImage(null);
        return;
      }
      wrap.innerHTML = "";
      items.forEach((x) => {
        const b = document.createElement("button");
        b.className = "secondary";
        b.textContent = `${x.image_id} | ${x.sensor}/${x.level} | ${x.fmt}`;
        b.onclick = () => renderSelectedImage(x);
        wrap.appendChild(b);
      });
      const first = wrap.querySelector("button");
      if (first && items.length > 10) {
        const st = window.getComputedStyle(first);
        const mb = parseFloat(st.marginBottom || "0");
        const mt = parseFloat(st.marginTop || "0");
        const itemH = first.getBoundingClientRect().height + mb + mt;
        const panelPad = 16; // images-list vertical padding (8px * 2)
        wrap.style.maxHeight = `${Math.ceil(itemH * 10 + panelPad)}px`;
      } else {
        wrap.style.maxHeight = "none";
      }
      renderSelectedImage(items[0]);
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
        $("healthBanner").textContent = `서비스 상태: ${body.status} / 샘플 ${body.images}건`;
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
        $("adminStatus").textContent = `스토어: ${body.store_dir} / 생성 폴더 ${folders.length}개 / 생성 파일 ${files.length}개${locked}`;
        if (!files.length) {
          $("folderList").innerHTML = '<span class="warn">생성된 파일이 없습니다.</span>';
          return;
        }
        $("folderList").innerHTML = files.map((f) => `<div class="folder-item">${f}</div>`).join("");
      } catch (e) {
        $("adminStatus").innerHTML = `<span class="warn">스토어 정보 조회 실패: ${String(e)}</span>`;
      }
    }

    async function refreshStatus() {
      await Promise.all([callHealth(), loadMockStoreInfo()]);
    }

    async function rebuildSamples() {
      if (jobRunning) return;
      openJobModal("샘플 위성영상 생성 중", "현재 백그라운드 작업이 진행 중입니다. 완료되면 닫기 버튼이 활성화됩니다.");
      try {
        const { res, body } = await getJson("/admin/mock-store/rebuild", { method: "POST" });
        if (!res.ok) {
          $("detailOut").textContent = pretty({ status: res.status, body });
          const reason = body && body.detail
            ? String(body.detail)
            : "샘플 생성 요청에 실패했습니다. 상세 내용은 Raw API 응답을 확인해주세요.";
          finishJobModal(false, reason);
          return;
        }
        $("detailOut").textContent = pretty({ status: res.status, body });
        await callImages();
        await refreshStatus();
        finishJobModal(true, "샘플 생성이 완료되었습니다. 확인 후 닫기를 누르세요.");
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e) });
        finishJobModal(false, `오류가 발생했습니다: ${String(e)}`);
      }
    }

    async function deleteSamples() {
      const ok = await openDeleteConfirm();
      if (!ok) return;
      try {
        const { res, body } = await getJson("/admin/mock-store/delete", { method: "POST" });
        if (!res.ok) {
          $("detailOut").textContent = pretty({ status: res.status, body });
          return;
        }
        $("detailOut").textContent = pretty({ status: res.status, body });
        $("imagesList").innerHTML = '<span class="warn">샘플이 삭제되었습니다.</span>';
        $("imageKv").innerHTML = '<div class="k">상태</div><div class="warn">샘플 없음</div>';
        $("folderList").innerHTML = '<span class="warn">생성된 파일이 없습니다.</span>';
        await callImages();
        await refreshStatus();
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e) });
      }
    }

    function resetFilters() {
      $("sensor").value = "";
      $("level").value = "";
      $("fmt").value = "";
      $("q").value = "";
      callImages();
    }

    async function callImages() {
      const qs = buildQuery({
        sensor: $("sensor").value,
        level: $("level").value,
        fmt: $("fmt").value,
        q: $("q").value
      });
      const url = "/images" + (qs ? ("?" + qs) : "");
      try {
        const { body } = await getJson(url);
        const items = body.items || [];
        updateStats(items);
        renderImagesList(items);
      } catch (e) {
        $("imagesList").innerHTML = `<span class="warn">목록 조회 실패: ${String(e)}</span>`;
      }
    }

    async function callDetail() {
      const pid = $("pid").value.trim();
      if (!pid) {
        $("detailOut").textContent = pretty({ error: "image_id required" });
        return;
      }
      try {
        const { res, body } = await getJson(`/images/${encodeURIComponent(pid)}`);
        $("detailOut").textContent = pretty({ status: res.status, body });
        if (res.ok) renderSelectedImage(body);
      } catch (e) {
        $("detailOut").textContent = pretty({ error: String(e) });
      }
    }

    async function callDownload() {
      const pid = $("pid").value.trim();
      if (!pid) {
        $("detailOut").textContent = pretty({ error: "image_id required" });
        return;
      }
      const url = `/images/${encodeURIComponent(pid)}/download`;
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
    refreshStatus();
    ["sensor", "level", "fmt"].forEach((id) => {
      $(id).addEventListener("change", callImages);
    });
    callImages();
  </script>
</body>
</html>"""


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["ui"],
    summary="루트 안내 페이지",
    description="`/ui`와 `/docs`로 이동할 수 있는 간단한 시작 페이지를 반환합니다.",
    response_description="HTML 문서",
)
def root() -> str:
    return '<html><body style="font-family:sans-serif;padding:20px;"><h2>Satti (Virtual Satellite Imagery Service) API</h2><p><a href="/ui">위상영상이미지포맷 L0~L4 학습하기</a></p><p><a href="/docs">Swagger Docs</a></p></body></html>'

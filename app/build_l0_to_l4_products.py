#!/usr/bin/env python3
"""
L0~L4 포맷 가정을 기준으로 L0 입력에서 L1~L4 산출물을 생성하는 스크립트.

[지원 산출물]
1) L1: GeoTIFF + CEOS L1A archival .bin
2) L2: GeoTIFF + RPC/DEM 사이드카(.json, .txt)
3) L3: Mosaic GeoTIFF + 256x256 Tiled GeoTIFF
4) L4: Classified raster + Index map (둘 다 TIFF)

[실행 예]
cd /Users/jaehojoo/Desktop/codex-lgcns/sattie
python3 build_l0_to_l4_products.py eo  -i ./samples/eo_l0.bin  -o ./products/eo_from_l0
python3 build_l0_to_l4_products.py sar -i ./samples/sar_l0.bin -o ./products/sar_from_l0

[생성 결과]
- EO: ./products/eo_from_l0
- SAR: ./products/sar_from_l0
- 각 폴더에 product_summary.json 포함

[주의]
- 본 파이프라인은 테스트용 합성 처리입니다.
- 미션급 정식 처리(정확한 GeoTIFF GeoKey, 실 DEM 기반 정사보정,
  SAR focusing 알고리즘)는 별도 고도화가 필요합니다.

Assumed product mapping:
- L0: Raw binary / Complex IQ
- L1: GeoTIFF, CEOS L1A archival
- L2: GeoTIFF with RPC/DEM sidecar
- L3: Tiled GeoTIFF, Mosaic
- L4: Classified raster, index maps

No external Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from array import array
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def clip_u16(v: float) -> int:
    if v <= 0:
        return 0
    if v >= 65535:
        return 65535
    return int(round(v))


def write_tiff_gray_u16(path: Path, width: int, height: int, data_u16: list[int]) -> None:
    """
    Minimal uncompressed grayscale TIFF writer (little-endian, uint16).
    """
    if len(data_u16) != width * height:
        raise ValueError("TIFF data length mismatch.")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Pixel bytes
    img = array("H", data_u16)
    img_bytes = img.tobytes()

    # TIFF layout
    # [header 8 bytes][image data][IFD]
    header = bytearray()
    header += b"II"  # little-endian
    header += struct.pack("<H", 42)
    ifd_offset = 8 + len(img_bytes)
    header += struct.pack("<I", ifd_offset)

    # IFD entries (tag, type, count, value)
    # type: SHORT=3, LONG=4
    entries: list[tuple[int, int, int, int]] = [
        (256, 4, 1, width),              # ImageWidth
        (257, 4, 1, height),             # ImageLength
        (258, 3, 1, 16),                 # BitsPerSample
        (259, 3, 1, 1),                  # Compression = none
        (262, 3, 1, 1),                  # PhotometricInterpretation = BlackIsZero
        (273, 4, 1, 8),                  # StripOffsets = image starts after header
        (277, 3, 1, 1),                  # SamplesPerPixel
        (278, 4, 1, height),             # RowsPerStrip
        (279, 4, 1, len(img_bytes)),     # StripByteCounts
        (284, 3, 1, 1),                  # PlanarConfiguration = chunky
    ]

    ifd = bytearray()
    ifd += struct.pack("<H", len(entries))
    for tag, typ, count, value in entries:
        ifd += struct.pack("<HHII", tag, typ, count, value)
    ifd += struct.pack("<I", 0)  # next IFD

    with path.open("wb") as f:
        f.write(header)
        f.write(img_bytes)
        f.write(ifd)


def write_bmp_rgb8(path: Path, width: int, height: int, rgb: list[tuple[int, int, int]]) -> None:
    """
    Write uncompressed 24-bit BMP (BGR, bottom-up).
    """
    if len(rgb) != width * height:
        raise ValueError("BMP RGB length mismatch.")
    path.parent.mkdir(parents=True, exist_ok=True)

    row_bytes = width * 3
    pad = (4 - (row_bytes % 4)) % 4
    pixel_array_size = (row_bytes + pad) * height
    file_size = 14 + 40 + pixel_array_size

    with path.open("wb") as f:
        # BITMAPFILEHEADER
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HH", 0, 0))
        f.write(struct.pack("<I", 14 + 40))

        # BITMAPINFOHEADER
        f.write(struct.pack("<I", 40))          # header size
        f.write(struct.pack("<i", width))
        f.write(struct.pack("<i", height))      # positive -> bottom-up
        f.write(struct.pack("<H", 1))           # planes
        f.write(struct.pack("<H", 24))          # bpp
        f.write(struct.pack("<I", 0))           # BI_RGB
        f.write(struct.pack("<I", pixel_array_size))
        f.write(struct.pack("<i", 2835))        # 72 DPI
        f.write(struct.pack("<i", 2835))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        # Pixels (bottom-up, BGR)
        for y in range(height - 1, -1, -1):
            row = y * width
            for x in range(width):
                r, g, b = rgb[row + x]
                f.write(struct.pack("BBB", b, g, r))
            if pad:
                f.write(b"\x00" * pad)


def u16_to_gray8(v: int) -> int:
    return max(0, min(255, v >> 8))


def colorize_eo_u16(data: list[int]) -> list[tuple[int, int, int]]:
    """
    Pseudo-natural color from single-band EO intensity.
    """
    out: list[tuple[int, int, int]] = []
    for v in data:
        g = u16_to_gray8(v)
        r = max(0, min(255, int(g * 1.02)))
        gg = max(0, min(255, int(g * 0.95 + 8)))
        b = max(0, min(255, int(g * 0.82 + 14)))
        out.append((r, gg, b))
    return out


def colorize_sar_u16(data: list[int]) -> list[tuple[int, int, int]]:
    """
    Blue->cyan->yellow style gradient for SAR intensity.
    """
    out: list[tuple[int, int, int]] = []
    for v in data:
        x = v / 65535.0
        if x < 0.5:
            t = x / 0.5
            r = int(20 * t)
            g = int(220 * t)
            b = int(90 + 165 * (1 - t))
        else:
            t = (x - 0.5) / 0.5
            r = int(20 + 235 * t)
            g = int(220 + 25 * (1 - t))
            b = int(15 + 30 * (1 - t))
        out.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    return out


def colorize_class_u16(data: list[int]) -> list[tuple[int, int, int]]:
    """
    4-class palette for L4 classified raster.
    """
    palette = {
        0: (30, 30, 30),          # class 0
        21845: (65, 140, 240),    # class 1
        43690: (70, 180, 90),     # class 2
        65535: (245, 200, 70),    # class 3
    }
    out: list[tuple[int, int, int]] = []
    for v in data:
        out.append(palette.get(v, (255, 0, 255)))
    return out


def read_eo_l0_u16(path: Path, width: int, height: int) -> list[int]:
    raw = path.read_bytes()
    expected = width * height * 2
    if len(raw) != expected:
        raise ValueError(f"EO size mismatch: got {len(raw)}, expected {expected}")
    vals = array("H")
    vals.frombytes(raw)
    return vals.tolist()


def read_sar_l0_iq(path: Path, pulses: int, spp: int) -> list[tuple[int, int]]:
    raw = path.read_bytes()
    expected = pulses * spp * 4
    if len(raw) != expected:
        raise ValueError(f"SAR size mismatch: got {len(raw)}, expected {expected}")
    vals = array("h")
    vals.frombytes(raw)
    out: list[tuple[int, int]] = []
    for i in range(0, len(vals), 2):
        out.append((vals[i], vals[i + 1]))
    return out


def eo_calibrate_l1(data: list[int], dark: int, gain: float) -> list[int]:
    return [clip_u16((v - dark) * gain) for v in data]


def eo_l2_orthorectify_mock(data: list[int], width: int, height: int) -> list[int]:
    """
    Synthetic orthorectification surrogate:
    row-dependent horizontal shift compensation.
    """
    out = [0] * (width * height)
    for y in range(height):
        # small smooth shift [-2..+2] pixels
        shift = int(round(2.0 * math.sin(2.0 * math.pi * y / max(1, height))))
        row = y * width
        for x in range(width):
            sx = x + shift
            if 0 <= sx < width:
                out[row + x] = data[row + sx]
            else:
                out[row + x] = 0
    return out


def sar_l1_intensity_u16(iq: list[tuple[int, int]]) -> list[int]:
    """
    IQ -> log-intensity mapped to uint16 for image products.
    """
    # compute power
    p = [float(i * i + q * q) for i, q in iq]
    # robust log scaling
    eps = 1.0
    logp = [math.log10(v + eps) for v in p]
    mn = min(logp)
    mx = max(logp)
    span = max(1e-9, mx - mn)
    return [clip_u16((v - mn) / span * 65535.0) for v in logp]


def build_mosaic_side_by_side(data: list[int], width: int, height: int) -> tuple[list[int], int, int]:
    """
    Simple synthetic L3 mosaic: duplicate image side-by-side with slight gain delta.
    """
    new_w = width * 2
    out = [0] * (new_w * height)
    for y in range(height):
        src_row = y * width
        dst_row = y * new_w
        for x in range(width):
            v = data[src_row + x]
            out[dst_row + x] = v
            out[dst_row + width + x] = clip_u16(v * 0.97)
    return out, new_w, height


def write_tiled_products(
    out_dir: Path,
    data: list[int],
    width: int,
    height: int,
    tile_size: int = 256,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0 = tx * tile_size
            y0 = ty * tile_size
            tw = min(tile_size, width - x0)
            th = min(tile_size, height - y0)
            tile = [0] * (tw * th)
            for yy in range(th):
                src_row = (y0 + yy) * width
                dst_row = yy * tw
                for xx in range(tw):
                    tile[dst_row + xx] = data[src_row + (x0 + xx)]
            tile_name = f"tile_{ty:03d}_{tx:03d}.tif"
            tile_path = out_dir / tile_name
            write_tiff_gray_u16(tile_path, tw, th, tile)
            files.append(tile_name)
    return files


def build_l4_products(data_l2: list[int], width: int, height: int) -> tuple[list[int], list[int]]:
    """
    Build:
    - index_map_u16: normalized index-like product
    - class_map_u16: simple 4-class classification map
    """
    mn = min(data_l2)
    mx = max(data_l2)
    span = max(1, mx - mn)
    idx = [clip_u16((v - mn) / span * 65535.0) for v in data_l2]

    cls = []
    for v in idx:
        # 4 classes encoded as 0, 21845, 43690, 65535
        if v < 16384:
            cls.append(0)
        elif v < 32768:
            cls.append(21845)
        elif v < 49152:
            cls.append(43690)
        else:
            cls.append(65535)
    return idx, cls


def build_products(mode: str, l0_path: Path, out_root: Path) -> None:
    meta_path = Path(str(l0_path) + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Input metadata not found: {meta_path}")
    meta = read_json(meta_path)

    out_root.mkdir(parents=True, exist_ok=True)

    if mode == "eo":
        width = int(meta["width"])
        height = int(meta["height"])
        l0 = read_eo_l0_u16(l0_path, width, height)

        # L1
        l1 = eo_calibrate_l1(l0, dark=64, gain=1.12)
        l1_tif = out_root / "L1_eo_geotiff.tif"
        write_tiff_gray_u16(l1_tif, width, height, l1)
        l1_preview = out_root / "L1_eo_preview_rgb.bmp"
        write_bmp_rgb8(l1_preview, width, height, colorize_eo_u16(l1))
        l1_ceos = out_root / "L1_eo_ceos_l1a.bin"
        l1_ceos.write_bytes(bytes(array("H", l1)))

        # L2
        l2 = eo_l2_orthorectify_mock(l1, width, height)
        l2_tif = out_root / "L2_eo_geotiff_rpc_dem.tif"
        write_tiff_gray_u16(l2_tif, width, height, l2)
        l2_preview = out_root / "L2_eo_preview_rgb.bmp"
        write_bmp_rgb8(l2_preview, width, height, colorize_eo_u16(l2))
        write_json(
            out_root / "L2_eo_rpc.json",
            {
                "rpc_model": "mock",
                "line_off": height / 2.0,
                "samp_off": width / 2.0,
                "lat_off": 36.0,
                "lon_off": 127.5,
            },
        )
        (out_root / "L2_eo_dem.txt").write_text("DEM: mock_dem_source, vertical_datum: EGM96\n")

        # L3
        mosaic, mw, mh = build_mosaic_side_by_side(l2, width, height)
        l3_mosaic = out_root / "L3_eo_mosaic.tif"
        write_tiff_gray_u16(l3_mosaic, mw, mh, mosaic)
        l3_preview = out_root / "L3_eo_mosaic_preview_rgb.bmp"
        write_bmp_rgb8(l3_preview, mw, mh, colorize_eo_u16(mosaic))
        tiles = write_tiled_products(out_root / "L3_eo_tiles_256", mosaic, mw, mh, tile_size=256)

        # L4
        idx, cls = build_l4_products(l2, width, height)
        l4_idx = out_root / "L4_eo_index_map.tif"
        l4_cls = out_root / "L4_eo_classified_raster.tif"
        write_tiff_gray_u16(l4_idx, width, height, idx)
        write_tiff_gray_u16(l4_cls, width, height, cls)
        l4_idx_preview = out_root / "L4_eo_index_preview_rgb.bmp"
        l4_cls_preview = out_root / "L4_eo_classified_preview_rgb.bmp"
        write_bmp_rgb8(l4_idx_preview, width, height, colorize_sar_u16(idx))
        write_bmp_rgb8(l4_cls_preview, width, height, colorize_class_u16(cls))

        summary = {
            "mode": "eo",
            "input_l0": str(l0_path),
            "products": {
                "L1": [str(l1_tif.name), str(l1_ceos.name)],
                "L2": [str(l2_tif.name), "L2_eo_rpc.json", "L2_eo_dem.txt"],
                "L3": [str(l3_mosaic.name), f"L3_eo_tiles_256 ({len(tiles)} tiles)"],
                "L4": [str(l4_idx.name), str(l4_cls.name)],
                "Preview": [
                    str(l1_preview.name),
                    str(l2_preview.name),
                    str(l3_preview.name),
                    str(l4_idx_preview.name),
                    str(l4_cls_preview.name),
                ],
            },
            "notes": "Synthetic EO processing chain based on assumed format mapping.",
        }
    else:
        pulses = int(meta["pulses"])
        spp = int(meta["samples_per_pulse"])
        iq = read_sar_l0_iq(l0_path, pulses, spp)

        # L1
        l1 = sar_l1_intensity_u16(iq)
        l1_tif = out_root / "L1_sar_geotiff.tif"
        write_tiff_gray_u16(l1_tif, spp, pulses, l1)
        l1_preview = out_root / "L1_sar_preview_colormap.bmp"
        write_bmp_rgb8(l1_preview, spp, pulses, colorize_sar_u16(l1))
        l1_ceos = out_root / "L1_sar_ceos_l1a.bin"
        l1_ceos.write_bytes(bytes(array("H", l1)))

        # L2
        # synthetic geocoded variant via mild row smoothing
        l2 = l1[:]
        for y in range(1, pulses - 1):
            row = y * spp
            prev = (y - 1) * spp
            nxt = (y + 1) * spp
            for x in range(spp):
                l2[row + x] = (l1[prev + x] + 2 * l1[row + x] + l1[nxt + x]) // 4

        l2_tif = out_root / "L2_sar_geotiff_rpc_dem.tif"
        write_tiff_gray_u16(l2_tif, spp, pulses, l2)
        l2_preview = out_root / "L2_sar_preview_colormap.bmp"
        write_bmp_rgb8(l2_preview, spp, pulses, colorize_sar_u16(l2))
        write_json(
            out_root / "L2_sar_rpc.json",
            {
                "rpc_model": "mock_sar",
                "line_off": pulses / 2.0,
                "samp_off": spp / 2.0,
                "lat_off": 36.0,
                "lon_off": 127.5,
            },
        )
        (out_root / "L2_sar_dem.txt").write_text("DEM: mock_dem_source, vertical_datum: EGM96\n")

        # L3
        mosaic, mw, mh = build_mosaic_side_by_side(l2, spp, pulses)
        l3_mosaic = out_root / "L3_sar_mosaic.tif"
        write_tiff_gray_u16(l3_mosaic, mw, mh, mosaic)
        l3_preview = out_root / "L3_sar_mosaic_preview_colormap.bmp"
        write_bmp_rgb8(l3_preview, mw, mh, colorize_sar_u16(mosaic))
        tiles = write_tiled_products(out_root / "L3_sar_tiles_256", mosaic, mw, mh, tile_size=256)

        # L4
        idx, cls = build_l4_products(l2, spp, pulses)
        l4_idx = out_root / "L4_sar_index_map.tif"
        l4_cls = out_root / "L4_sar_classified_raster.tif"
        write_tiff_gray_u16(l4_idx, spp, pulses, idx)
        write_tiff_gray_u16(l4_cls, spp, pulses, cls)
        l4_idx_preview = out_root / "L4_sar_index_preview_colormap.bmp"
        l4_cls_preview = out_root / "L4_sar_classified_preview_rgb.bmp"
        write_bmp_rgb8(l4_idx_preview, spp, pulses, colorize_sar_u16(idx))
        write_bmp_rgb8(l4_cls_preview, spp, pulses, colorize_class_u16(cls))

        summary = {
            "mode": "sar",
            "input_l0": str(l0_path),
            "products": {
                "L1": [str(l1_tif.name), str(l1_ceos.name)],
                "L2": [str(l2_tif.name), "L2_sar_rpc.json", "L2_sar_dem.txt"],
                "L3": [str(l3_mosaic.name), f"L3_sar_tiles_256 ({len(tiles)} tiles)"],
                "L4": [str(l4_idx.name), str(l4_cls.name)],
                "Preview": [
                    str(l1_preview.name),
                    str(l2_preview.name),
                    str(l3_preview.name),
                    str(l4_idx_preview.name),
                    str(l4_cls_preview.name),
                ],
            },
            "notes": "Synthetic SAR processing chain based on assumed format mapping.",
        }

    write_json(out_root / "product_summary.json", summary)
    print(f"Created product set in: {out_root}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic L1~L4 products from L0.")
    parser.add_argument("mode", choices=["eo", "sar"], help="Sensor mode")
    parser.add_argument("-i", "--input-l0", required=True, help="Input L0 binary path")
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Output directory for L1~L4 products",
    )
    args = parser.parse_args()

    build_products(
        mode=args.mode,
        l0_path=Path(args.input_l0).expanduser().resolve(),
        out_root=Path(args.out_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()

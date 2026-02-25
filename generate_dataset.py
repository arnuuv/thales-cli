#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset generator script for Thales mapping application.
Generates synthetic terrain and imagery datasets with proper georeferencing.

Usage:
    python generate_dataset.py --biome city --width 1000 --height 800 --seed 1234 \
        --outdir ./output --bounds -122.5 37.5 -122.3 37.7
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Literal
import argparse
import json
import math
import os
import shutil
import time
from io import BytesIO
from typing import Tuple, Optional

import numpy as np
from PIL import Image

# ============================================================================
# HEARTBEAT (for diagnosing hangs)
# ============================================================================

_last_heartbeat = time.time()


def heartbeat(label: str, force: bool = False):
    """Print periodic heartbeat to show progress during long operations."""
    global _last_heartbeat
    now = time.time()
    if force or (now - _last_heartbeat) > 3:  # Print every 3+ seconds
        print(f"⏳ {label}...", flush=True)
        _last_heartbeat = now


# ============================================================================
# WORLD-COORDINATE NOISE (for cross-resolution consistency)
# ============================================================================


def grid_px(feature_m: float, px_m: float) -> int:
    """
    Convert a feature size in meters to a pixel grid size.

    This ensures noise features have consistent physical size across resolutions.

    Args:
        feature_m: Desired feature size in meters (e.g., 600m for large hills)
        px_m: Pixel size in meters (GSD)

    Returns:
        Grid size in pixels (minimum 2)

    Example:
        At 1m GSD: grid_px(600, 1.0) = 600 pixels
        At 10m GSD: grid_px(600, 10.0) = 60 pixels
        → Same 600m feature, different sampling
    """
    return max(2, int(round(feature_m / max(px_m, 1e-6))))


try:
    import rasterio
    from rasterio.enums import Resampling, ColorInterp
    from rasterio.transform import from_bounds

    RIO = True
except Exception:
    RIO = False
    print("Warning: rasterio not available. GeoTIFF output will be disabled.")

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. OGF tile downloading disabled.")

try:
    from pyproj import Transformer

    PYPROJ = True
except Exception:
    PYPROJ = False
    print(
        "Warning: pyproj not available. UTM mode requires pyproj (pip install pyproj)"
    )

# OpenCV availability flag
OPENCV_AVAILABLE = False
try:
    import cv2

    OPENCV_AVAILABLE = True
except Exception:
    pass

try:
    from opencv_photorealism import synthesize_photorealistic_eo, SensorConfig

    OPENCV_PHOTOREALISM_AVAILABLE = True
except Exception as e:
    OPENCV_PHOTOREALISM_AVAILABLE = False
    print(f"Warning: OpenCV photorealism not available: {e}")

try:
    from eo_forward_pipeline import eo_forward_pipeline, EOConfig

    EO_PIPELINE_AVAILABLE = True
except Exception as e:
    EO_PIPELINE_AVAILABLE = False
    EOConfig = object  # Fallback type to prevent NameError
    print(f"Warning: EO forward pipeline not available: {e}")

# ------------------ utils ------------------


def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)


def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile coordinates"""
    import math

    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Convert tile coordinates to lat/lon"""
    import math

    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def blur(img, k=3):
    """Simple box blur using rolling average."""
    if k <= 1:
        return img
    out = img
    for _ in range(k):
        out = (np.roll(out, 1, 0) + out + np.roll(out, -1, 0)) / 3.0
        out = (np.roll(out, 1, 1) + out + np.roll(out, -1, 1)) / 3.0
    return out


def clean_water_mask(
    mask: np.ndarray, min_area_px: int = 200, use_morph: bool = True
) -> np.ndarray:
    """
    Clean water mask by removing small speckles and applying morphological operations.

    Args:
        mask: Boolean water mask
        min_area_px: Minimum connected component area in pixels
        use_morph: Apply morphological opening/closing

    Returns:
        Cleaned boolean mask
    """
    if mask.sum() == 0:
        return mask

    mask_clean = mask.copy()

    if use_morph and OPENCV_AVAILABLE:
        mask_u8 = mask_clean.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_clean = mask_u8 > 127
    else:
        mask_clean = blur(mask_clean.astype(np.float32), k=2) > 0.4

    try:
        from scipy import ndimage

        labeled, num_features = ndimage.label(mask_clean)
        if num_features > 0:
            sizes = ndimage.sum(mask_clean, labeled,
                                range(1, num_features + 1))
            mask_clean = np.isin(labeled, np.where(
                sizes >= min_area_px)[0] + 1)
    except ImportError:
        pass

    return mask_clean


def extract_ogf_water_mask(ogf_rgb: np.ndarray) -> np.ndarray:
    """
    Extract water mask from OGF cartographic tiles.

    Args:
        ogf_rgb: OGF RGB image (H, W, 3), float32 [0-1]

    Returns:
        Water mask (H, W), boolean
    """
    blue_dominance = ogf_rgb[..., 2] - \
        np.maximum(ogf_rgb[..., 0], ogf_rgb[..., 1])
    ogf_water_mask = (
        (blue_dominance > 0.12)
        | ((ogf_rgb[..., 2] > 0.4) & (ogf_rgb[..., 0] < 0.3) & (ogf_rgb[..., 1] < 0.4))
        | (
            (ogf_rgb[..., 2] > ogf_rgb[..., 0] + 0.2)
            & (ogf_rgb[..., 2] > ogf_rgb[..., 1] + 0.15)
        )
    )

    not_label = np.mean(ogf_rgb, axis=2) < 0.92
    not_dark = np.mean(ogf_rgb, axis=2) > 0.10
    ogf_water_mask = ogf_water_mask & not_label & not_dark

    ogf_water_mask = blur(ogf_water_mask.astype(np.float32), k=2) > 0.3

    return ogf_water_mask


def download_ogf_tiles(
    bounds: Tuple[float, float, float, float], zoom: int = 14
) -> np.ndarray:
    """
    Download OGF map tiles for the given bounds and stitch them together.
    Returns RGB imagery array.
    """
    if not REQUESTS_AVAILABLE:
        print("   Warning: requests not available, cannot download OGF tiles")
        return None

    minLon, minLat, maxLon, maxLat = bounds

    # Get tile range
    x_min, y_max = deg2num(maxLat, minLon, zoom)  # Note: y increases downward
    x_max, y_min = deg2num(minLat, maxLon, zoom)

    # Ensure proper ordering
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(
        f"   Downloading OGF tiles at zoom {zoom}: x={x_min}-{x_max}, y={y_min}-{y_max} ({tile_count} tiles)",
        flush=True,
    )

    # HARD LIMIT: Never download more than 40 tiles (prevents 10+ minute waits)
    if tile_count > 40:
        print(
            f"   ⚠️  Too many OGF tiles ({tile_count}), skipping OGF entirely")
        print(
            f"   💡 OGF is only for water detection - using procedural generation instead"
        )
        return None

    if tile_count > 100:
        if zoom <= 1:
            print(
                f"   Warning: {tile_count} tiles needed but zoom too low to reduce further. Giving up on OGF."
            )
            return None
        print(
            f"   Warning: {tile_count} tiles needed, reducing to zoom {zoom-1}")
        return download_ogf_tiles(bounds, zoom - 1)

    if tile_count == 0:
        print(f"   Error: No tiles in range, bounds may be invalid")
        return None

    tiles = []

    # Download tiles
    downloaded = 0
    for y in range(y_min, y_max + 1):
        row = []
        for x in range(x_min, x_max + 1):
            try:
                # OGF tile server URL - use ogf-carto for better imagery
                url = f"https://tile.opengeofiction.net/ogf-carto/{zoom}/{x}/{y}.png"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    tile_img = Image.open(BytesIO(response.content))
                    row.append(np.array(tile_img.convert("RGB")))
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(
                            f"   Downloaded {downloaded}/{tile_count} tiles...",
                            flush=True,
                        )
                else:
                    # Blank tile if download fails
                    row.append(np.full((256, 256, 3), 240, dtype=np.uint8))
            except Exception as e:
                print(f"   Failed to download tile {x},{y}: {e}")
                row.append(np.full((256, 256, 3), 240, dtype=np.uint8))

        if row:
            tiles.append(np.concatenate(row, axis=1))

    if not tiles:
        return None

    # Stitch tiles together
    stitched = np.concatenate(tiles, axis=0)
    print(
        f"   Stitched {downloaded} tiles into {stitched.shape[1]}x{stitched.shape[0]} image",
        flush=True,
    )

    return stitched.astype(np.float32) / 255.0


def detect_real_terrain_type(
    bounds: Tuple[float, float, float, float],
) -> Tuple[str, float]:
    """
    Sample real-world elevation data to detect if area is water, coast, or land.
    Returns (terrain_type, water_percentage) where terrain_type is 'water', 'coast', or 'land'
    """
    if not REQUESTS_AVAILABLE:
        return "unknown", 0.0

    minLon, minLat, maxLon, maxLat = bounds

    # Sample a grid of points (5x5 = 25 points for speed)
    sample_points = []
    for i in range(5):
        for j in range(5):
            lat = minLat + (maxLat - minLat) * i / 4
            lon = minLon + (maxLon - minLon) * j / 4
            sample_points.append((lat, lon))

    # Query Open-Elevation API (free, no auth required)
    # Note: This may be slow or unavailable, so we have a timeout
    water_count = 0
    valid_samples = 0

    try:
        # Batch query for efficiency
        locations = [{"latitude": lat, "longitude": lon}
                     for lat, lon in sample_points]
        response = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=10,
        )

        if response.status_code == 200:
            results = response.json().get("results", [])
            for result in results:
                elevation = result.get("elevation", 0)
                valid_samples += 1
                # Water is typically elevation <= 1m (accounting for tides, errors)
                if elevation <= 1:
                    water_count += 1
        else:
            print(
                f"   Elevation API returned status {response.status_code}, using biome as-is"
            )
            return "unknown", 0.0

    except Exception as e:
        print(f"   Could not sample real elevation (using biome as-is): {e}")
        return "unknown", 0.0

    if valid_samples == 0:
        return "unknown", 0.0

    water_percentage = water_count / valid_samples

    # Classify based on water percentage
    if water_percentage > 0.90:
        return "water", water_percentage
    elif water_percentage > 0.30:
        return "coast", water_percentage
    else:
        return "land", water_percentage


def to01(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    m, M = float(a.min()), float(a.max())
    if M - m < 1e-9:
        return np.zeros_like(a, np.float32)
    return ((a - m) / (M - m)).astype(np.float32)


def save_png_rgb(rgb: np.ndarray, path: str):
    Image.fromarray(np.clip(rgb * 255 + 0.5, 0,
                    255).astype(np.uint8), "RGB").save(path)


def lonlat_bounds_to_mercator(
    bounds: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Convert WGS84 lat/lon bounds to Web Mercator (EPSG:3857) bounds in meters.

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees

    Returns:
        (minX, minY, maxX, maxY) in Web Mercator meters
    """
    minLon, minLat, maxLon, maxLat = bounds
    R = 6378137.0  # Earth radius in meters (Web Mercator standard)

    def project(lon: float, lat: float) -> Tuple[float, float]:
        """Project a single lon/lat point to Web Mercator x/y"""
        x = R * math.radians(lon)
        y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return x, y

    minx, miny = project(minLon, minLat)
    maxx, maxy = project(maxLon, maxLat)
    return (minx, miny, maxx, maxy)


def bounds_cross_utm_zone(bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if bounding box crosses UTM zone boundaries.

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees

    Returns:
        True if bounds span multiple UTM zones
    """
    minLon, _, maxLon, _ = bounds
    zone_min = int((minLon + 180) / 6) + 1
    zone_max = int((maxLon + 180) / 6) + 1
    return zone_min != zone_max


def mercator_safe(bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if bounds are within Web Mercator valid latitude range.

    EPSG:3857 breaks beyond ±85.0511° due to log(tan()) singularity.

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees

    Returns:
        True if bounds are safe for Web Mercator projection
    """
    _, minLat, _, maxLat = bounds
    return abs(minLat) <= 85.0511 and abs(maxLat) <= 85.0511


def resolve_metric_crs(
    bounds: Tuple[float, float, float, float], crs_metric: str
) -> str:
    """
    Resolve CRS string, handling UTM zone crossing automatically.

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees
        crs_metric: User-specified CRS (can be "UTM" or an EPSG code)

    Returns:
        Resolved EPSG code string
    """
    if crs_metric.upper() == "UTM":
        if bounds_cross_utm_zone(bounds):
            print("⚠️  Bounds cross UTM zones → using EPSG:3857", flush=True)
            return "EPSG:3857"
        return utm_crs_from_bounds(bounds)
    return crs_metric


def normalize_dem(dem_m: np.ndarray) -> np.ndarray:
    """
    Normalize DEM from meters to 0-1 range for color generation.

    ⚠️  CRITICAL: Only use normalized DEM for RGB generation.
    For hillshade, slope, contours, and GeoTIFF export, use dem_m (meters).

    Args:
        dem_m: DEM in meters (float32)

    Returns:
        DEM normalized to 0-1 range
    """
    return (dem_m - dem_m.min()) / (dem_m.max() - dem_m.min() + 1e-6)


def utm_crs_from_bounds(bounds: Tuple[float, float, float, float]) -> str:
    """
    Determine appropriate UTM CRS from lat/lon bounds.

    WARNING: Assumes bounds fit within a SINGLE UTM zone.
    Use bounds_cross_utm_zone() to check first.

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees

    Returns:
        EPSG code for UTM zone (e.g., "EPSG:32630" for UTM 30N)
    """
    minLon, minLat, maxLon, maxLat = bounds
    lon0 = (minLon + maxLon) / 2.0
    lat0 = (minLat + maxLat) / 2.0

    # Compute UTM zone
    zone = int((lon0 + 180) / 6) + 1

    # Northern vs Southern hemisphere
    if lat0 >= 0:
        epsg_code = 32600 + zone  # UTM North
    else:
        epsg_code = 32700 + zone  # UTM South

    return f"EPSG:{epsg_code}"


def bounds_to_projected(
    bounds: Tuple[float, float, float, float], crs_out: str
) -> Tuple[float, float, float, float]:
    """
    Project WGS84 bounds to any metric CRS (UTM, Web Mercator, etc.).

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees
        crs_out: Target CRS (e.g., "EPSG:32630" for UTM, "EPSG:3857" for Web Mercator)

    Returns:
        (minX, minY, maxX, maxY) in target CRS meters
    """
    minLon, minLat, maxLon, maxLat = bounds

    # Fallback to Web Mercator if pyproj not available
    if not PYPROJ:
        if crs_out != "EPSG:3857":
            raise RuntimeError(
                f"pyproj required for {crs_out} (pip install pyproj). Fallback: use EPSG:3857"
            )
        return lonlat_bounds_to_mercator(bounds)

    # Use pyproj for accurate projection
    tf = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)
    minx, miny = tf.transform(minLon, minLat)
    maxx, maxy = tf.transform(maxLon, maxLat)

    # Ensure proper ordering (some projections may flip)
    minx, maxx = (min(minx, maxx), max(minx, maxx))
    miny, maxy = (min(miny, maxy), max(miny, maxy))

    return (minx, miny, maxx, maxy)


def compute_world_dimensions_projected(
    bounds: Tuple[float, float, float, float], crs_out: str
) -> Tuple[float, float]:
    """
    Compute world dimensions in any metric CRS (TRUE ground meters for UTM).

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees
        crs_out: Target CRS (UTM for true meters, 3857 for web)

    Returns:
        (world_width_m, world_height_m) in target CRS meters
    """
    minx, miny, maxx, maxy = bounds_to_projected(bounds, crs_out)
    world_width_m = maxx - minx
    world_height_m = maxy - miny
    return world_width_m, world_height_m


def write_tif_with_bounds(
    arr: np.ndarray,
    bounds: Tuple[float, float, float, float],
    path: str,
    bands=1,
    photometric=None,
    overviews=True,
    crs_out: str = "UTM",
    snap_gsd_m: Optional[float] = None,
    transform_override=None,
):
    """
    Write GeoTIFF with proper georeferencing from bounds (minLon, minLat, maxLon, maxLat).

    Args:
        crs_out: Output CRS
            - "UTM": Auto-select UTM zone (TRUE ground meters, recommended)
            - "EPSG:3857": Web Mercator (projected meters, scale varies with latitude)
            - "EPSG:326XX": Specific UTM zone
            - "EPSG:4326": WGS84 degrees (legacy, only for RGB web maps)
        snap_gsd_m: DEPRECATED - use transform_override instead
        transform_override: If provided, use this transform directly (for multi-resolution)

    Default is "UTM" for true ground meters.
    """
    if not RIO:
        raise RuntimeError(
            "❌ rasterio not available - cannot write GeoTIFF outputs! Install: pip install rasterio"
        )
    h, w = arr.shape[:2]
    minLon, minLat, maxLon, maxLat = bounds

    # Determine dtype and band count
    count = arr.shape[2] if (bands != 1 and arr.ndim == 3) else 1
    nodata_value = None

    if arr.ndim == 3:
        # RGB imagery: enforce uint8 with proper scaling
        dtype = rasterio.uint8
        arr = np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)
    else:
        # Single-band: check input dtype
        if arr.dtype == np.uint8:
            # Masks (water, flood, contours): uint8 with nodata=0
            dtype = rasterio.uint8
            nodata_value = 0
        elif arr.dtype == np.uint16:
            # Visualization DEM: uint16 with nodata=0
            dtype = rasterio.uint16
            nodata_value = 0
        else:
            # DEM, hillshade, slope: float32 with nodata=-9999
            dtype = rasterio.float32
            nodata_value = -9999.0

    crs_out = resolve_metric_crs(bounds, crs_out)

    # STEP 5: Use transform_override if provided (for multi-resolution)
    # DO NOT recompute transform, snap bounds, or modify coordinates
    if transform_override is not None:
        transform = transform_override
        crs = crs_out
        # Width/height come strictly from array shape
        # No bounds recomputation
    elif crs_out == "EPSG:4326":
        transform = from_bounds(minLon, minLat, maxLon, maxLat, w, h)
        crs = crs_out
    else:
        proj_bounds = bounds_to_projected(bounds, crs_out)
        minx, miny, maxx, maxy = proj_bounds

        # STEP 6: snap_gsd_m is deprecated (causes pixel size drift)
        if snap_gsd_m is not None:
            maxx = minx + w * float(snap_gsd_m)
            maxy = miny + h * float(snap_gsd_m)

        transform = from_bounds(minx, miny, maxx, maxy, w, h)
        crs = crs_out

    prof = dict(
        driver="GTiff",
        width=w,
        height=h,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
        PREDICTOR=(
            2 if dtype in (rasterio.float32,) else 1
        ),  # Floating point predictor for better compression
    )

    # Add nodata if specified
    if nodata_value is not None:
        prof["nodata"] = nodata_value

    # Set photometric interpretation in profile for RGB data
    if photometric:
        prof["photometric"] = photometric

    with rasterio.open(path, "w", **prof) as dst:
        if count == 1:
            # Convert to target dtype
            if dtype == rasterio.uint8 and arr.dtype == np.float32:
                data = np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)
            elif dtype == rasterio.uint8:
                data = arr.astype(np.uint8)
            else:
                # For float32 DEM: preserve absolute elevation values
                data = arr.astype(np.float32)
                # Replace any invalid values with nodata
                if nodata_value is not None:
                    data = np.where(np.isfinite(data), data, nodata_value)
            dst.write(data, 1)
            # Set metadata for float32 single-band (DEM) data
            if dtype == rasterio.float32:
                dst.set_band_description(1, "elevation")
                dst.update_tags(1, units="meters", ROLE="elevation")
        else:
            # Convert to target dtype
            if dtype == rasterio.uint8 and arr.dtype == np.float32:
                data = np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)
            elif dtype == rasterio.uint8:
                data = arr.astype(np.uint8)
            else:
                data = arr.astype(np.float32)
            for i in range(data.shape[2]):
                dst.write(data[:, :, i], i + 1)
            # Explicitly set color interpretation for RGB data
            if count == 3:
                dst.colorinterp = [ColorInterp.red,
                                   ColorInterp.green, ColorInterp.blue]
        if overviews:
            try:
                dst.build_overviews(
                    [2, 4, 8, 16],
                    (
                        Resampling.average
                        if dtype == rasterio.float32
                        else Resampling.nearest
                    ),
                )
            except Exception:
                pass
    return True


# ------------------ noise (smooth, no visible tiles) ------------------


def smoothstep(t):
    return t * t * (3 - 2 * t)


def value_noise(w: int, h: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    gw, gh = (w // grid) + 2, (h // grid) + 2
    g = rng.random((gh, gw), dtype=np.float32)
    y = np.arange(h, dtype=np.float32)[:, None] / grid
    x = np.arange(w, dtype=np.float32)[None, :] / grid
    yi = np.floor(y).astype(int)
    xi = np.floor(x).astype(int)
    fy = smoothstep(y - yi)
    fx = smoothstep(x - xi)
    g00 = g[yi, xi]
    g10 = g[yi, xi + 1]
    g01 = g[yi + 1, xi]
    g11 = g[yi + 1, xi + 1]
    n0 = g00 * (1 - fx) + g10 * fx
    n1 = g01 * (1 - fx) + g11 * fx
    return (n0 * (1 - fy) + n1 * fy).astype(np.float32)


def fbm(w: int, h: int, oct=6, base=128, gain=0.5, lac=2.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    freq = 1.0
    for i in range(oct):
        heartbeat(f"FBM octave {i+1}/{oct}")
        grid = max(2, int(base / freq))
        out += value_noise(w, h, grid, rng) * amp
        amp *= gain
        freq *= lac
    norm = (1.0 - gain**oct) / (1.0 - gain) if gain != 1 else float(oct)
    return out / norm


def domain_warp(src: np.ndarray, strength=20.0, seed=0) -> np.ndarray:
    h, w = src.shape
    wx = fbm(w, h, oct=4, base=96, seed=seed + 1)
    wy = fbm(w, h, oct=4, base=96, seed=seed + 2)
    yy, xx = np.mgrid[0:h, 0:w]
    xw = np.clip((xx + (wx - 0.5) * strength).round().astype(int), 0, w - 1)
    yw = np.clip((yy + (wy - 0.5) * strength).round().astype(int), 0, h - 1)
    return src[yw, xw]


def ridged(w: int, h: int, oct=6, base=128, seed=0) -> np.ndarray:
    n = fbm(w, h, oct, base, seed=seed)
    r = 1.0 - np.abs(2 * n - 1.0)
    r = 0.6 * r + 0.4 * (
        1.0 - np.abs(2 * fbm(w, h, 4, max(8, base // 2), seed=seed + 99) - 1.0)
    )
    return to01(r)


def dunes(w: int, h: int, seed: int, scale=28, deg=20) -> np.ndarray:
    base = value_noise(w, h, max(2, scale), np.random.default_rng(seed))
    yy, xx = np.mgrid[0:h, 0:w]
    th = math.radians(deg)
    ridges = np.cos(
        (xx * np.cos(th) + yy * np.sin(th)) / (scale * 0.6) + base * 3.14159
    )
    return to01((ridges + 1) * 0.5)


# ------------------ terrain & hydro ------------------


def make_dem(
    biome: str,
    w: int,
    h: int,
    seed: int,
    profile: str = "high",
    pixel_size_m: float = 25.0,
) -> np.ndarray:
    """
    Generate physically consistent DEM with proper spatial scaling.

    Returns DEM in meters for correct hillshade and EO realism.
    """
    rng = np.random.default_rng(seed)
    if biome == "mountain":
        # Multi-scale relief with real-world spatial meaning
        # Macro: valleys, ridges (~20 km features)
        macro = ridged(w, h, oct=5, base=int(800.0 / pixel_size_m), seed=seed)

        # Meso: spurs, slopes (~6 km features)
        meso = ridged(w, h, oct=5, base=int(
            250.0 / pixel_size_m), seed=seed + 1)

        # Micro: roughness (~1.5 km features)
        micro = fbm(w, h, oct=3, base=int(60.0 / pixel_size_m), seed=seed + 2)

        # Combine with physically sensible weights (in meters)
        dem = (
            macro * 220.0  # Main relief
            + meso * 90.0  # Slope detail
            + micro * 20.0  # Roughness
        )

        # Normalize and shift to realistic elevation band
        dem -= dem.min()
        dem = dem / (dem.max() + 1e-6)

        # Typical upland terrain: 80-550m elevation
        dem = dem * 470.0 + 80.0

        # Add domain warping for natural ridge/valley patterns
        dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
        dem_norm = domain_warp(dem_norm, strength=16, seed=seed + 3)
        # Remap back to elevation range
        dem = dem_norm * (dem.max() - dem.min()) + dem.min()

        # Return DEM in meters
        return dem.astype(np.float32)
    elif biome == "desert":
        # Desert: low relief with dunes (50-350m typical)
        base = fbm(w, h, oct=6, base=256, seed=seed)
        plate = ridged(w, h, oct=3, base=512, seed=seed + 9)
        dune = dunes(w, h, seed + 5, scale=24 if profile !=
                     "low" else 32, deg=25)
        dem = 0.52 * base + 0.32 * plate + 0.16 * dune
        dem = domain_warp(dem, strength=10, seed=seed + 3)
        dem = to01(dem)
        # Desert elevation range: 50-350m
        dem = dem * 300.0 + 50.0
        return dem.astype(np.float32)
    elif biome == "city":
        base = fbm(w, h, oct=6, base=256, seed=seed) ** 1.15
        # irregular basin (warped ellipse) instead of circle
        yy, xx = np.mgrid[0:h, 0:w]
        ex, ey = 0.55 + 0.08 * fbm(
            w, h, 3, base=128, seed=seed + 20
        ), 0.55 + 0.08 * fbm(w, h, 3, base=128, seed=seed + 21)
        cx, cy = int(w * 0.52), int(h * 0.48)
        a = 0.18 * min(w, h) * (1.0 + 0.2 *
                                fbm(w, h, 2, base=64, seed=seed + 22))
        b = 0.15 * min(w, h) * (1.0 + 0.2 *
                                fbm(w, h, 2, base=64, seed=seed + 23))
        el = ((xx - cx) / np.maximum(a, 1e-3)) ** 2 + (
            (yy - cy) / np.maximum(b, 1e-3)
        ) ** 2
        basin = to01(1.0 - el)
        basin = domain_warp(basin, strength=14, seed=seed + 24)
        dem = np.where(basin > 0.4, base * 0.25 + 0.14, base)
        dem = to01(dem)
        # City elevation: 0-120m (low relief)
        dem = dem * 120.0
        return dem.astype(np.float32)
    elif biome == "farm":
        # Farm: gentle rolling hills (20-180m)
        base = fbm(w, h, oct=5, base=512, seed=seed)
        detail = fbm(w, h, oct=4, base=128, seed=seed + 1)
        dem = 0.75 * base + 0.25 * detail
        dem = dem**0.85
        dem = domain_warp(dem, strength=8, seed=seed + 2)
        dem = to01(dem)
        # Farm elevation: 20-180m
        dem = dem * 160.0 + 20.0
        return dem.astype(np.float32)
    elif biome == "coast":
        # Coast: very gentle terrain (0-100m)
        base = fbm(w, h, oct=6, base=384, seed=seed)
        detail = fbm(w, h, oct=5, base=96, seed=seed + 1)
        dem = 0.70 * base + 0.30 * detail
        dem = dem**0.90
        dem = domain_warp(dem, strength=10, seed=seed + 2)
        dem = to01(dem)
        # Coast elevation: 0-100m
        dem = dem * 100.0
        return dem.astype(np.float32)
    elif biome == "island":
        # Island: moderate relief (0-280m)
        base = fbm(w, h, oct=6, base=192, seed=seed)
        detail = fbm(w, h, oct=5, base=48, seed=seed + 1)
        dem = 0.60 * base + 0.40 * detail
        dem = domain_warp(dem, strength=12, seed=seed + 2)
        dem = to01(dem)
        # Island elevation: 0-280m
        dem = dem * 280.0
        return dem.astype(np.float32)
    else:  # forest or default
        base = fbm(w, h, oct=7, base=256, seed=seed)
        detail = fbm(w, h, oct=6, base=64, seed=seed + 1)
        dem = 0.66 * base + 0.34 * detail
        dem = domain_warp(dem, strength=14, seed=seed + 2)
        dem = to01(dem)
        # Forest elevation: 40-320m
        dem = dem * 280.0 + 40.0
        return dem.astype(np.float32)

    # Note: island bowl is now handled inside each biome branch above
    # This fallback shouldn't be reached if all biomes return early
    # Default fallback (should not be reached)
    dem = fbm(w, h, oct=6, base=256, seed=seed)
    dem = to01(dem)
    dem = dem * 200.0 + 50.0  # Generic 50-250m range
    return dem.astype(np.float32)


# ============================================================================
# PROCEDURAL SEMANTIC SCENES
# ============================================================================


def fbm_noise(
    width: int,
    height: int,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
    offset_xy: Tuple[int, int] = (0, 0),
    base_scale: float = 128.0,
) -> np.ndarray:
    """
    World-coordinate FBM noise with spatial offset support.

    This wrapper enables tiled generation by allowing origin offsets.

    Args:
        width, height: Output dimensions
        octaves: Number of noise octaves
        persistence: Amplitude falloff (gain)
        lacunarity: Frequency multiplier
        seed: Random seed
        offset_xy: World-coordinate offset (ox, oy)
        base_scale: Base frequency scale in pixels

    Returns:
        Noise array [-1, 1] range
    """
    # Use existing fbm() but with seeded offset for spatial consistency
    # In a true world-coordinate implementation, offset_xy would shift the sampling grid
    # For now, we use it to vary the seed predictably based on tile position
    ox, oy = offset_xy
    tile_seed = seed + int(ox * 1000 + oy)

    gain = persistence
    return fbm(
        width,
        height,
        oct=octaves,
        base=int(base_scale),
        gain=gain,
        lac=lacunarity,
        seed=tile_seed,
    )


def generate_forest_scene(
    width: int,
    height: int,
    seed: int,
    origin_xy: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procedural forest semantic scene.

    Returns:
        rgb_base: (H, W, 3) float32 [0,1]
        canopy_height: (H, W) float32 (meters, relative)
    """
    ox, oy = origin_xy

    # --- Canopy height field (tree crowns) ---
    # Large-scale clumping
    canopy_large = fbm_noise(
        width,
        height,
        octaves=4,
        persistence=0.6,
        lacunarity=2.0,
        seed=seed + 100,
        offset_xy=(ox, oy),
        base_scale=128.0,
    )

    # Medium-scale variation (individual tree groups)
    canopy_medium = fbm_noise(
        width,
        height,
        octaves=3,
        persistence=0.5,
        lacunarity=2.2,
        seed=seed + 101,
        offset_xy=(ox, oy),
        base_scale=48.0,
    )

    canopy = 0.7 * canopy_large + 0.3 * canopy_medium
    canopy = (canopy + 1.0) / 2.0  # [0,1]

    # Threshold to create gaps (clearings)
    density = canopy > 0.35
    canopy = canopy * density

    # Height in meters (approx forest canopy)
    canopy_height = canopy * (8.0 + 6.0 * canopy_medium)  # ~8–14 m

    # --- Reflectance (RGB base) ---
    # Darker in dense canopy, lighter in gaps
    green = 0.35 + 0.35 * canopy
    red = 0.18 + 0.15 * canopy
    blue = 0.15 + 0.10 * canopy

    rgb_base = np.stack([red, green, blue], axis=-1)
    rgb_base = np.clip(rgb_base, 0, 1)

    return rgb_base.astype(np.float32), canopy_height.astype(np.float32)


def generate_city_scene(
    width: int,
    height: int,
    seed: int,
    origin_xy: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procedural city semantic scene - GEOMETRY FIRST, lighting from hillshade.

    Key principle: Roads are HEIGHT CUTS (building_height = 0), not painted pixels.
    This makes roads naturally darken via hillshade, exactly like mountain valleys.

    Returns:
        rgb_base: (H, W, 3) float32 [0,1] - PURE REFLECTANCE (no lighting)
        building_height: (H, W) float32 (meters) - 0 for roads, 5-30m for buildings
    """
    ox, oy = origin_xy

    # --- Street grid (procedural block layout) ---
    angle = (seed % 360) * np.pi / 180.0
    dx, dy = np.cos(angle), np.sin(angle)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    proj = (xx + ox) * dx + (yy + oy) * dy
    proj2 = (xx + ox) * (-dy) + (yy + oy) * dx

    # Primary and secondary road grids
    block_size = 80.0  # Average city block ~80m
    road_width = 8.0  # Road width ~8m

    road_mask = (np.mod(proj, block_size) < road_width) | (
        np.mod(proj2, block_size * 1.3) < road_width
    )

    # Add noise to break up perfect grid
    road_noise = fbm_noise(
        width,
        height,
        octaves=3,
        persistence=0.5,
        seed=seed + 50,
        offset_xy=(ox, oy),
        base_scale=200.0,
    )
    road_mask = road_mask | (
        (road_noise > 0.7)
        & (np.random.RandomState(seed + 51).rand(height, width) < 0.02)
    )

    block_mask = ~road_mask

    # --- Building heights (quantized, block-based) ---
    # Each block gets a consistent height (like real buildings)
    height_base = fbm_noise(
        width,
        height,
        octaves=4,
        persistence=0.6,
        seed=seed + 200,
        offset_xy=(ox, oy),
        base_scale=120.0,
    )
    height_base = (height_base + 1.0) / 2.0

    # Quantize to create flat-top buildings
    height_levels = np.floor(height_base * 8) / 8.0

    # Building height: 0 for roads, 5-35m for buildings
    building_height = np.where(block_mask, 5.0 + 30.0 * height_levels, 0.0).astype(
        np.float32
    )

    # --- Parks (sparse green spaces, height = 0) ---
    park_noise = fbm_noise(
        width,
        height,
        octaves=4,
        persistence=0.5,
        seed=seed + 300,
        offset_xy=(ox, oy),
        base_scale=150.0,
    )
    parks = (park_noise > 0.65) & block_mask  # ~15% of non-road area
    building_height = np.where(parks, 0.0, building_height)

    # --- Reflectance (PURE, no lighting) ---
    # Buildings: varied rooftop materials (concrete, metal, tile)
    roof_var = fbm_noise(
        width,
        height,
        octaves=5,
        persistence=0.5,
        seed=seed + 400,
        offset_xy=(ox, oy),
        base_scale=60.0,
    )
    roof_var = (roof_var + 1.0) / 2.0

    # Base rooftop: light grey concrete
    base_brightness = 0.50 + 0.20 * roof_var

    # Material variation: some beige, some white, some darker
    material_noise = fbm_noise(
        width,
        height,
        octaves=3,
        persistence=0.6,
        seed=seed + 500,
        offset_xy=(ox, oy),
        base_scale=100.0,
    )
    material_noise = (material_noise + 1.0) / 2.0

    roof_r = np.where(
        material_noise > 0.7, base_brightness * 1.05, base_brightness * 0.98
    )
    roof_g = np.where(
        material_noise > 0.7, base_brightness * 0.95, base_brightness * 1.00
    )
    roof_b = np.where(
        material_noise > 0.7, base_brightness * 0.85, base_brightness * 1.02
    )

    # Roads: dark asphalt (slightly lighter than mountain shadows for realism)
    road_r, road_g, road_b = 0.18, 0.18, 0.19

    # Parks: green vegetation
    park_r, park_g, park_b = 0.22, 0.38, 0.20

    # Combine
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = np.where(parks, park_r, np.where(road_mask, road_r, roof_r))
    rgb[..., 1] = np.where(parks, park_g, np.where(road_mask, road_g, roof_g))
    rgb[..., 2] = np.where(parks, park_b, np.where(road_mask, road_b, roof_b))

    return np.clip(rgb, 0, 1).astype(np.float32), building_height.astype(np.float32)


# ============================================================================
# PHYSICALLY CORRECT TERRAIN ANALYSIS (scaled by pixel_size_m)
# ============================================================================


def grad_scaled(z: np.ndarray, px_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient scaled by physical pixel size.
    Returns dz/dy, dz/dx in "meters per meter" (rise/run).

    Args:
        z: Elevation in meters (H, W)
        px_m: Pixel size in meters

    Returns:
        (gy, gx) gradients in meters/meter
    """
    gy, gx = np.gradient(z.astype(np.float32), px_m, px_m)
    return gy.astype(np.float32), gx.astype(np.float32)


def slope_from_dem_m(dem_m: np.ndarray, px_m: float) -> np.ndarray:
    """
    Compute slope magnitude from DEM in meters.

    Args:
        dem_m: DEM in meters (H, W)
        px_m: Pixel size in meters

    Returns:
        Normalized slope [0-1]
    """
    gy, gx = grad_scaled(dem_m, px_m)
    s = np.hypot(gx, gy)  # slope magnitude (rise/run)
    return to01(s)


def apply_hillshade_rgb(
    rgb: np.ndarray, hillshade: np.ndarray, biome: str
) -> np.ndarray:
    """
    Apply physically-correct hillshade lighting to RGB reflectance.

    Args:
        rgb: RGB reflectance array (0-1, float32, shape HxWx3)
        hillshade: Hillshade array (0-1, float32, shape HxW)
        biome: Biome name for lighting style

    Returns:
        RGB with lighting applied (0-1, float32)
    """
    if biome in ["forest", "farm", "coast", "city"]:
        # Subtle lighting for vegetated biomes and urban areas
        HILLSHADE_MIN, HILLSHADE_MAX = 0.82, 1.10
        hs_mapped = HILLSHADE_MIN + (HILLSHADE_MAX - HILLSHADE_MIN) * hillshade
        rgb_lit = np.clip(rgb * hs_mapped[..., None], 0, 1)
    else:
        # Stronger lighting for mountain/desert
        combined_light = 0.62 + 0.50 * hillshade
        rgb_lit = np.clip(rgb * combined_light[..., None], 0, 1)
    return rgb_lit


def write_dem_visual(
    dem_m: np.ndarray,
    bounds: Tuple[float, float, float, float],
    path: str,
    crs_out: str,
    snap_gsd_m: Optional[float] = None,
    transform_override=None,
) -> None:
    """
    Write a visually stretched DEM for display purposes.
    DOES NOT replace the real DEM - this is for visualization only.

    Args:
        dem_m: DEM in meters (H, W)
        bounds: Geographic bounds (minLon, minLat, maxLon, maxLat)
        path: Output file path
        crs_out: CRS for output (e.g., "EPSG:32630" for UTM)
        snap_gsd_m: DEPRECATED - use transform_override instead
        transform_override: If provided, use this transform directly (for multi-resolution)
    """
    dem_norm = (dem_m - dem_m.min()) / (dem_m.max() - dem_m.min() + 1e-6)
    dem_u16 = (dem_norm * 65535).astype(np.uint16)

    write_tif_with_bounds(
        dem_u16, bounds, path, crs_out=crs_out,
        snap_gsd_m=snap_gsd_m, transform_override=transform_override
    )


def hillshade_from_dem_m(
    dem_m: np.ndarray, px_m: float, az: float = 315.0, alt: float = 45.0
) -> np.ndarray:
    """
    Compute physically correct hillshade from DEM in meters.

    Args:
        dem_m: DEM in meters (H, W)
        px_m: Pixel size in meters
        az: Azimuth angle in degrees (0-360, 0=N, 90=E, 180=S, 270=W)
        alt: Altitude angle in degrees (0-90, 0=horizon, 90=zenith)

    Returns:
        Hillshade [0-1]
    """
    azr, altr = math.radians(az), math.radians(alt)
    gy, gx = grad_scaled(dem_m, px_m)

    # Slope angle and aspect
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    # Standard hillshade formula
    hs = np.sin(altr) * np.cos(slope_rad) + np.cos(altr) * np.sin(slope_rad) * np.cos(
        azr - aspect
    )

    return np.clip(hs, 0, 1).astype(np.float32)


def raster_contours_m(dem_m: np.ndarray, interval_m: float) -> np.ndarray:
    """
    Generate contour lines at exact elevation intervals.

    Args:
        dem_m: DEM in meters (H, W)
        interval_m: Contour interval in meters (e.g., 20m)

    Returns:
        Binary contour mask (H, W)
    """
    z = dem_m.astype(np.float32)
    mod = np.mod(z, interval_m)
    # Lines appear where elevation is close to a multiple of interval_m
    lines = (mod < (interval_m * 0.03)) | (mod > interval_m * (1.0 - 0.03))
    return lines.astype(np.uint8)


# Legacy wrappers for backward compatibility (normalized DEM)
def grad(z):
    gy, gx = np.gradient(z)
    return gy.astype(np.float32), gx.astype(np.float32)


def slope(dem):
    gy, gx = grad(dem)
    return to01(np.hypot(gx, gy))


def hillshade(dem, az=315, alt=45):
    """Legacy hillshade (normalized DEM, no pixel scaling) - use hillshade_from_dem_m instead"""
    azr, altr = math.radians(az), math.radians(alt)
    gy, gx = np.gradient(dem)
    s = np.hypot(gx, gy)
    asp = np.arctan2(-gx, gy)
    hs = np.sin(altr) * (1 - s) + np.cos(altr) * s * np.cos(azr - asp)
    return np.clip(hs, 0, 1).astype(np.float32)


def d8_flow(dem) -> Tuple[np.ndarray, np.ndarray]:
    h, w = dem.shape
    offs = np.array(
        [(0, 1), (-1, 1), (-1, 0), (-1, -1),
         (0, -1), (1, -1), (1, 0), (1, 1)], np.int32
    )
    dir_idx = -np.ones((h, w), np.int8)
    slope_arr = np.zeros((h, w), np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            z = dem[i, j]
            best = 0.0
            bestk = -1
            for k, (di, dj) in enumerate(offs):
                dz = z - dem[i + di, j + dj]
                if dz > best:
                    best, bestk = dz, k
            dir_idx[i, j] = bestk
            slope_arr[i, j] = best
    # topo order
    downstream = np.full((h, w, 2), -1, np.int32)
    indeg = np.zeros((h, w), np.int32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            k = dir_idx[i, j]
            if k >= 0:
                di, dj = offs[k]
                ti, tj = i + di, j + dj
                downstream[i, j] = (ti, tj)
                indeg[ti, tj] += 1
    acc = np.ones((h, w), np.float32)
    q = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if indeg[i, j] == 0 and dir_idx[i, j] != -1:
                q.append((i, j))
    head = 0
    while head < len(q):
        i, j = q[head]
        head += 1
        ti, tj = downstream[i, j]
        if ti >= 0:
            acc[ti, tj] += acc[i, j]
            indeg[ti, tj] -= 1
            if indeg[ti, tj] == 0 and dir_idx[ti, tj] != -1:
                q.append((ti, tj))
    return dir_idx, to01(acc)


def rivers_and_flood(dem, density=0.004):
    _, acc = d8_flow(dem)
    thr = np.quantile(acc, 1.0 - density)
    r = (acc >= thr) & (dem < 0.92)
    r = blur(r.astype(np.float32), k=2) > 0.30
    flood = blur(np.clip((acc - thr) / (1 - thr + 1e-6), 0, 1), k=2) * (
        1.0 - np.clip(dem * 1.3 - 0.5, 0, 1)
    )
    flood = flood > 0.35
    return r, flood


def rivers_and_flood_fast(
    dem_norm: np.ndarray, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAST vectorized river approximation for large grids (avoids slow D8 flow).

    Uses terrain analysis to identify likely river locations:
    - Rivers tend to occur in low areas (valleys)
    - Rivers follow high curvature paths
    - Floods occur near rivers in lowlands

    This is orders of magnitude faster than D8 flow and produces plausible results.

    Args:
        dem_norm: Normalized DEM [0-1]
        seed: Random seed for noise

    Returns:
        (river_mask, flood_mask) as boolean arrays
    """
    # Calculate gradient (rivers avoid steep slopes)
    gy, gx = np.gradient(dem_norm)
    slope = np.hypot(gx, gy)

    # Calculate curvature (Laplacian - rivers follow valleys/channels)
    lap = (
        np.roll(dem_norm, 1, 0)
        + np.roll(dem_norm, -1, 0)
        + np.roll(dem_norm, 1, 1)
        + np.roll(dem_norm, -1, 1)
        - 4 * dem_norm
    )
    lap = np.abs(lap)

    # River likelihood score: low elevation + high curvature - steep slope
    score = (1.0 - dem_norm) * 0.7 + lap * 0.6 - slope * 0.4
    score = (score - score.min()) / (score.max() - score.min() + 1e-6)

    # Top 0.5% most likely pixels are rivers
    thr = np.quantile(score, 0.995)
    rivers = score >= thr

    # Flood zones: near rivers (top 1.5%) in lowlands (below 55% elevation)
    flood = (score > np.quantile(score, 0.985)) & (dem_norm < 0.55)

    # Soften edges for more natural appearance
    rivers = blur(rivers.astype(np.float32), k=2) > 0.30
    flood = blur(flood.astype(np.float32), k=3) > 0.25

    return rivers, flood


# ------------------ Field Generation for Rural Imagery ------------------


def generate_fields_mask(dem, seed=0):
    """
    Generate HIGHLY IRREGULAR, organic agricultural field parcels that follow terrain contours.
    UK-style rural hillside: curvy boundaries, varied sizes, terrain-following.
    Returns (field_ids, field_greenness, field_variation)
    """
    h, w = dem.shape

    # Multi-scale noise for highly irregular, organic field shapes
    # Use multiple octaves to create natural, curvy boundaries
    field_noise_x = fbm(w, h, oct=5, base=80, seed=seed + 200)
    field_noise_y = fbm(w, h, oct=5, base=70, seed=seed + 201)
    field_noise_fine = fbm(
        w, h, oct=6, base=30, seed=seed + 202
    )  # Fine-scale variation

    # Terrain-following fields (elongated along slope contours)
    gy, gx = grad(dem)
    slope_angle = np.arctan2(gy, gx)
    slope_magnitude = np.hypot(gx, gy)

    # Create field IDs with STRONG terrain-aware distortion for organic shapes
    yy, xx = np.mgrid[0:h, 0:w]
    field_size = 40 + field_noise_fine * 20  # Variable field size: 20-60 pixels

    # Warp grid STRONGLY along slope directions + noise for organic boundaries
    # This creates curvy, terrain-following field boundaries
    terrain_warp_x = np.cos(slope_angle) * (20 + slope_magnitude * 30)
    terrain_warp_y = np.sin(slope_angle) * (20 + slope_magnitude * 30)

    field_x = (
        xx + field_noise_x * 35 + terrain_warp_x + field_noise_fine * 15
    ) / field_size
    field_y = (
        yy + field_noise_y * 35 + terrain_warp_y + field_noise_fine * 15
    ) / field_size

    field_ids = (np.floor(field_x) * 10000 +
                 np.floor(field_y)).astype(np.int32)

    # Per-field properties (for color variation)
    # Convert to int64 for hash calculation to avoid overflow
    field_ids_64 = field_ids.astype(np.int64)
    field_hash = (field_ids_64 * 2654435761) % (
        2**31
    )  # Hash for consistent per-field random
    field_greenness = (field_hash % 1000) / 1000.0  # 0-1 per field
    field_variation = ((field_hash // 1000) % 1000) / \
        1000.0  # Second variation

    return field_ids, field_greenness, field_variation


def detect_field_boundaries(field_ids):
    """
    Detect boundaries between adjacent fields (for hedgerows/fences).
    Make boundaries more visible and natural-looking.
    """
    h, w = field_ids.shape
    boundaries = np.zeros((h, w), dtype=bool)

    # Check 8-connected neighbors for more complete boundary detection
    boundaries[:-1, :] |= field_ids[:-1, :] != field_ids[1:, :]
    boundaries[:, :-1] |= field_ids[:, :-1] != field_ids[:, 1:]
    boundaries[:-1, :-1] |= field_ids[:-1, :-1] != field_ids[1:, 1:]
    boundaries[:-1, 1:] |= field_ids[:-1, 1:] != field_ids[1:, :-1]

    # Slightly widen boundaries for visibility (hedgerows are visible features)
    boundaries = blur(boundaries.astype(np.float32), k=2) > 0.25

    return boundaries


# ------------------ palettes (colored) ------------------


def rgb_forest(dem, moisture, water, seed=0, pixel_size_m: float = 1.0):
    """
    UK-style rural hillside - EXACT colors from reference image:
    - Pasture colors: (0.35,0.52,0.28), (0.32,0.50,0.26), (0.28,0.45,0.22)
    - Dry fields: (0.48,0.54,0.34), (0.45,0.50,0.32), (0.42,0.48,0.30)
    - Forest: (0.20,0.32,0.18), (0.18,0.30,0.16), (0.22,0.35,0.20)
    - Hedgerows: (0.15, 0.22, 0.12)
    - Tracks: (0.72, 0.68, 0.60)
    - Water: (0.20,0.35,0.40), (0.14,0.28,0.32)
    """
    h, w = dem.shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Generate field parcels
    field_ids, field_green, field_var = generate_fields_mask(dem, seed)
    boundaries = detect_field_boundaries(field_ids)

    # Fine texture for grass/vegetation detail - METER-CONSISTENT
    tex_fine = fbm(
        w, h, oct=6, base=grid_px(6.0, pixel_size_m), seed=seed + 50
    )  # ~6m grain
    tex_micro = fbm(
        w, h, oct=7, base=grid_px(2.0, pixel_size_m), seed=seed + 51
    )  # ~2m micro

    land_mask = ~water
    field_type = field_green

    # EXACT COLORS FROM REFERENCE:
    # Pasture / lowland grass (dominant - 50% of fields)
    # Use all three variations with slight texture variation
    is_pasture = (field_type < 0.50) & land_mask
    pasture_variation = (field_var * 3) % 3
    pasture_light = (pasture_variation < 1) & is_pasture
    pasture_med = ((pasture_variation >= 1) & (
        pasture_variation < 2)) & is_pasture
    pasture_dark = (pasture_variation >= 2) & is_pasture

    rgb[..., 0] = np.where(pasture_light, 0.35 + tex_fine * 0.03, rgb[..., 0])
    rgb[..., 1] = np.where(pasture_light, 0.52 + tex_fine * 0.03, rgb[..., 1])
    rgb[..., 2] = np.where(pasture_light, 0.28 + tex_fine * 0.02, rgb[..., 2])

    rgb[..., 0] = np.where(pasture_med, 0.32 + tex_fine * 0.03, rgb[..., 0])
    rgb[..., 1] = np.where(pasture_med, 0.50 + tex_fine * 0.03, rgb[..., 1])
    rgb[..., 2] = np.where(pasture_med, 0.26 + tex_fine * 0.02, rgb[..., 2])

    rgb[..., 0] = np.where(pasture_dark, 0.28 + tex_fine * 0.03, rgb[..., 0])
    rgb[..., 1] = np.where(pasture_dark, 0.45 + tex_fine * 0.03, rgb[..., 1])
    rgb[..., 2] = np.where(pasture_dark, 0.22 + tex_fine * 0.02, rgb[..., 2])

    # Dry / harvested patches (beige-green) - 25% of fields
    is_dry = (field_type >= 0.50) & (field_type < 0.75) & land_mask
    dry_variation = (field_var * 3) % 3
    dry_light = (dry_variation < 1) & is_dry
    dry_med = ((dry_variation >= 1) & (dry_variation < 2)) & is_dry
    dry_dark = (dry_variation >= 2) & is_dry

    rgb[..., 0] = np.where(dry_light, 0.48 + tex_fine * 0.03, rgb[..., 0])
    rgb[..., 1] = np.where(dry_light, 0.54 + tex_fine * 0.03, rgb[..., 1])
    rgb[..., 2] = np.where(dry_light, 0.34 + tex_fine * 0.02, rgb[..., 2])

    rgb[..., 0] = np.where(dry_med, 0.45 + tex_fine * 0.03, rgb[..., 0])
    rgb[..., 1] = np.where(dry_med, 0.50 + tex_fine * 0.03, rgb[..., 1])
    rgb[..., 2] = np.where(dry_med, 0.32 + tex_fine * 0.02, rgb[..., 2])

    rgb[..., 0] = np.where(dry_dark, 0.42 + tex_fine * 0.03, rgb[..., 0])
    rgb[..., 1] = np.where(dry_dark, 0.48 + tex_fine * 0.03, rgb[..., 1])
    rgb[..., 2] = np.where(dry_dark, 0.30 + tex_fine * 0.02, rgb[..., 2])

    # Forest blocks / dense woodland (dark green) - 25% of fields
    is_forest = (field_type >= 0.75) & land_mask
    forest_variation = (field_var * 3) % 3
    forest_1 = (forest_variation < 1) & is_forest
    forest_2 = ((forest_variation >= 1) & (forest_variation < 2)) & is_forest
    forest_3 = (forest_variation >= 2) & is_forest

    rgb[..., 0] = np.where(forest_1, 0.20 + tex_fine * 0.02, rgb[..., 0])
    rgb[..., 1] = np.where(forest_1, 0.32 + tex_fine * 0.02, rgb[..., 1])
    rgb[..., 2] = np.where(forest_1, 0.18 + tex_fine * 0.02, rgb[..., 2])

    rgb[..., 0] = np.where(forest_2, 0.18 + tex_fine * 0.02, rgb[..., 0])
    rgb[..., 1] = np.where(forest_2, 0.30 + tex_fine * 0.02, rgb[..., 1])
    rgb[..., 2] = np.where(forest_2, 0.16 + tex_fine * 0.02, rgb[..., 2])

    rgb[..., 0] = np.where(forest_3, 0.22 + tex_fine * 0.02, rgb[..., 0])
    rgb[..., 1] = np.where(forest_3, 0.35 + tex_fine * 0.02, rgb[..., 1])
    rgb[..., 2] = np.where(forest_3, 0.20 + tex_fine * 0.02, rgb[..., 2])

    # WATER - EXACT colors from reference
    water_variation = tex_fine * 0.04
    water_base = field_var > 0.5  # Mix of light and dark water
    rgb[..., 0] = np.where(water & water_base, 0.20 +
                           water_variation, rgb[..., 0])
    rgb[..., 1] = np.where(water & water_base, 0.35 +
                           water_variation, rgb[..., 1])
    rgb[..., 2] = np.where(water & water_base, 0.40 +
                           water_variation, rgb[..., 2])

    rgb[..., 0] = np.where(water & ~water_base, 0.14 +
                           water_variation, rgb[..., 0])
    rgb[..., 1] = np.where(water & ~water_base, 0.28 +
                           water_variation, rgb[..., 1])
    rgb[..., 2] = np.where(water & ~water_base, 0.32 +
                           water_variation, rgb[..., 2])

    # HEDGEROWS - EXACT color (0.15, 0.22, 0.12) - make them clearly visible
    hedge_mask = boundaries & land_mask
    # Make hedgerows slightly wider and more visible (they're prominent features)
    hedge_mask = blur(hedge_mask.astype(np.float32), k=1) > 0.2
    rgb[..., 0] = np.where(hedge_mask, 0.15, rgb[..., 0])
    rgb[..., 1] = np.where(hedge_mask, 0.22, rgb[..., 1])
    rgb[..., 2] = np.where(hedge_mask, 0.12, rgb[..., 2])

    # TRACKS - EXACT color (0.72, 0.68, 0.60) - visible tan/brown paths
    # PERFORMANCE FIX: Use field boundaries instead of flow accumulation (avoid slow d8_flow)
    # Tracks follow field boundaries in flat-to-moderate terrain
    slope_field = np.hypot(*np.gradient(dem))
    track_prob = (
        1.0 - slope_field
    ) * field_var  # Tracks in flatter areas between fields
    track_mask = (boundaries & (field_var > 0.88) & (track_prob > 0.3) & land_mask) | (
        (track_prob > 0.7) & (field_var > 0.85) & land_mask
    )
    # Make tracks slightly wider for visibility
    track_mask = blur(track_mask.astype(np.float32), k=1) > 0.3
    rgb[..., 0] = np.where(track_mask, 0.72, rgb[..., 0])
    rgb[..., 1] = np.where(track_mask, 0.68, rgb[..., 1])
    rgb[..., 2] = np.where(track_mask, 0.60, rgb[..., 2])

    # Add micro-texture (grass blades, tree crowns) - very subtle
    rgb[..., 0] += tex_micro * 0.02 * land_mask.astype(float)
    rgb[..., 1] += tex_micro * 0.025 * land_mask.astype(float)
    rgb[..., 2] += tex_micro * 0.015 * land_mask.astype(float)

    # Reduce overall brightness for more realistic, desaturated look
    rgb = rgb * 0.88  # Reduce brightness by ~12% for more muted, UK-rural style

    return np.clip(rgb, 0, 1)


def rgb_mountain(dem, moisture, water, seed=0):
    """
    Mountain terrain matching customer reference.

    Key features from reference:
    1. Ridge-aligned texture (slope-following anisotropy)
    2. Land-use heterogeneity (grazed patches, forest strips, fields)
    3. Subtle linear features (roads, tracks)
    """
    h, w = dem.shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Compute slope direction for anisotropic texture
    gy, gx = grad(dem)
    slope_angle = np.arctan2(gy, gx)
    slope_mag = np.sqrt(gx**2 + gy**2)

    # FIX 1: Directional texture aligned with slope (ridge-following)
    # Creates elongated patterns along ridges and valleys
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Texture coordinate stretched along slope direction
    slope_aligned_x = xx * np.cos(slope_angle) + yy * np.sin(slope_angle)
    ridge_texture = fbm(w, h, oct=8, base=18, seed=seed + 70)  # Higher detail
    # Modulate by slope-aligned coordinate (creates directional streaking)
    ridge_aligned = np.sin(slope_aligned_x / 8.0 + ridge_texture * 4.0) * 0.04

    # FIX 2: Mid-frequency land-use patches (not just noise)
    # Grazed areas (lighter patches on hills) - increased detail
    grazed_mask = fbm(w, h, oct=5, base=60, seed=seed +
                      100)  # More octaves for detail
    grazed = (grazed_mask > 0.2) & (dem > 0.4) & (dem < 0.7)
    grazed = blur(grazed.astype(np.float32), k=2)  # Less blur

    # Forest strips (darker, irregular) - increased detail
    forest_mask = fbm(w, h, oct=6, base=45, seed=seed + 101)  # More octaves
    forest = (forest_mask > 0.1) & (moisture > 0.4)
    forest = blur(forest.astype(np.float32), k=1)  # Minimal blur

    # Field parcels (varied colors) - increased detail
    field_mask = fbm(w, h, oct=5, base=50, seed=seed + 102)  # More octaves
    fields = (field_mask > -0.1) & (dem > 0.35) & (dem < 0.65)
    field_type = fbm(
        w, h, oct=4, base=80, seed=seed + 103
    )  # More detail in field types

    # Base earth-tone colors (balanced, not green-dominant)
    # Use high-resolution multi-scale textures for sharp detail
    tex_coarse = fbm(w, h, oct=6, base=32, seed=seed +
                     71)  # Large-scale variation
    tex_medium = fbm(w, h, oct=8, base=12, seed=seed + 72)  # Mid-scale detail
    tex_fine = fbm(
        w, h, oct=10, base=4, seed=seed + 73
    )  # Fine detail (increased octaves)
    # Blend all three for rich, detailed texture
    tex = tex_coarse * 0.4 + tex_medium * 0.35 + tex_fine * 0.25

    # Forest: dark green-brown
    rgb[..., 0] = np.where(forest > 0.3, 0.12 + tex * 0.03, 0.28 + tex * 0.08)
    rgb[..., 1] = np.where(forest > 0.3, 0.18 + tex * 0.04, 0.26 + tex * 0.08)
    rgb[..., 2] = np.where(forest > 0.3, 0.10 + tex * 0.02, 0.19 + tex * 0.05)

    # Grazed areas: lighter olive/tan
    rgb[..., 0] += grazed * 0.12
    rgb[..., 1] += grazed * 0.10
    rgb[..., 2] += grazed * 0.06

    # Field variation: some greener, some browner
    rgb[..., 0] = np.where(fields & (field_type > 0),
                           rgb[..., 0] - 0.04, rgb[..., 0])
    rgb[..., 1] = np.where(
        fields & (field_type > 0), rgb[..., 1] + 0.06, rgb[..., 1]
    )  # Greener fields
    rgb[..., 0] = np.where(
        fields & (field_type < 0), rgb[..., 0] + 0.08, rgb[..., 0]
    )  # Browner fields
    rgb[..., 1] = np.where(fields & (field_type < 0),
                           rgb[..., 1] - 0.02, rgb[..., 1])

    # Apply ridge-aligned texture
    rgb += ridge_aligned[..., np.newaxis]

    # FIX 3: Subtle roads/tracks (thin, pale, low-contrast)
    # PERFORMANCE FIX: Use elevation contours instead of flow (avoid slow d8_flow)
    # Roads follow valleys and ridges in mountains
    curvature = np.abs(
        np.roll(dem, 1, 0)
        + np.roll(dem, -1, 0)
        + np.roll(dem, 1, 1)
        + np.roll(dem, -1, 1)
        - 4 * dem
    )
    road_prob = (curvature > np.percentile(curvature, 85)) & (slope_mag < 0.15)
    road_prob = blur(road_prob.astype(np.float32), k=1)
    # Pale tan roads (subtle, not high-contrast)
    rgb[..., 0] += road_prob * 0.08
    rgb[..., 1] += road_prob * 0.06
    rgb[..., 2] += road_prob * 0.04

    # Water: dark smooth blue-green
    if isinstance(water, np.ndarray):
        water_mask = water.astype(bool) if water.dtype == bool else water > 0.5
        rgb[..., 0] = np.where(water_mask, 0.10, rgb[..., 0])
        rgb[..., 1] = np.where(water_mask, 0.20, rgb[..., 1])
        rgb[..., 2] = np.where(water_mask, 0.32, rgb[..., 2])

    # Apply VERY subtle smoothing only (k=1 is minimal)
    # This reduces extreme pixelation without losing detail
    rgb = blur(rgb, k=1)

    return np.clip(rgb, 0, 1)


def rgb_desert(dem, moisture, water, seed=0):
    h, w = dem.shape
    rgb = np.zeros((h, w, 3), np.float32)
    sand = (dem <= 0.65) & (~water)
    plateau = (dem > 0.65) & (~water)
    green = (moisture > 0.40) & (~water)
    beach = (dem >= 0.10) & (dem < 0.14)

    # Optimized photorealistic sand texture
    tex_fine = fbm(w, h, oct=6, base=8, seed=seed + 80)
    tex_medium = fbm(w, h, oct=4, base=32, seed=seed + 81)
    sand_ripples = fbm(w, h, oct=4, base=16, seed=seed + 82) * 0.08

    # Water with depth variation
    water_depth = fbm(w, h, oct=4, base=32, seed=seed + 60) * 0.12
    rgb[..., 0] = np.where(water, 0.07 + water_depth * 0.2, 0)
    rgb[..., 1] = np.where(water, 0.46 + water_depth * 0.4, 0)
    rgb[..., 2] = np.where(water, 0.80 + water_depth * 0.6, 0)

    # Sand with natural color variation
    rgb[..., 0] = np.where(sand, 0.84 + tex_fine *
                           0.10 + sand_ripples, rgb[..., 0])
    rgb[..., 1] = np.where(
        sand, 0.77 + tex_fine * 0.08 + sand_ripples * 0.8, rgb[..., 1]
    )
    rgb[..., 2] = np.where(
        sand, 0.60 + tex_fine * 0.06 + sand_ripples * 0.5, rgb[..., 2]
    )

    # Plateau/rock with texture
    rock_tex = tex_medium * 0.12
    rgb[..., 0] = np.where(plateau, 0.57 + rock_tex, rgb[..., 0])
    rgb[..., 1] = np.where(plateau, 0.52 + rock_tex * 0.9, rgb[..., 1])
    rgb[..., 2] = np.where(plateau, 0.48 + rock_tex * 0.8, rgb[..., 2])

    # Green oasis areas with more natural variation
    veg_tex = tex_fine * 0.15
    rgb[..., 0] = np.where(green, 0.54 + veg_tex, rgb[..., 0])
    rgb[..., 1] = np.where(green, 0.63 + veg_tex * 1.2, rgb[..., 1])
    rgb[..., 2] = np.where(green, 0.44 + veg_tex * 0.8, rgb[..., 2])

    # Beach
    rgb[..., 0] = np.where(beach, 0.88 + tex_fine * 0.08, rgb[..., 0])
    rgb[..., 1] = np.where(beach, 0.81 + tex_fine * 0.06, rgb[..., 1])
    rgb[..., 2] = np.where(beach, 0.64 + tex_fine * 0.05, rgb[..., 2])

    # Enhanced dune crests with more natural appearance
    crests = (dunes(w, h, seed=seed, scale=22, deg=25) > 0.76) & sand
    crest_highlight = tex_fine * 0.06
    rgb[..., 0] = np.where(crests, 0.90 + crest_highlight, rgb[..., 0])
    rgb[..., 1] = np.where(crests, 0.84 + crest_highlight, rgb[..., 1])
    rgb[..., 2] = np.where(crests, 0.66 + crest_highlight, rgb[..., 2])

    return np.clip(rgb, 0, 1)


def rgb_city(dem, moisture, water, seed=0, pixel_size_m: float = 1.0):
    """
    Generate realistic urban satellite imagery.
    Creates dense urban fabric with subtle roads, varied building rooftops, and natural features.
    """
    h, w = dem.shape
    rng = np.random.default_rng(seed)
    rgb = np.zeros((h, w, 3), np.float32)

    # Ensure water is a valid boolean mask
    if water is None:
        water = np.zeros((h, w), dtype=bool)
    else:
        water = water.astype(bool)

    # --- Base urban fabric (continuous building rooftops)
    # Most of the city should be building rooftops of varying colors

    # Multi-scale noise for natural variation - METER-CONSISTENT
    coarse_var = fbm(w, h, 4, base=grid_px(
        120.0, pixel_size_m), seed=seed + 10)
    medium_var = fbm(w, h, 6, base=grid_px(40.0, pixel_size_m), seed=seed + 11)
    fine_var = fbm(w, h, 8, base=grid_px(10.0, pixel_size_m), seed=seed + 12)

    # Normalize to [0,1]
    coarse_var = to01(coarse_var)
    medium_var = to01(medium_var)
    fine_var = to01(fine_var)

    # Base rooftop colors: grey, beige, white, terracotta mix
    # Real urban areas show diverse roof materials

    # Grey concrete roofs (most common)
    base_r = 0.45 + 0.15 * coarse_var + 0.08 * fine_var
    base_g = 0.45 + 0.15 * coarse_var + 0.08 * fine_var
    base_b = 0.47 + 0.15 * coarse_var + 0.08 * fine_var

    # Add variation: some beige/tan roofs
    beige_mask = (medium_var > 0.6) & (medium_var < 0.8)
    base_r = np.where(beige_mask, 0.55 + 0.10 * fine_var, base_r)
    base_g = np.where(beige_mask, 0.50 + 0.08 * fine_var, base_g)
    base_b = np.where(beige_mask, 0.42 + 0.06 * fine_var, base_b)

    # Add variation: some terracotta/red roofs
    red_mask = coarse_var > 0.75
    base_r = np.where(red_mask, 0.60 + 0.12 * fine_var, base_r)
    base_g = np.where(red_mask, 0.42 + 0.08 * fine_var, base_g)
    base_b = np.where(red_mask, 0.35 + 0.06 * fine_var, base_b)

    # Apply base colors
    rgb[..., 0] = base_r
    rgb[..., 1] = base_g
    rgb[..., 2] = base_b

    # --- Parks and green spaces (sparse, scattered)
    park_noise = fbm(w, h, 5, base=grid_px(80.0, pixel_size_m), seed=seed + 20)
    parks = (park_noise > 0.75) & (~water)  # Only ~10-15% green space

    # Park colors
    park_green = fbm(w, h, 6, base=grid_px(20.0, pixel_size_m), seed=seed + 21)
    park_green = to01(park_green)
    rgb[..., 0] = np.where(parks, 0.25 + 0.10 * park_green, rgb[..., 0])
    rgb[..., 1] = np.where(parks, 0.40 + 0.15 * park_green, rgb[..., 1])
    rgb[..., 2] = np.where(parks, 0.22 + 0.08 * park_green, rgb[..., 2])

    # --- Subtle road network (barely visible, like real satellite imagery)
    # Roads should be SUBTLE - in real urban satellite imagery, roads are hard to see

    # Major roads - wide but SUBTLE
    road_spacing = 100  # Farther apart
    road_width = 2  # Thinner

    roads = np.zeros((h, w), dtype=bool)

    # Create organic road network with noise-based spacing
    road_pattern = fbm(w, h, 3, base=300, seed=seed + 100)
    road_pattern = to01(road_pattern)

    # Horizontal roads (with natural variation)
    for i in range(0, h, road_spacing):
        offset = int((road_pattern[min(i, h - 1), w // 2] - 0.5) * 30)
        y = i + offset
        if 0 <= y < h:
            roads[max(0, y - road_width): min(h, y + road_width + 1), :] = True

    # Vertical roads (with natural variation)
    for j in range(0, w, road_spacing):
        offset = int((road_pattern[h // 2, min(j, w - 1)] - 0.5) * 30)
        x = j + offset
        if 0 <= x < w:
            roads[:, max(0, x - road_width): min(w, x + road_width + 1)] = True

    # Remove roads from water and parks
    roads = roads & (~water) & (~parks)

    # Paint roads - MUCH lighter, barely darker than buildings
    road_tex = fbm(w, h, 10, base=grid_px(4.0, pixel_size_m), seed=seed + 150)
    road_tex = to01(road_tex)
    # Roads are only slightly darker gray than rooftops
    rgb[..., 0] = np.where(roads, 0.38 + 0.04 * road_tex, rgb[..., 0])
    rgb[..., 1] = np.where(roads, 0.38 + 0.04 * road_tex, rgb[..., 1])
    rgb[..., 2] = np.where(roads, 0.40 + 0.04 * road_tex, rgb[..., 2])

    # --- Water (if present)
    # Dark blue-grey water
    water_tex = fbm(w, h, 8, base=40, seed=seed + 200)
    water_tex = to01(water_tex)
    rgb[..., 0] = np.where(water, 0.15 + 0.05 * water_tex, rgb[..., 0])
    rgb[..., 1] = np.where(water, 0.20 + 0.06 * water_tex, rgb[..., 1])
    rgb[..., 2] = np.where(water, 0.30 + 0.08 * water_tex, rgb[..., 2])

    building_heights = np.where(
        ~water & ~parks, coarse_var * 0.5 + fine_var * 0.3, 0.0)

    return (
        np.clip(rgb, 0, 1),
        roads.astype(np.uint8),
        building_heights.astype(np.float32),
    )


def farmland_mask(w: int, h: int, seed: int):
    """Generate farmland pattern with strips and patches"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]

    # Create grid-based patches
    grid = 192
    tiles_x = (w // grid) + 3
    tiles_y = (h // grid) + 3
    jitter = rng.random((tiles_y, tiles_x)).astype(np.float32)
    tx = (xx // grid) + 1
    ty = (yy // grid) + 1
    patch = jitter[ty, tx]

    # Create strip patterns (field boundaries)
    strips_a = (yy // 32) % 2 == 0
    strips_b = (xx // 32) % 2 == 0

    # Combine with noise to create farmland areas
    farmland = (fbm(w, h, oct=3, base=256, seed=seed)
                > 0.52) & (strips_a | strips_b)

    return farmland, patch


def rgb_farm(dem, moisture, water, seed=0):
    """
    Agricultural farmland imagery (UK-style):
    - Uses same field generation as rgb_forest
    - More emphasis on crops and plowed fields
    - NO rock/snow patches in center
    """
    # Farm biome is essentially the same as forest for rural areas
    # Just use the field-based generation
    return rgb_forest(dem, moisture, water, seed=seed)


def rgb_coast(dem, moisture, water, seed=0):
    """Coastal RGB coloring with emphasis on beaches and coastal vegetation"""
    h, w = dem.shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Optimized photorealistic textures
    tex_fine = fbm(w, h, oct=6, base=8, seed=seed + 90)
    tex_medium = fbm(w, h, oct=4, base=32, seed=seed + 91)

    # Water - varied coastal waters with wave patterns
    water_var = fbm(w, h, oct=6, base=24, seed=seed + 92) * 0.12
    rgb[..., 0] = np.where(water, 0.08 + water_var * 0.3, 0)
    rgb[..., 1] = np.where(water, 0.50 + water_var * 0.5, 0)
    rgb[..., 2] = np.where(water, 0.85 + water_var * 0.7, 0)

    # Beach - natural sand texture
    beach = (dem >= 0.08) & (dem < 0.18)
    sand_tex = tex_fine * 0.10
    rgb[..., 0] = np.where(beach, 0.88 + sand_tex, rgb[..., 0])
    rgb[..., 1] = np.where(beach, 0.82 + sand_tex * 0.8, rgb[..., 1])
    rgb[..., 2] = np.where(beach, 0.65 + sand_tex * 0.6, rgb[..., 2])

    # Coastal vegetation - photorealistic with texture variation
    veg = np.clip(
        0.70 * moisture + 0.30 *
        (1.0 - np.clip((dem - 0.30) / 0.60, 0, 1)), 0, 1
    )
    veg_detail = tex_fine * 0.15 + tex_medium * 0.12
    rgb[..., 0] = np.where(
        ~water & ~beach, 0.16 + 0.28 * veg + veg_detail * 0.8, rgb[..., 0]
    )
    rgb[..., 1] = np.where(~water & ~beach, 0.30 +
                           0.55 * veg + veg_detail, rgb[..., 1])
    rgb[..., 2] = np.where(
        ~water & ~beach, 0.18 + 0.25 * veg + veg_detail * 0.7, rgb[..., 2]
    )

    # Rock/cliffs with natural texture
    rock = (dem > 0.70) & (dem <= 0.85)
    rock_tex = tex_medium * 0.10
    rgb[..., 0] = np.where(rock, 0.58 + rock_tex, rgb[..., 0])
    rgb[..., 1] = np.where(rock, 0.56 + rock_tex, rgb[..., 1])
    rgb[..., 2] = np.where(rock, 0.54 + rock_tex, rgb[..., 2])

    # Snow at high elevations
    snow = dem > 0.85
    snow_tex = tex_fine * 0.04
    rgb[..., 0] = np.where(snow, 0.94 + snow_tex, rgb[..., 0])
    rgb[..., 1] = np.where(snow, 0.95 + snow_tex, rgb[..., 1])
    rgb[..., 2] = np.where(snow, 0.97 + snow_tex, rgb[..., 2])

    return np.clip(rgb, 0, 1)


def rgb_island(dem, moisture, water, seed=0):
    """Island RGB coloring with tropical/subtropical vegetation"""
    h, w = dem.shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Optimized photorealistic textures
    tex_fine = fbm(w, h, oct=6, base=8, seed=seed + 100)
    tex_medium = fbm(w, h, oct=4, base=32, seed=seed + 101)

    # Water - tropical blue with depth/wave variation
    water_var = fbm(w, h, oct=6, base=20, seed=seed + 102) * 0.14
    rgb[..., 0] = np.where(water, 0.10 + water_var * 0.3, 0)
    rgb[..., 1] = np.where(water, 0.48 + water_var * 0.5, 0)
    rgb[..., 2] = np.where(water, 0.82 + water_var * 0.7, 0)

    # Beach - white/tan sand with texture
    beach = (dem >= 0.10) & (dem < 0.16)
    sand_tex = tex_fine * 0.08
    rgb[..., 0] = np.where(beach, 0.90 + sand_tex, rgb[..., 0])
    rgb[..., 1] = np.where(beach, 0.85 + sand_tex, rgb[..., 1])
    rgb[..., 2] = np.where(beach, 0.70 + sand_tex, rgb[..., 2])

    # Tropical vegetation - very green with realistic variation
    veg = np.clip(
        0.75 * moisture + 0.25 *
        (1.0 - np.clip((dem - 0.35) / 0.55, 0, 1)), 0, 1
    )
    jungle_tex = tex_fine * 0.18 + tex_medium * 0.15  # Dense jungle texture
    rgb[..., 0] = np.where(
        ~water & ~beach, 0.12 + 0.32 * veg + jungle_tex * 0.6, rgb[..., 0]
    )
    rgb[..., 1] = np.where(~water & ~beach, 0.28 +
                           0.60 * veg + jungle_tex, rgb[..., 1])
    rgb[..., 2] = np.where(
        ~water & ~beach, 0.10 + 0.22 * veg + jungle_tex * 0.5, rgb[..., 2]
    )

    # Rock with texture
    rock = (dem > 0.72) & (dem <= 0.88)
    rock_tex = tex_medium * 0.10
    rgb[..., 0] = np.where(rock, 0.60 + rock_tex, rgb[..., 0])
    rgb[..., 1] = np.where(rock, 0.58 + rock_tex, rgb[..., 1])
    rgb[..., 2] = np.where(rock, 0.56 + rock_tex, rgb[..., 2])

    # Snow (rare)
    snow = dem > 0.88
    snow_tex = tex_fine * 0.04
    rgb[..., 0] = np.where(snow, 0.94 + snow_tex, rgb[..., 0])
    rgb[..., 1] = np.where(snow, 0.95 + snow_tex, rgb[..., 1])
    rgb[..., 2] = np.where(snow, 0.97 + snow_tex, rgb[..., 2])

    return np.clip(rgb, 0, 1)


def rgb_water(w: int, h: int, seed: int) -> np.ndarray:
    """Generate realistic water imagery for areas that are actually water bodies"""
    rgb = np.zeros((h, w, 3), np.float32)

    # Base water color (dark blue-green)
    base_r, base_g, base_b = 0.08, 0.42, 0.75

    # Add subtle depth variation using noise
    depth_var = fbm(w, h, oct=4, base=128, seed=seed) * 0.15

    # Add wave patterns (very subtle)
    yy, xx = np.mgrid[0:h, 0:w]
    waves = 0.03 * (np.sin(xx / 20 + seed) * 0.5 + 0.5)

    # Combine
    rgb[..., 0] = np.clip(base_r + depth_var - waves * 0.5, 0.05, 0.15)
    rgb[..., 1] = np.clip(base_g + depth_var * 0.8 - waves * 0.3, 0.35, 0.55)
    rgb[..., 2] = np.clip(base_b + depth_var * 0.5 + waves * 0.2, 0.65, 0.88)

    return np.clip(rgb, 0, 1)


# contours (cheap raster contours)
def raster_contours(dem: np.ndarray, interval_m: float, px_m: float) -> np.ndarray:
    # treat dem 0..1 as 0..1000 "m" relief for visual spacing, scale interval to that space
    scale = 1000.0
    step = max(1.0, interval_m / px_m)  # keep lines reasonably spaced visually
    z = to01(dem) * scale
    mod = np.mod(z, step)
    lines = (mod < 1.2) | (mod > step - 1.2)
    return lines.astype(np.uint8)


# ------------------ RGB "imagery" builder ------------------


def build_rgb(
    biome: str,
    dem: np.ndarray,
    rivers: np.ndarray,
    seed: int,
    water_mask: Optional[np.ndarray] = None,
    pixel_size_m: float = 1.0,
) -> np.ndarray:
    """
    Build RGB imagery for a given biome.

    Args:
        biome: Biome type
        dem: Digital Elevation Model (normalized [0-1] for color generation)
        rivers: River mask (for moisture computation)
        seed: Random seed
        water_mask: All water bodies (rivers + ocean + lakes) for land/water distinction
        pixel_size_m: Physical pixel size in meters (for meter-consistent textures)

    Returns:
        RGB image (H, W, 3), float32 [0-1]
    """
    h, w = dem.shape

    # Ensure DEM is normalized for color generation
    if dem.max() > 2.0:  # DEM is in meters
        dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

    # Special case: pure water
    if biome == "water":
        return rgb_water(w, h, seed)

    # Use water_mask if provided, otherwise fall back to rivers
    if water_mask is None:
        water_mask = rivers

    # Compute moisture from RIVERS only (not all water)
    moist = blur(rivers.astype(np.float32), k=6)
    gx = np.gradient(dem, axis=1)
    wind = to01(np.clip(-gx, 0, None))
    moisture = np.clip(0.6 * moist + 0.4 * wind, 0, 1)

    # Pass water_mask (all water) to biome functions for land/water distinction
    if biome == "mountain":
        rgb = rgb_mountain(dem, moisture, water_mask, seed=seed)
    elif biome == "desert":
        rgb = rgb_desert(dem, moisture, water_mask, seed=seed)
    elif biome == "city":
        rgb, _, _ = rgb_city(
            dem, moisture, water_mask, seed=seed, pixel_size_m=pixel_size_m
        )
    elif biome == "farm":
        rgb = rgb_farm(dem, moisture, water_mask, seed=seed)
    elif biome == "coast":
        rgb = rgb_coast(dem, moisture, water_mask, seed=seed)
    elif biome == "island":
        rgb = rgb_island(dem, moisture, water_mask, seed=seed)
    else:
        rgb = rgb_forest(
            dem, moisture, water_mask, seed=seed, pixel_size_m=pixel_size_m
        )

    # LIGHTING REMOVED - Applied outside with physically-correct hillshade
    # This avoids double-lighting and ensures consistent lighting across resolutions

    # Minimal atmospheric haze (only for distant mountains)
    if biome not in ["forest", "farm"]:
        haze = np.clip((dem - 0.70) * 0.4, 0, 0.06)
        rgb[..., 0] = np.clip(rgb[..., 0] + haze * 0.12, 0, 1)
        rgb[..., 1] = np.clip(rgb[..., 1] + haze * 0.15, 0, 1)
        rgb[..., 2] = np.clip(rgb[..., 2] + haze * 0.25, 0, 1)

    return rgb


def resample(
    path_in: str,
    path_out: str,
    scale_factor: float,
    bounds: Tuple[float, float, float, float],
):
    """
    Resample GeoTIFF by scale factor while preserving CRS and spatial accuracy.

    Args:
        path_in: Input GeoTIFF path
        path_out: Output GeoTIFF path
        scale_factor: Resampling factor. scale_factor > 1 reduces size (coarser resolution).
                      E.g., scale_factor=4 means output is 1/4 the size of input.
        bounds: (minLon, minLat, maxLon, maxLat) for reference (not used if CRS preserved)

    Note: Preserves source CRS (EPSG:3857 for terrain, EPSG:4326 for RGB).
          Transform is scaled correctly for both meter-based and degree-based CRS.
    """
    if not RIO:
        raise RuntimeError(
            "❌ rasterio not available - cannot resample images! Install: pip install rasterio"
        )

    assert scale_factor > 0, "scale_factor must be positive"

    with rasterio.open(path_in) as src:
        new_w = max(1, int(round(src.width / scale_factor)))
        new_h = max(1, int(round(src.height / scale_factor)))
        data = src.read(
            out_shape=(src.count, new_h, new_w), resampling=Resampling.average
        )

        # Preserve source profile (including CRS)
        prof = src.profile

        # Scale transform correctly (works for both EPSG:3857 and EPSG:4326)
        transform = src.transform * src.transform.scale(
            src.width / new_w, src.height / new_h
        )

        prof.update(width=new_w, height=new_h, transform=transform)

        # Ensure RGB photometric interpretation for 3-band images
        if src.count == 3:
            # Force RGB even if present but different
            prof["photometric"] = "RGB"

        with rasterio.open(path_out, "w", **prof) as dst:
            dst.write(data)
            # Explicitly set color interpretation for RGB data
            if src.count == 3:
                dst.colorinterp = [ColorInterp.red,
                                   ColorInterp.green, ColorInterp.blue]
            try:
                dst.build_overviews([2, 4, 8, 16], Resampling.average)
            except Exception:
                pass


# ------------------ Photorealistic Enhancement ------------------


def degrade_resolution(
    rgb_hi: np.ndarray, gsd_hi: float, gsd_lo: float, seed: int
) -> np.ndarray:
    """
    Degrade high-resolution imagery to lower resolution via sensor integration.

    Simulates the physical process of a coarser sensor integrating over larger pixels.

    Args:
        rgb_hi: High-resolution RGB image (H, W, 3), float32 [0-1]
        gsd_hi: GSD of input image (meters)
        gsd_lo: Target GSD (meters)
        seed: Random seed

    Returns:
        Degraded RGB image at lower resolution, float32 [0-1]
    """
    scale = float(gsd_lo / gsd_hi)
    if scale <= 1.0:
        return rgb_hi

    h, w = rgb_hi.shape[:2]
    h_lo = max(1, int(round(h / scale)))
    w_lo = max(1, int(round(w / scale)))

    sigma = min(10.0, 0.45 * scale)

    if OPENCV_AVAILABLE:
        rgb_blurred = cv2.GaussianBlur(
            rgb_hi, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
        )
        rgb_lo = cv2.resize(rgb_blurred, (w_lo, h_lo),
                            interpolation=cv2.INTER_AREA)
    else:
        # Fallback: PIL LANCZOS (not as accurate as AREA + blur, but stable)
        from PIL import Image as PILImage

        u8 = (np.clip(rgb_hi, 0, 1) * 255 + 0.5).astype(np.uint8)
        pil = PILImage.fromarray(u8, "RGB")
        pil_lo = pil.resize((w_lo, h_lo), PILImage.Resampling.LANCZOS)
        rgb_lo = np.asarray(pil_lo).astype(np.float32) / 255.0

    if gsd_lo >= 10.0:
        gray = np.mean(rgb_lo, axis=2, keepdims=True)
        rgb_lo = rgb_lo * 0.88 + gray * 0.12
        rgb_lo = np.clip(rgb_lo**1.04, 0, 1)

    return rgb_lo


def create_eo_config(gsd: float, seed: int, image_size: tuple = None) -> "EOConfig":
    """
    Create EO pipeline configuration optimized for a given GSD and image size.

    Automatically adjusts supersample_factor to prevent memory issues with large images.

    Args:
        gsd: Ground Sample Distance in meters (0.25, 1.0, or 10.0)
        seed: Random seed for reproducibility
        image_size: (height, width) tuple, used to adjust supersample_factor

    Returns:
        EOConfig instance with resolution-appropriate parameters
    """
    # Determine supersample factor based on image size to prevent memory issues
    supersample_factor = 4  # Default

    if image_size is not None:
        h, w = image_size
        total_pixels = h * w

        # Adjust supersample factor based on image size (memory safety)
        # Very conservative thresholds to prevent OOM kills (especially on macOS)
        if total_pixels >= 1_000_000:  # >1MP (e.g., 1000×1000)
            supersample_factor = 1
            print(
                f"   ℹ️  Large image ({w}×{h} = {total_pixels/1e6:.2f}MP), using 1× (no supersampling)"
            )
        elif total_pixels >= 300_000:  # >0.3MP (e.g., 548×548)
            supersample_factor = 2
            print(
                f"   ℹ️  Medium image ({w}×{h} = {total_pixels/1e6:.2f}MP), using 2× supersampling"
            )
        else:
            supersample_factor = 4
            print(
                f"   ℹ️  Small image ({w}×{h} = {total_pixels/1e6:.2f}MP), using 4× supersampling"
            )

    if gsd <= 2.0:
        if image_size is not None and (h * w) < 600_000:
            supersample_factor = max(2, supersample_factor)

    if gsd >= 10.0:
        supersample_factor = 1

    if gsd <= 0.5:  # Very high resolution (25cm)
        return EOConfig(
            supersample_factor=supersample_factor,
            fbm_octaves=7,  # More octaves for fine detail
            fbm_persistence=0.5,
            fbm_lacunarity=2.0,
            fbm_base_scale=80.0,  # Moderate scale for 25cm
            sun_azimuth=315.0,
            sun_altitude=45.0,
            hillshade_strength=0.4,
            edge_blur_sigma=1.5,  # Less blur to preserve detail
            optics_blur_sigma=0.4,  # Sharp optics
            desaturation_factor=0.95,  # Less desaturation
            tone_curve_gamma=1.1,
            green_yellow_bias=1.05,
            warp_strength_px=0.0,  # Disable warp for 25cm (too fine)
            final_smooth_mix=0.05,  # Minimal smoothing
            sharpen_strength=0.0,  # No sharpening
            seed=seed,
        )
    elif gsd <= 2.0:  # High resolution (1m)
        return EOConfig(
            supersample_factor=supersample_factor,
            fbm_octaves=6,  # Standard octaves
            fbm_persistence=0.5,
            fbm_lacunarity=2.0,
            # CRITICAL FIX: Larger scale to create 8-20px blobs after downsampling
            fbm_base_scale=160.0,
            sun_azimuth=315.0,
            sun_altitude=45.0,
            hillshade_strength=0.4,
            edge_blur_sigma=2.0,  # Standard blur
            optics_blur_sigma=1.0,  # CRITICAL FIX: Stronger blur for realistic 1m PSF
            desaturation_factor=0.9,  # Standard desaturation
            tone_curve_gamma=1.05,  # Reduced gamma
            green_yellow_bias=1.05,
            # CRITICAL FIX: Disable warp unless supersampled (handled in pipeline)
            warp_strength_px=0.0,
            final_smooth_mix=0.05,  # CRITICAL FIX: Reduced bilateral smoothing
            sharpen_strength=0.0,  # CRITICAL FIX: Disable sharpening for 1m
            seed=seed,
        )
    else:  # Medium resolution (10m)
        return EOConfig(
            supersample_factor=supersample_factor,
            fbm_octaves=5,  # Fewer octaves for coarser detail
            fbm_persistence=0.5,
            fbm_lacunarity=2.0,
            fbm_base_scale=64.0,  # Default scale for 10m
            sun_azimuth=315.0,
            sun_altitude=45.0,
            hillshade_strength=0.5,  # More terrain shading
            edge_blur_sigma=2.5,  # More blur
            optics_blur_sigma=0.8,  # Softer optics
            desaturation_factor=0.85,  # More desaturation
            tone_curve_gamma=1.15,  # More contrast
            green_yellow_bias=1.08,  # More vegetation bias
            warp_strength_px=0.6,  # Standard warp for 10m
            final_smooth_mix=0.12,  # Standard smoothing
            sharpen_strength=0.15,  # Standard sharpening
            seed=seed,
        )


def apply_photorealism(
    rgb: np.ndarray,
    gsd: float,
    seed: int = 42,
    dem: np.ndarray = None,
    hillshade: np.ndarray = None,
    water_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Apply photorealistic effects to synthetic imagery using EO forward pipeline.

    The EO forward pipeline removes pixelation and creates photorealistic satellite
    imagery by simulating the complete sensor acquisition chain:
    1. Super-sampling (4× internal resolution)
    2. Intra-biome continuous texture (FBM noise)
    3. Terrain-aware shading (hillshade from DEM)
    4. Edge realism (soft boundaries)
    5. EO optics simulation (sensor MTF)
    6. Downsample to target (INTER_AREA)
    7. EO colour science / ISP

    Args:
        rgb: Input RGB imagery (H, W, 3), float32 [0-1]
        gsd: Ground Sample Distance in meters (0.25, 1.0, or 10.0)
        seed: Random seed for reproducibility
        dem: Digital Elevation Model (H, W), float32, optional for terrain shading
        hillshade: Hillshade array (H, W), float32 [0-1] (used by OpenCV fallback only)
        water_mask: Water mask (H, W), boolean or float [0-1] (not used by EO pipeline)

    Returns:
        Photorealistic RGB imagery (H, W, 3), float32 [0-1]
    """
    # Try EO forward pipeline first (best quality)
    if EO_PIPELINE_AVAILABLE:
        try:
            print(f"   🛰️  Applying EO forward pipeline at {gsd}m GSD...")

            # Create resolution-appropriate configuration with size-aware super-sampling
            config = create_eo_config(gsd, seed, image_size=rgb.shape[:2])

            # Apply EO forward pipeline
            # Note: DEM can be passed for terrain shading, but it's optional
            # Debug safety: inputs must be normalized
            assert (
                rgb.min() >= 0.0 and rgb.max() <= 1.0
            ), f"RGB out of range: min={rgb.min()}, max={rgb.max()}"
            if dem is not None:
                assert dem.min() >= 0.0 and dem.max() <= 1.0, "DEM must be normalized"

            photorealistic = eo_forward_pipeline(
                rgb_base=np.clip(rgb, 0.0, 1.0),
                dem=dem.astype(np.float32) if dem is not None else None,
                config=config,
                water_mask=(
                    water_mask.astype(
                        np.float32) if water_mask is not None else None
                ),
            )

            print(f"   ✅ EO forward pipeline applied")
            return photorealistic

        except Exception as e:
            print(
                f"   ⚠️  EO pipeline failed: {e}, falling back to OpenCV photorealism")

    # Fallback to OpenCV photorealism if EO pipeline not available
    if OPENCV_PHOTOREALISM_AVAILABLE and hillshade is not None:
        try:
            print(
                f"   🛰️  Applying OpenCV photorealistic effects at {gsd}m GSD...")

            # Configure sensor for this resolution
            config = SensorConfig(
                gsd=gsd,
                sun_azimuth=135.0,  # Sun from southeast
                sun_elevation=45.0,  # Mid-day sun
                seed=seed,
            )

            # Create vegetation mask from water mask
            vegetation_mask = None
            if water_mask is not None:
                if water_mask.dtype == bool:
                    vegetation_mask = (~water_mask).astype(np.float32)
                else:
                    vegetation_mask = 1.0 - water_mask.astype(np.float32)

            if vegetation_mask is None:
                vegetation_mask = np.ones(
                    (hillshade.shape[0], hillshade.shape[1]), dtype=np.float32
                )

            photorealistic = synthesize_photorealistic_eo(
                reflectance=rgb.copy(),
                hillshade=hillshade.copy(),
                vegetation_mask=vegetation_mask,
                config=config,
                output_uint8=False,
            )

            print(f"   ✅ OpenCV photorealistic effects applied")
            return photorealistic

        except Exception as e:
            print(
                f"   ⚠️  OpenCV photorealism failed: {e}, using base imagery")

    # Final fallback: return base imagery
    print(f"   ⚠️  No photorealism available, using base imagery")
    return rgb


# ------------------ Main function ------------------


def compute_world_dimensions_mercator(
    bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    DEPRECATED: Use compute_world_dimensions_projected() instead.
    Kept for backwards compatibility.
    """
    return compute_world_dimensions_projected(bounds, "EPSG:3857")


def compute_dimensions_from_gsd(
    bounds: Tuple[float, float, float, float], gsd: float, crs_metric: str = "EPSG:3857"
) -> Tuple[int, int, float]:
    """
    Compute pixel dimensions from world bounds and GSD using a metric CRS.

    CRITICAL: This ensures each GSD produces a different pixel grid with consistent meter-based spacing.

    Args:
        bounds: (minLon, minLat, maxLon, maxLat) in degrees
        gsd: Ground Sample Distance in meters
        crs_metric: Metric CRS for true ground meters
                   - "EPSG:3857": Web Mercator (projected meters, scale varies with latitude)
                   - "UTM": Auto-select UTM zone (TRUE ground meters, recommended)
                   - "EPSG:326XX": Specific UTM zone

    Returns:
        (width, height, pixel_size_m) in pixels and meters
    """
    # Always resolve CRS once (handles UTM + zone-crossing safely)
    crs_metric = resolve_metric_crs(bounds, crs_metric)

    # Compute world dimensions in target CRS
    world_width_m, world_height_m = compute_world_dimensions_projected(
        bounds, crs_metric
    )

    # Compute pixel dimensions from GSD
    width = int(math.ceil(world_width_m / gsd))
    height = int(math.ceil(world_height_m / gsd))

    # CRITICAL: Hard cap to prevent memory explosion (especially at 25cm)
    MAX_DIM = 12000  # Maximum dimension on any side
    MAX_PIXELS = 12000 * 12000  # ~144 MP maximum

    total_pixels = width * height
    if width > MAX_DIM or height > MAX_DIM or total_pixels > MAX_PIXELS:
        area_km2 = (world_width_m / 1000.0) * (world_height_m / 1000.0)
        raise ValueError(
            f"❌ Requested grid too large for {gsd}m GSD:\n"
            f"   Dimensions: {width} × {height} pixels ({total_pixels / 1e6:.1f} MP)\n"
            f"   Area: {world_width_m / 1000:.2f} × {world_height_m / 1000:.2f} km ({area_km2:.2f} km²)\n"
            f"   Maximum: {MAX_DIM} × {MAX_DIM} pixels ({MAX_PIXELS / 1e6:.0f} MP)\n"
            f"   💡 Solution: Reduce bounds OR use coarser resolution (1m or 10m)"
        )

    return width, height, gsd


# ============================================================================
# DOWNSAMPLING HELPERS (for base-first, derive-down pipeline)
# ============================================================================


def resize_area_float(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize float array to exact dimensions using area averaging."""
    if OPENCV_AVAILABLE:
        return cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    from PIL import Image as PILImage

    img = PILImage.fromarray(a.astype(np.float32), mode="F")
    img = img.resize((w, h), PILImage.Resampling.BOX)
    return np.asarray(img).astype(np.float32)


def resize_area_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize RGB array to exact dimensions using area averaging."""
    if OPENCV_AVAILABLE:
        return cv2.resize(rgb.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    from PIL import Image as PILImage

    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    img = PILImage.fromarray(u8, "RGB").resize((w, h), PILImage.Resampling.BOX)
    return np.asarray(img).astype(np.float32) / 255.0


def resize_any_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize boolean mask to exact dimensions using nearest neighbor."""
    m = (m > 0.5).astype(np.uint8) * 255
    if OPENCV_AVAILABLE:
        out = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return out > 0
    from PIL import Image as PILImage

    img = PILImage.fromarray(m, "L").resize(
        (w, h), PILImage.Resampling.NEAREST)
    return np.asarray(img) > 127


def downsample_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """
    Area-average downsample by integer factor (physical area integration).

    Args:
        a: Input array (H, W) or (H, W, C)
        factor: Downsample factor (e.g., 4 for 0.25m → 1m)

    Returns:
        Downsampled array
    """
    if factor <= 1:
        return a

    h, w = a.shape[:2]
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    a = a[:h2, :w2]

    if a.ndim == 2:
        return (
            a.reshape(h2 // factor, factor, w2 // factor, factor)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )
    else:
        c = a.shape[2]
        return (
            a.reshape(h2 // factor, factor, w2 // factor, factor, c)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )


def downsample_max(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample boolean/0-1 masks by any-presence (preserve features).

    Args:
        mask: Input mask (H, W) with values 0/1 or boolean
        factor: Downsample factor

    Returns:
        Downsampled mask (boolean)
    """
    if factor <= 1:
        return mask > 0.5

    m = mask > 0.5
    h, w = m.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    m = m[:h2, :w2]
    return m.reshape(h2 // factor, factor, w2 // factor, factor).any(axis=(1, 3))


def pick_base_gsd(resolutions: list) -> float:
    """
    Pick the finest (smallest) GSD from requested resolutions as the base.

    Args:
        resolutions: List of resolution labels (e.g., ['25cm', '1m', '10m'])

    Returns:
        Base GSD in meters (e.g., 0.25 for 25cm)
    """
    gsd_map = {"25cm": 0.25, "1m": 1.0, "10m": 10.0}
    return min(gsd_map[r] for r in resolutions if r in gsd_map)


def generate_single_resolution(
    biome: str,
    gsd: float,
    seed: int,
    outdir: str,
    bounds: Tuple[float, float, float, float],
    profile: str,
    enable_photorealism: bool,
    crs_metric: str = "UTM",
    rgb_wgs84: bool = False,
    ogf_tiles: Optional[np.ndarray] = None,
) -> None:
    """
    DEPRECATED: Use generate_from_base_world() instead.
    Generate imagery at a SINGLE resolution with correct geospatial grid and TRUE ground meters.

    CRITICAL: This function generates imagery on a pixel grid determined by GSD,
    NOT by arbitrary width/height. Each GSD produces a genuinely different sampling.

    Args:
        biome: Biome type
        gsd: Ground Sample Distance in meters (0.25, 1.0, or 10.0)
        seed: Random seed
        outdir: Output directory
        bounds: (minLon, minLat, maxLon, maxLat)
        profile: Quality profile
        enable_photorealism: Whether to apply EO pipeline
        ogf_tiles: Optional OGF tiles for water detection (will be resampled)
    """
    minLon, minLat, maxLon, maxLat = bounds

    # STEP 1: Compute correct pixel dimensions from GSD using metric CRS
    width, height, pixel_size_m = compute_dimensions_from_gsd(
        bounds, gsd, crs_metric)

    # Resolve CRS if auto-select (with UTM zone crossing guard)
    crs_resolved = resolve_metric_crs(bounds, crs_metric)

    print(f"\n{'='*60}", flush=True)
    print(f"🎯 Generating {gsd}m GSD imagery", flush=True)
    print(f"   CRS: {crs_resolved} (TRUE ground meters)", flush=True)
    print(f"   Dimensions: {width} × {height} pixels", flush=True)
    print(
        f"   Pixel size: {pixel_size_m:.2f}m × {pixel_size_m:.2f}m", flush=True)
    print(
        f"   Bounds: ({minLon:.6f}, {minLat:.6f}) → ({maxLon:.6f}, {maxLat:.6f})",
        flush=True,
    )
    print(f"{'='*60}\n", flush=True)

    # Validate dimensions
    total_pixels = width * height
    max_pixels = 32768 * 32768
    if total_pixels > max_pixels:
        raise ValueError(
            f"Image too large at {gsd}m GSD: {width}×{height} = {total_pixels:,} pixels. "
            f"Maximum: {max_pixels:,} pixels. Reduce bounding box size."
        )

    if total_pixels > 16384 * 16384:
        print(
            f"⚠️  Large image at {gsd}m GSD: {width}×{height} = {total_pixels/1e6:.1f}MP",
            flush=True,
        )

    # STEP 2: Generate terrain at THIS resolution
    import time

    t0 = time.time()
    print(f"⛰️  Generating terrain at {gsd}m resolution...", flush=True)

    if biome == "water":
        # Water biome: flat DEM
        base_depth = 5.0
        depth_noise = fbm(width, height, oct=4, base=256, seed=seed) * 20.0
        dem = np.clip(base_depth + depth_noise, 0.0, 50.0).astype(np.float32)
        river_mask = np.ones((height, width), dtype=bool)
        water_mask = river_mask.copy()
        flood_mask = np.zeros((height, width), dtype=bool)
        hs = np.full((height, width), 0.5, dtype=np.float32)
        sl = np.zeros((height, width), dtype=np.float32)
        cnt = np.zeros((height, width), dtype=np.uint8)
    else:
        # Generate DEM at correct resolution
        dem = make_dem(biome, width, height, seed,
                       profile, pixel_size_m=pixel_size_m)
        print(
            f"   DEM complete (elevation: {dem.min():.1f}-{dem.max():.1f}m)", flush=True
        )

        # Terrain analysis with conditional D8 flow (PERFORMANCE CRITICAL)
        # CRITICAL: dem_norm is ONLY for RGB generation. Use dem (meters) for hillshade/slope/contours/GeoTIFF
        dem_norm = normalize_dem(dem)

        # D8 flow is O(H*W) in pure Python - use fast approximation for large grids
        # 500K pixels (~0.5MP) - D8 flow is VERY slow in Python
        D8_THRESHOLD = 500_000
        if total_pixels > D8_THRESHOLD:
            print(
                f"   ⚠️  Grid too large for D8 flow ({total_pixels/1e6:.1f}MP). Using fast approximation...",
                flush=True,
            )
            river_mask, flood_mask = rivers_and_flood_fast(dem_norm, seed=seed)
        else:
            river_mask, flood_mask = rivers_and_flood(dem_norm, density=0.004)

        water_mask = river_mask.copy()

        # Ocean for island/coast biomes
        if biome == "island":
            ocean_mask = dem_norm < 0.20
            water_mask = water_mask | ocean_mask
        elif biome == "coast":
            ocean_mask = dem_norm < 0.12
            water_mask = water_mask | ocean_mask

        # PHYSICALLY CORRECT terrain analysis (scaled by pixel_size_m)
        # Use higher sun angle for rural/forest biomes (softer shadows)
        sun_alt = 60.0 if biome in ["forest", "farm", "coast"] else 50.0
        hs = hillshade_from_dem_m(dem, pixel_size_m, az=315, alt=sun_alt)
        sl = slope_from_dem_m(dem, pixel_size_m)
        cnt = raster_contours_m(dem, interval_m=20.0)

    print(f"   ⏱️  Terrain generation: {time.time()-t0:.1f}s", flush=True)

    # STEP 3: Reproject OGF tiles to THIS resolution if available
    ogf_rgb = None
    if ogf_tiles is not None:
        print(f"   Reprojecting OGF tiles to {gsd}m resolution...", flush=True)
        from PIL import Image as PILImage

        ogf_pil = PILImage.fromarray((ogf_tiles * 255).astype(np.uint8))
        ogf_resized = ogf_pil.resize(
            (width, height), PILImage.Resampling.LANCZOS)
        ogf_rgb = np.array(ogf_resized).astype(np.float32) / 255.0

        ogf_water_mask = extract_ogf_water_mask(ogf_rgb)
        water_mask = water_mask | ogf_water_mask
        print(
            f"   ✅ Detected {ogf_water_mask.sum()} water pixels from OGF", flush=True
        )

    # STEP 4: Generate RGB imagery at THIS resolution
    t0 = time.time()
    print(f"🎨 Generating RGB imagery at {gsd}m...", flush=True)
    rgb = build_rgb(
        biome, dem, river_mask, seed, water_mask=water_mask, pixel_size_m=gsd
    )

    # Add fine texture (WORLD-COORDINATE based for cross-resolution consistency)
    # Features have consistent physical size across all GSDs
    tex_fine = fbm(
        width, height, oct=7, base=grid_px(6.0, pixel_size_m), seed=seed + 200
    )  # 6m grain
    tex_medium = fbm(
        width, height, oct=5, base=grid_px(18.0, pixel_size_m), seed=seed + 201
    )  # 18m grain
    texture = (0.60 * tex_fine + 0.40 * tex_medium) * 0.08
    rgb[..., 0] += texture * 0.90 + tex_fine * 0.012
    rgb[..., 1] += texture * 1.00 + tex_medium * 0.010
    rgb[..., 2] += texture * 0.85 - tex_medium * 0.005
    rgb = np.clip(rgb, 0, 1)

    # Apply physically-correct hillshade lighting (using pre-computed hs from terrain analysis)
    # This ensures consistent lighting across all resolutions
    if biome in ["forest", "farm", "coast"]:
        # Rural scenes: soft hillshade (EXACT range from reference: 0.82 to 1.10)
        HILLSHADE_MIN = 0.82
        HILLSHADE_MAX = 1.10
        hs_mapped = HILLSHADE_MIN + (HILLSHADE_MAX - HILLSHADE_MIN) * hs
        rgb = np.clip(rgb * hs_mapped[..., None], 0, 1)
    else:
        # Mountain/desert: stronger terrain relief (0.62 to 1.37 range)
        hs_strength = 0.50
        combined_light = 0.62 + hs_strength * hs
        rgb = np.clip(rgb * combined_light[..., None], 0, 1)

    print(f"   ⏱️  RGB generation: {time.time()-t0:.1f}s", flush=True)

    # STEP 5: Apply EO pipeline at THIS resolution (CONDITIONAL FOR PERFORMANCE)
    t0 = time.time()
    EO_MAX_PIXELS = (
        2_000_000  # Skip EO for images >2MP to avoid timeout (EO is VERY slow)
    )
    if enable_photorealism and EO_PIPELINE_AVAILABLE:
        if total_pixels > EO_MAX_PIXELS:
            print(
                f"   ⚠️  Skipping EO pipeline at {gsd}m ({total_pixels/1e6:.1f}MP too large, max {EO_MAX_PIXELS/1e6:.0f}MP)",
                flush=True,
            )
            print(
                f"   💡 Use smaller bounds or coarser resolution for photorealism",
                flush=True,
            )
        else:
            print(f"🛰️  Applying EO pipeline at {gsd}m GSD...", flush=True)
            config = create_eo_config(gsd, seed, image_size=(height, width))

            # CRITICAL FIX: Always normalize DEM for EO (avoid stale/mismatched dem_norm)
            # EO pipeline expects normalized [0-1] DEM for hillshade calculations
            dem_for_eo = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

            rgb = apply_photorealism(
                rgb,
                gsd=gsd,
                seed=seed,
                dem=dem_for_eo,
                hillshade=hs,
                water_mask=water_mask,
            )
            print(f"   ⏱️  EO pipeline: {time.time()-t0:.1f}s", flush=True)
            print(f"   ✅ EO pipeline complete", flush=True)
    else:
        if not enable_photorealism:
            print(f"   ℹ️  Photorealism disabled by user", flush=True)
        elif not EO_PIPELINE_AVAILABLE:
            print(f"   ℹ️  EO pipeline not available", flush=True)

    # STEP 6: Write outputs at THIS resolution
    gsd_label = f"{int(gsd*100)}cm" if gsd < 1 else f"{int(gsd)}m"
    print(f"💾 Writing {gsd_label} outputs...", flush=True)

    # Write terrain layers (all in same metric CRS for perfect alignment)
    write_tif_with_bounds(
        dem, bounds, os.path.join(outdir, f"dem_{gsd_label}.tif"), crs_out=crs_resolved
    )
    write_tif_with_bounds(
        hs,
        bounds,
        os.path.join(outdir, f"hillshade_{gsd_label}.tif"),
        crs_out=crs_resolved,
    )
    write_tif_with_bounds(
        sl, bounds, os.path.join(outdir, f"slope_{gsd_label}.tif"), crs_out=crs_resolved
    )
    write_tif_with_bounds(
        water_mask.astype(np.uint8),
        bounds,
        os.path.join(outdir, f"water_{gsd_label}.tif"),
        crs_out=crs_resolved,
    )
    write_tif_with_bounds(
        flood_mask.astype(np.uint8),
        bounds,
        os.path.join(outdir, f"floodplain_{gsd_label}.tif"),
        crs_out=crs_resolved,
    )
    write_tif_with_bounds(
        cnt,
        bounds,
        os.path.join(outdir, f"contours_{gsd_label}.tif"),
        crs_out=crs_resolved,
    )

    # Write RGB (same metric CRS for perfect QGIS alignment with terrain layers)
    write_tif_with_bounds(
        rgb,
        bounds,
        os.path.join(outdir, f"rgb_{gsd_label}.tif"),
        bands=3,
        photometric="RGB",
        crs_out=crs_resolved,
    )

    # Optionally write WGS84 version for web maps
    if rgb_wgs84:
        write_tif_with_bounds(
            rgb,
            bounds,
            os.path.join(outdir, f"rgb_{gsd_label}_wgs84.tif"),
            bands=3,
            photometric="RGB",
            crs_out="EPSG:4326",
        )

    # Validation: Check transform (all metric CRS outputs have pixel size in meters)
    if RIO:
        with rasterio.open(os.path.join(outdir, f"rgb_{gsd_label}.tif")) as src:
            transform = src.transform
            pixel_size_x_m = abs(transform.a)
            pixel_size_y_m = abs(transform.e)

            print(
                f"   ✅ Validation: pixel size = {pixel_size_x_m:.3f}m × {pixel_size_y_m:.3f}m (target: {gsd}m)",
                flush=True,
            )
            print(
                f"   CRS: {src.crs}, Dimensions: {src.width}×{src.height}", flush=True
            )

            # Hard validation
            # 5% tolerance (accounts for Earth's curvature and lat/lon distortion)
            tolerance = 0.05
            if (
                abs(pixel_size_x_m - gsd) / gsd > tolerance
                or abs(pixel_size_y_m - gsd) / gsd > tolerance
            ):
                raise ValueError(
                    f"VALIDATION FAILED: Pixel size mismatch!\n"
                    f"Expected: {gsd}m × {gsd}m\n"
                    f"Actual: {pixel_size_x_m:.3f}m × {pixel_size_y_m:.3f}m"
                )

    print(f"✅ {gsd_label} generation complete!\n", flush=True)


def generate_from_base_world(
    biome: str,
    seed: int,
    outdir: str,
    bounds: Tuple[float, float, float, float],
    profile: str,
    enable_photorealism: bool,
    crs_metric: str,
    rgb_wgs84: bool,
    resolutions: list,
    ogf_tiles: Optional[np.ndarray] = None,
    ref_map_path: Optional[str] = None,
) -> None:
    """
    Generate multi-resolution dataset using BASE-FIRST, DERIVE-DOWN approach.

    This ensures:
    1. All resolutions show the same world (not regenerated noise)
    2. Coarser resolutions are physically correct downsamplings of the base
    3. Perfect alignment across all layers and resolutions

    Args:
        biome: Terrain biome type
        seed: Random seed for reproducibility
        outdir: Output directory
        bounds: (minLon, minLat, maxLon, maxLat) in degrees
        profile: Quality profile ('low', 'high', 'ultra')
        enable_photorealism: Whether to apply EO pipeline
        crs_metric: Metric CRS ('UTM' or 'EPSG:3857')
        rgb_wgs84: Whether to also write EPSG:4326 RGB
        resolutions: List of resolution labels (e.g., ['25cm', '1m', '10m'])
        ogf_tiles: Optional OGF water tiles
    """
    minLon, minLat, maxLon, maxLat = bounds

    # Resolution handling (CRITICAL: respect user-selected resolutions exactly)
    resolutions = list(resolutions)
    gsd_map = {"25cm": 0.25, "1m": 1.0, "10m": 10.0}
    gsds = [gsd_map[r] for r in resolutions if r in gsd_map]

    if not gsds:
        raise ValueError("No valid resolutions specified")

    base_gsd = min(gsds)  # finest requested resolution
    multi_res = len(resolutions) > 1

    print(f"🎯 Requested resolutions: {resolutions}", flush=True)
    print(
        f"📏 Base GSD: {base_gsd}m ({int(base_gsd*100) if base_gsd < 1 else int(base_gsd)}{'cm' if base_gsd < 1 else 'm'})",
        flush=True,
    )
    if multi_res:
        print(
            f"📦 Multi-resolution mode: will generate {len(resolutions)} outputs from base",
            flush=True,
        )
    else:
        print(
            f"📦 Single-resolution mode: generating ONLY {resolutions[0]}", flush=True
        )
    print()

    # 1️⃣ Pick base (finest) GSD and generate base world
    base_label = f"{int(base_gsd*100)}cm" if base_gsd < 1 else f"{int(base_gsd)}m"

    print(f"\n{'='*70}", flush=True)
    print(f"🌍 BASE-FIRST GENERATION: {base_label} → derive others", flush=True)
    print(f"{'='*70}\n", flush=True)

    base_w, base_h, base_px = compute_dimensions_from_gsd(
        bounds, base_gsd, crs_metric)

    max_factor = max(int(round(gsd_map[r] / base_gsd)) for r in resolutions)
    base_w = max(max_factor, (base_w // max_factor) * max_factor)
    base_h = max(max_factor, (base_h // max_factor) * max_factor)

    if base_w < 64 or base_h < 64:
        print(
            f"⚠️  Warning: Very small base image ({base_w}×{base_h}) - consider larger bounds",
            flush=True,
        )

    total_pixels = base_w * base_h
    crs_resolved = resolve_metric_crs(bounds, crs_metric)

    print(
        f"📐 Base resolution: {base_label} ({base_w}×{base_h} = {total_pixels:,} pixels)",
        flush=True,
    )
    print(f"   CRS: {crs_resolved} (TRUE ground meters)", flush=True)
    print(f"   Pixel size: {base_px:.3f}m × {base_px:.3f}m\n", flush=True)

    # STEP 2: Compute base transform once (for proper georeferencing inheritance)
    # CRITICAL: Use projected bounds (meters), not WGS84 bounds (degrees)
    from rasterio.transform import from_bounds
    from affine import Affine
    proj_bounds = bounds_to_projected(bounds, crs_resolved)
    minx, miny, maxx, maxy = proj_bounds
    base_transform = from_bounds(
        minx, miny, maxx, maxy,
        base_w, base_h
    )
    # STEP 6: Prevent float precision drift when scaling multiple times
    # Round each coefficient to 12 decimal places
    base_transform = Affine(
        round(base_transform.a, 12),
        round(base_transform.b, 12),
        round(base_transform.c, 12),
        round(base_transform.d, 12),
        round(base_transform.e, 12),
        round(base_transform.f, 12)
    )
    print(f"   Base transform: {base_transform}", flush=True)
    print(
        f"   Projected bounds: ({minx:.1f}, {miny:.1f}) → ({maxx:.1f}, {maxy:.1f}) meters", flush=True)

    # 2️⃣ Generate base DEM (meters)
    print(f"⛰️  Generating base terrain at {base_label}...", flush=True)
    t0 = time.time()
    terrain_dem_m = make_dem(biome, base_w, base_h,
                             seed, profile, pixel_size_m=base_px)
    terrain_dem_norm = normalize_dem(terrain_dem_m)
    surface_height_m = terrain_dem_m.copy()
    print(f"   ⏱️  Base terrain: {time.time()-t0:.1f}s\n", flush=True)

    # 3️⃣ Generate base hydrology (TERRAIN-FIRST approach)
    print(f"💧 Generating base hydrology...", flush=True)
    D8_THRESHOLD = 500_000
    if total_pixels > D8_THRESHOLD:
        print(
            f"   Using fast approximation ({total_pixels/1e6:.1f}MP)", flush=True)
        river_mask, flood_mask = rivers_and_flood_fast(
            terrain_dem_norm, seed=seed)
    else:
        river_mask, flood_mask = rivers_and_flood(
            terrain_dem_norm, density=0.004)

    # PRIMARY TRUTH: Terrain-based water (rivers + elevation-based ocean)
    water_mask = river_mask.copy()

    # Elevation-based ocean for coastal biomes
    if biome == "island":
        ocean_mask = terrain_dem_norm < 0.20
        water_mask = water_mask | ocean_mask
        print(f"   Terrain-based water: {water_mask.sum()} pixels", flush=True)
    elif biome == "coast":
        ocean_mask = terrain_dem_norm < 0.15
        water_mask = water_mask | ocean_mask
        print(f"   Terrain-based water: {water_mask.sum()} pixels", flush=True)
    else:
        print(
            f"   Terrain-based water: {water_mask.sum()} pixels (rivers only)",
            flush=True,
        )

    # SECONDARY: Color-based hints from OGF tiles (refines, does not define)
    if ogf_tiles is not None:
        print(f"   Integrating OGF color hints...", flush=True)
        try:
            from PIL import Image as PILImage

            ogf_pil = PILImage.fromarray((ogf_tiles * 255).astype(np.uint8))
            ogf_resized = ogf_pil.resize(
                (base_w, base_h), PILImage.Resampling.LANCZOS)
            ogf_rgb = np.asarray(ogf_resized).astype(np.float32) / 255.0

            ogf_water = extract_ogf_water_mask(ogf_rgb)
            ogf_water = clean_water_mask(ogf_water, min_area_px=200)
            water_mask = water_mask | ogf_water
            print(
                f"   ✅ OGF color hints: +{ogf_water.sum()} pixels", flush=True)
        except Exception as e:
            print(f"   ⚠️  OGF integration failed: {e}", flush=True)

    # SECONDARY: Reference map color hints (user-provided)
    if ref_map_path is not None:
        print(f"   Integrating reference map color hints...", flush=True)
        try:
            from PIL import Image as PILImage

            ref = PILImage.open(ref_map_path).convert("RGB")
            ref_resized = ref.resize(
                (base_w, base_h), PILImage.Resampling.LANCZOS)
            ref_rgb = np.asarray(ref_resized).astype(np.float32) / 255.0

            ref_water = extract_ogf_water_mask(ref_rgb)
            ref_water = clean_water_mask(ref_water, min_area_px=200)
            water_mask = water_mask | ref_water
            print(
                f"   ✅ Reference color hints: +{ref_water.sum()} pixels", flush=True)
        except Exception as e:
            print(f"   ⚠️  Reference map loading failed: {e}", flush=True)

    print(f"   ✅ Base hydrology complete\n", flush=True)

    # 4️⃣ Generate base terrain analysis (physically correct)
    print(f"🗺️  Computing terrain analysis...", flush=True)
    sun_alt = 60.0 if biome in ["forest", "farm", "coast"] else 50.0
    hs_base = hillshade_from_dem_m(
        surface_height_m, base_px, az=315, alt=sun_alt)
    sl_base = slope_from_dem_m(terrain_dem_m, base_px)
    cnt_base = raster_contours_m(terrain_dem_m, interval_m=20.0)
    print(f"   ✅ Hillshade, slope, contours ready\n", flush=True)

    # 5️⃣ Generate base RGB (pre-EO) + semantic height field
    print(f"🎨 Generating base RGB + semantic scene...", flush=True)
    t0 = time.time()

    if biome == "forest":
        print(f"   🌲 Generating forest with canopy semantics", flush=True)

        rgb_base = build_rgb(
            biome,
            terrain_dem_norm,
            river_mask,
            seed,
            water_mask=water_mask,
            pixel_size_m=base_px,
        )

        tex_fine = fbm(
            base_w, base_h, oct=7, base=grid_px(6.0, base_px), seed=seed + 200
        )
        tex_medium = fbm(
            base_w, base_h, oct=5, base=grid_px(18.0, base_px), seed=seed + 201
        )
        texture = (0.60 * tex_fine + 0.40 * tex_medium) * 0.08
        rgb_base[..., 0] += texture * 0.90 + tex_fine * 0.012
        rgb_base[..., 1] += texture * 1.00 + tex_medium * 0.010
        rgb_base[..., 2] += texture * 0.85 - tex_medium * 0.005
        rgb_base = np.clip(rgb_base, 0, 1)

        _, canopy_height = generate_forest_scene(
            base_w, base_h, seed, origin_xy=(0, 0))
        surface_height_m = (terrain_dem_m + canopy_height).astype(np.float32)
        print(
            f"   ✅ Forest: terrain + canopy (max {surface_height_m.max():.1f}m)",
            flush=True,
        )

        hs_base = hillshade_from_dem_m(
            surface_height_m, base_px, az=315, alt=sun_alt)
        print(f"   ✅ Terrain+canopy hillshade ready", flush=True)

    elif biome == "city":
        print(
            f"   🏙️  Generating city with building semantics (like forest canopy)",
            flush=True,
        )

        # Generate semantic city scene: reflectance + building heights
        rgb_base, building_h_m = generate_city_scene(
            base_w, base_h, seed, origin_xy=(0, 0)
        )

        # Add water bodies to RGB (parks already in semantic scene)
        water_color = np.array([0.15, 0.20, 0.28], dtype=np.float32)
        rgb_base = np.where(water_mask[..., None], water_color, rgb_base)

        # Surface = terrain + building heights (same as forest = terrain + canopy)
        surface_height_m = (terrain_dem_m + building_h_m).astype(np.float32)
        print(
            f"   ✅ City: terrain + buildings (max {surface_height_m.max():.1f}m)",
            flush=True,
        )

        # Recompute hillshade with buildings (roads will naturally darken in shadows)
        hs_base = hillshade_from_dem_m(
            surface_height_m, base_px, az=315, alt=sun_alt)
        print(
            f"   ✅ Terrain+building hillshade ready (roads = height cuts)", flush=True
        )

    else:
        print(f"   ⛰️  Using terrain-based RGB generation", flush=True)
        rgb_base = build_rgb(
            biome,
            terrain_dem_norm,
            river_mask,
            seed,
            water_mask=water_mask,
            pixel_size_m=base_px,
        )

        tex_fine = fbm(
            base_w, base_h, oct=7, base=grid_px(6.0, base_px), seed=seed + 200
        )
        tex_medium = fbm(
            base_w, base_h, oct=5, base=grid_px(18.0, base_px), seed=seed + 201
        )
        texture = (0.60 * tex_fine + 0.40 * tex_medium) * 0.08
        rgb_base[..., 0] += texture * 0.90 + tex_fine * 0.012
        rgb_base[..., 1] += texture * 1.00 + tex_medium * 0.010
        rgb_base[..., 2] += texture * 0.85 - tex_medium * 0.005
        rgb_base = np.clip(rgb_base, 0, 1)

    # Keep as REFLECTANCE (no lighting yet) - we'll apply hillshade per-resolution
    rgb_reflect_base = rgb_base.copy()

    print(f"   ⏱️  Base RGB reflectance: {time.time()-t0:.1f}s\n", flush=True)

    # 6️⃣ Optional EO pipeline at base (only if safe) - works on REFLECTANCE
    EO_MAX_PIXELS = 2_000_000
    EO_OUTPUT_IS_REFLECTANCE = True
    if (
        enable_photorealism
        and EO_PIPELINE_AVAILABLE
        and (total_pixels <= EO_MAX_PIXELS)
    ):
        print(f"🛰️  Applying EO pipeline to base {base_label}...", flush=True)
        t0 = time.time()
        try:
            dem_for_eo = normalize_dem(surface_height_m).astype(np.float32)
            rgb_reflect_base = apply_photorealism(
                rgb_reflect_base,
                gsd=base_gsd,
                seed=seed,
                dem=dem_for_eo,
                hillshade=hs_base,
                water_mask=water_mask,
            )
            print(f"   ⏱️  EO pipeline: {time.time()-t0:.1f}s", flush=True)
            print(f"   ✅ EO complete\n", flush=True)
        except Exception as e:
            print(f"   ⚠️  EO failed: {e}", flush=True)
            print(f"   Using base reflectance without EO\n", flush=True)
    elif total_pixels > EO_MAX_PIXELS:
        print(
            f"   ⚠️  Skipping EO at base ({total_pixels/1e6:.1f}MP > {EO_MAX_PIXELS/1e6:.0f}MP limit)\n",
            flush=True,
        )

    # 7️⃣ For each requested resolution: derive by downsampling
    print(f"{'='*70}", flush=True)
    print(f"📦 DERIVING RESOLUTIONS FROM BASE", flush=True)
    print(f"{'='*70}\n", flush=True)

    for label in resolutions:
        try:
            gsd = gsd_map[label]
            gsd_label = f"{int(gsd*100)}cm" if gsd < 1 else f"{int(gsd)}m"
            factor = int(round(gsd / base_gsd))
            if factor < 1:
                factor = 1

            # Sanity: factor should be integer multiples (0.25->1m->10m are exact)
            if abs((gsd / base_gsd) - factor) > 1e-6:
                raise ValueError(
                    f"Non-integer downsample factor: base={base_gsd}, target={gsd}"
                )

            print(f"🎯 {gsd_label} (factor {factor}x from base):", flush=True)

            # STEP 2: Enforce exact divisibility (prevent silent drift)
            if base_w % factor != 0 or base_h % factor != 0:
                raise ValueError(
                    f"Base dimensions ({base_w}×{base_h}) not divisible by factor {factor}. "
                    f"Adjust bounds or base GSD."
                )

            w_t = base_w // factor
            h_t = base_h // factor
            px_t = base_px * factor

            # STEP 3: Derive transform from base (not recompute from bounds)
            # STEP 1: Scale full Affine matrix (future-proof against rotation/shear)
            from affine import Affine
            scale = factor
            transform_t = Affine(
                base_transform.a * scale,
                base_transform.b * scale,
                base_transform.c,
                base_transform.d * scale,
                base_transform.e * scale,
                base_transform.f
            )
            print(
                f"   Derived transform: pixel_size=({abs(transform_t.a):.6f}m, {abs(transform_t.e):.6f}m)", flush=True)

            if factor == 1:
                terrain_dem_m_t = terrain_dem_m
                surface_height_m_t = surface_height_m
                hs_t = hs_base
                sl_t = sl_base
                cnt_t = cnt_base
                river_t = river_mask
                flood_t = flood_mask
                water_t = water_mask
                rgb_reflect_t = rgb_reflect_base
                print(
                    f"   Using base arrays directly | dims={w_t}×{h_t} px | px={px_t:.3f}m",
                    flush=True,
                )
            else:
                print(f"   Downsampling from {base_label}...", flush=True)
                terrain_dem_m_t = downsample_mean(terrain_dem_m, factor)
                surface_height_m_t = downsample_mean(surface_height_m, factor)
                river_t = downsample_max(river_mask, factor)
                flood_t = downsample_max(flood_mask, factor)
                water_t = downsample_max(water_mask, factor)

                hs_t = hillshade_from_dem_m(
                    surface_height_m_t, px_t, az=315, alt=sun_alt)
                sl_t = slope_from_dem_m(terrain_dem_m_t, px_t)
                cnt_t = raster_contours_m(terrain_dem_m_t, interval_m=20.0)

                # STEP 1: Pure area-based integration (NO degrade_resolution, NO blur, NO noise)
                import cv2
                rgb_reflect_t = cv2.resize(
                    rgb_reflect_base,
                    (w_t, h_t),
                    interpolation=cv2.INTER_AREA
                )
                print(f"   ✓ Area-based downsampling (cv2.INTER_AREA)", flush=True)
                print(
                    f"   Downsampled to {w_t}×{h_t} px | px={px_t:.3f}m", flush=True)

            if EO_OUTPUT_IS_REFLECTANCE:
                rgb_t = apply_hillshade_rgb(rgb_reflect_t, hs_t, biome)
                print(f"   Applied hillshade lighting", flush=True)
            else:
                rgb_t = rgb_reflect_t
                print(f"   EO output is lit, skipping hillshade", flush=True)

            # 8️⃣ Write outputs
            print(f"   💾 Writing {gsd_label} outputs...", flush=True)
            # Write outputs with verification
            dem_path = os.path.join(outdir, f"dem_{gsd_label}.tif")
            dem_vis_path = os.path.join(outdir, f"dem_{gsd_label}_vis.tif")

            # STEP 4: Use derived transform (not recompute from bounds)
            write_tif_with_bounds(
                terrain_dem_m_t, bounds, dem_path, crs_out=crs_resolved, transform_override=transform_t
            )
            write_dem_visual(
                terrain_dem_m_t, bounds, dem_vis_path, crs_out=crs_resolved, transform_override=transform_t
            )

            if RIO:
                with rasterio.open(dem_path) as ds:
                    a = ds.read(1)
                    print(
                        f"   🔎 {os.path.basename(dem_path)}: {ds.dtypes[0]}, range={float(a.min()):.1f}-{float(a.max()):.1f}m",
                        flush=True,
                    )
                with rasterio.open(dem_vis_path) as ds:
                    a = ds.read(1)
                    print(
                        f"   🔎 {os.path.basename(dem_vis_path)}: {ds.dtypes[0]}, range={int(a.min())}-{int(a.max())}",
                        flush=True,
                    )

            write_tif_with_bounds(
                hs_t,
                bounds,
                os.path.join(outdir, f"hillshade_{gsd_label}.tif"),
                crs_out=crs_resolved,
                transform_override=transform_t,
            )
            write_tif_with_bounds(
                sl_t,
                bounds,
                os.path.join(outdir, f"slope_{gsd_label}.tif"),
                crs_out=crs_resolved,
                transform_override=transform_t,
            )
            write_tif_with_bounds(
                water_t.astype(np.uint8),
                bounds,
                os.path.join(outdir, f"water_{gsd_label}.tif"),
                crs_out=crs_resolved,
                transform_override=transform_t,
            )
            write_tif_with_bounds(
                flood_t.astype(np.uint8),
                bounds,
                os.path.join(outdir, f"floodplain_{gsd_label}.tif"),
                crs_out=crs_resolved,
                transform_override=transform_t,
            )
            write_tif_with_bounds(
                cnt_t.astype(np.uint8),
                bounds,
                os.path.join(outdir, f"contours_{gsd_label}.tif"),
                crs_out=crs_resolved,
                transform_override=transform_t,
            )
            write_tif_with_bounds(
                rgb_t,
                bounds,
                os.path.join(outdir, f"rgb_{gsd_label}.tif"),
                bands=3,
                photometric="RGB",
                crs_out=crs_resolved,
                transform_override=transform_t,
            )

            if RIO:
                rgb_path = os.path.join(outdir, f"rgb_{gsd_label}.tif")
                dem_path = os.path.join(outdir, f"dem_{gsd_label}.tif")
                base_rgb_path = os.path.join(outdir, f"rgb_{base_label}.tif")

                with rasterio.open(rgb_path) as ds:
                    assert (
                        ds.width == w_t and ds.height == h_t
                    ), f"RGB dims mismatch: expected {w_t}×{h_t}, got {ds.width}×{ds.height}"
                    print(
                        f"   🔎 {gsd_label} RGB: {ds.width}×{ds.height} | "
                        f"px={abs(ds.transform.a):.3f}m × {abs(ds.transform.e):.3f}m | bounds={ds.bounds}",
                        flush=True,
                    )
                    assert ds.crs == rasterio.crs.CRS.from_string(
                        crs_resolved
                    ), f"RGB CRS mismatch: expected {crs_resolved}, got {ds.crs}"

                    # STEP 3: Validate transform consistency (mathematically exact)
                    expected_px_x = abs(base_transform.a) * factor
                    expected_px_y = abs(base_transform.e) * factor
                    actual_px_x = abs(ds.transform.a)
                    actual_px_y = abs(ds.transform.e)

                    error_x = abs(actual_px_x - expected_px_x)
                    error_y = abs(actual_px_y - expected_px_y)

                    assert error_x < 1e-6, \
                        f"Pixel size X mismatch: expected {expected_px_x:.12f}, got {actual_px_x:.12f} (error={error_x:.12f})"
                    assert error_y < 1e-6, \
                        f"Pixel size Y mismatch: expected {expected_px_y:.12f}, got {actual_px_y:.12f} (error={error_y:.12f})"

                    # STEP 4: Validate bounds match base
                    if factor > 1 and os.path.exists(base_rgb_path):
                        with rasterio.open(base_rgb_path) as base_ds:
                            base_bounds = base_ds.bounds
                        assert ds.bounds == base_bounds, \
                            f"Bounds mismatch for {gsd_label}: derived bounds differ from base"

                with rasterio.open(dem_path) as ds:
                    assert (
                        ds.width == w_t and ds.height == h_t
                    ), f"DEM dims mismatch: expected {w_t}×{h_t}, got {ds.width}×{ds.height}"
                    dem_arr = ds.read(1)
                    print(
                        f"   🔎 {gsd_label} DEM: {ds.width}×{ds.height} | "
                        f"range={float(dem_arr.min()):.1f}-{float(dem_arr.max()):.1f}m",
                        flush=True,
                    )
                    # Assert CRS matches expected
                    assert ds.crs == rasterio.crs.CRS.from_string(
                        crs_resolved
                    ), f"DEM CRS mismatch: expected {crs_resolved}, got {ds.crs}"

                    # STEP 3: Validate transform consistency (mathematically exact)
                    expected_px_x = abs(base_transform.a) * factor
                    expected_px_y = abs(base_transform.e) * factor
                    actual_px_x = abs(ds.transform.a)
                    actual_px_y = abs(ds.transform.e)

                    error_x = abs(actual_px_x - expected_px_x)
                    error_y = abs(actual_px_y - expected_px_y)

                    assert error_x < 1e-6, \
                        f"DEM pixel size X mismatch: expected {expected_px_x:.12f}, got {actual_px_x:.12f} (error={error_x:.12f})"
                    assert error_y < 1e-6, \
                        f"DEM pixel size Y mismatch: expected {expected_px_y:.12f}, got {actual_px_y:.12f} (error={error_y:.12f})"

            if rgb_wgs84:
                write_tif_with_bounds(
                    rgb_t,
                    bounds,
                    os.path.join(outdir, f"rgb_{gsd_label}_wgs84.tif"),
                    bands=3,
                    photometric="RGB",
                    crs_out="EPSG:4326",
                )

            print(f"   ✅ {gsd_label} complete!\n", flush=True)
        except Exception as e:
            print(f"   ❌ Error generating {gsd_label}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise  # Re-raise to stop generation

    print(f"{'='*70}", flush=True)
    print(f"✅ All resolutions generated from base world!", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Generate preview image
    write_preview_image(outdir)


def write_preview_image(outdir: str) -> None:
    """
    Generate a preview PNG from the best available RGB resolution.

    Priority: 1m > 10m > 25cm (fallback).
    Handles both uint8 and float32 RGB imagery correctly.
    """
    import os

    print(f"🖼️  Generating preview in {outdir}", flush=True)

    if not RIO:
        raise RuntimeError(
            "Preview generation requires rasterio - install with: pip install rasterio"
        )

    candidates = ["rgb_1m.tif", "rgb_10m.tif", "rgb_25cm.tif"]

    rgb_path = None
    for c in candidates:
        p = os.path.join(outdir, c)
        if os.path.exists(p):
            rgb_path = p
            break

    if rgb_path is None:
        raise FileNotFoundError(
            f"No RGB TIF found in {outdir} - expected one of: {candidates}"
        )

    try:
        print(f"   Using source RGB: {rgb_path}", flush=True)

        with rasterio.open(rgb_path) as ds:
            # Read RGB bands (may be float32 or uint8)
            rgb = ds.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
            rgb = np.transpose(rgb, (1, 2, 0))  # Convert to (H, W, 3)

            # Handle both float32 (0-1) and uint8 (0-255) imagery
            if rgb.max() <= 1.0:
                print(f"   Detected float32 imagery, scaling to 0-255", flush=True)
                rgb = rgb * 255.0

            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Downscale to reasonable preview size
        from PIL import Image as PILImage

        img = PILImage.fromarray(rgb, "RGB")
        img.thumbnail((512, 512), PILImage.Resampling.LANCZOS)

        preview_path = os.path.join(outdir, "preview_imagery.png")
        img.save(preview_path, format="PNG", optimize=True)

        print(
            f"   ✅ Preview saved: preview_imagery.png ({img.size[0]}×{img.size[1]})",
            flush=True,
        )
    except Exception as e:
        raise RuntimeError(f"Preview generation failed: {e}") from e


def main(
    biome: str,
    width: int,
    height: int,
    seed: int,
    outdir: str,
    bounds: Tuple[float, float, float, float],
    profile: str = "high",
    resolutions: list = None,
    enable_photorealism: bool = True,
    crs_metric: str = "UTM",
    rgb_wgs84: bool = False,
    ref_map: Optional[str] = None,
):
    """
    Generate dataset with geospatially correct multi-resolution imagery in TRUE ground meters.

    CRITICAL: Each resolution is generated independently on its own pixel grid,
    determined by GSD + bounds. This ensures different resolutions are genuinely
    different zoom levels, not just rescaled versions of the same image.
    """
    ensure_outdir(outdir)

    minLon, minLat, maxLon, maxLat = bounds

    # Validate CRS + bounds compatibility
    if crs_metric.upper() == "EPSG:3857" and not mercator_safe(bounds):
        raise ValueError(
            f"❌ EPSG:3857 not valid beyond ±85.0511° latitude.\n"
            f"   Your bounds: {minLat:.2f}° to {maxLat:.2f}°\n"
            f"   Use --crs_metric utm or reduce bounds."
        )

    # CRITICAL FIX: width/height from CLI are now IGNORED for multi-resolution
    # Each GSD computes its own dimensions from bounds
    # This ensures different resolutions have different pixel grids

    if resolutions is None:
        resolutions = ["25cm", "1m", "10m"]
        print("ℹ️  No resolutions specified, defaulting to: 25cm, 1m, 10m\n", flush=True)
    else:
        # User explicitly requested these resolutions - respect them exactly
        print(
            f"✓ User-requested resolutions: {', '.join(resolutions)}\n", flush=True)

    # Validate that we're not using fixed dimensions with multiple resolutions
    if len(resolutions) > 1 and (width > 0 or height > 0):
        print(
            "⚠️  Warning: --width and --height are ignored for multi-resolution generation",
            flush=True,
        )
        print("   Each resolution computes dimensions from GSD + bounds", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"🌍 Generating {biome} dataset", flush=True)
    print(
        f"   Bounds: ({minLon:.6f}, {minLat:.6f}) → ({maxLon:.6f}, {maxLat:.6f})",
        flush=True,
    )
    print(f"   Resolutions: {', '.join(resolutions)}", flush=True)
    print(f"   Seed: {seed}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Convert resolution labels to GSD values
    gsd_map = {"25cm": 0.25, "1m": 1.0, "10m": 10.0}

    # Estimate total pixels for all requested resolutions (for OGF decision)
    total_max_pixels = 0
    for res_label in resolutions:
        if res_label in gsd_map:
            try:
                w, h, _ = compute_dimensions_from_gsd(
                    bounds, gsd_map[res_label], crs_metric=crs_metric
                )
                total_max_pixels = max(total_max_pixels, w * h)
            except ValueError:
                pass  # Skip invalid resolutions

    # Download OGF tiles (CONDITIONAL - ONLY for coarse resolutions >= 10m)
    # OGF is slow (serial HTTP), flaky, and only used for water detection
    # Fine resolutions (25cm, 1m) use procedural generation only
    ogf_tiles = None
    finest_gsd = min(gsd_map[r] for r in resolutions if r in gsd_map)
    OGF_MAX_PIXELS = 5_000_000  # Skip OGF for images >5MP

    if finest_gsd < 10.0:
        print(
            f"📥 Skipping OGF tiles (finest resolution {finest_gsd}m < 10m)",
            flush=True,
        )
        print(f"   Fine resolutions use procedural water generation only\n", flush=True)
        ogf_tiles = None
    elif total_max_pixels > OGF_MAX_PIXELS:
        print(
            f"📥 Skipping OGF tiles (grid too large: {total_max_pixels/1e6:.1f}MP > {OGF_MAX_PIXELS/1e6:.0f}MP)",
            flush=True,
        )
        print(f"   Using procedural water generation only\n", flush=True)
        ogf_tiles = None
    else:
        print(
            f"📥 Downloading OGF tiles for {finest_gsd}m resolution...",
            flush=True,
        )
        ogf_tiles = download_ogf_tiles(bounds, zoom=14)
        if ogf_tiles is not None:
            print("   ✅ OGF tiles downloaded\n", flush=True)
        else:
            print(
                "   ⚠️  OGF tiles not available, using procedural generation\n",
                flush=True,
            )

    # Generate using BASE-FIRST, DERIVE-DOWN approach
    # This ensures all resolutions show the same world with proper physical downsampling
    try:
        generate_from_base_world(
            biome=biome,
            seed=seed,
            outdir=outdir,
            bounds=bounds,
            profile=profile,
            enable_photorealism=enable_photorealism,
            crs_metric=crs_metric,
            rgb_wgs84=rgb_wgs84,
            resolutions=resolutions,
            ogf_tiles=ogf_tiles,
            ref_map_path=ref_map,
        )
    except Exception as e:
        print(f"❌ Error generating dataset: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return

    # Write zoom metadata (always)
    print("📝 Writing zoom level metadata...", flush=True)

    zoom_metadata = {
        "layers": {},
        "notes": "Use these ranges in Leaflet/MapLibre as layer visibility thresholds.",
    }

    if "25cm" in resolutions:
        zoom_metadata["layers"]["25cm"] = {
            "minZoom": 17,
            "maxZoom": 20,
            "nativeGSD": 0.25,
        }
    if "1m" in resolutions:
        zoom_metadata["layers"]["1m"] = {
            "minZoom": 14, "maxZoom": 17, "nativeGSD": 1.0}
    if "10m" in resolutions:
        zoom_metadata["layers"]["10m"] = {
            "minZoom": 10,
            "maxZoom": 14,
            "nativeGSD": 10.0,
        }

    zoom_json_path = os.path.join(outdir, "zoom_levels.json")
    with open(zoom_json_path, "w") as f:
        json.dump(zoom_metadata, f, indent=2)

    # Zip outputs (only requested resolutions)
    zip_outputs(outdir, resolutions)

    # Validate ZIP was created successfully
    zpath = os.path.join(outdir, "dataset.zip")
    if not os.path.exists(zpath) or os.path.getsize(zpath) < 100:
        raise RuntimeError(f"ZIP was not created or is empty: {zpath}")

    print(
        f"   ✅ ZIP archive validated: {os.path.getsize(zpath) / (1024*1024):.1f} MB\n",
        flush=True,
    )

    print(f"\n{'='*70}", flush=True)
    print("✅ Multi-resolution dataset generation complete!", flush=True)
    print(f"📁 Output directory: {outdir}", flush=True)
    print(f"{'='*70}\n", flush=True)


def zip_outputs(outdir: str, resolutions: list) -> None:
    """
    Create a ZIP archive containing only the requested resolutions.

    Args:
        outdir: Output directory containing GeoTIFF files
        resolutions: List of requested resolution labels (e.g., ['1m', '10m'])
    """
    import zipfile

    # Map resolution labels to file suffixes
    keep = set()
    for r in resolutions:
        if r == "25cm":
            keep.add("25cm")
        elif r == "1m":
            keep.add("1m")
        elif r == "10m":
            keep.add("10m")

    zpath = os.path.join(outdir, "dataset.zip")

    try:
        allowed_prefixes = [
            "dem_",
            "hillshade_",
            "slope_",
            "water_",
            "floodplain_",
            "contours_",
            "rgb_",
        ]

        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fn in os.listdir(outdir):
                if fn == "dataset.zip":
                    continue

                if fn == "zoom_levels.json" or fn == "preview_imagery.png":
                    z.write(os.path.join(outdir, fn), arcname=fn)
                    continue

                if not fn.endswith(".tif"):
                    continue

                has_allowed_prefix = any(
                    fn.startswith(prefix) for prefix in allowed_prefixes
                )
                if not has_allowed_prefix:
                    continue

                if "_25cm" in fn and "25cm" not in keep:
                    continue
                if "_1m" in fn and "1m" not in keep:
                    continue
                if "_10m" in fn and "10m" not in keep:
                    continue

                if not any(tag in fn for tag in ["_25cm", "_1m", "_10m"]):
                    continue

                z.write(os.path.join(outdir, fn), arcname=fn)

        print(f"📦 Zipped outputs: {zpath}", flush=True)
    except Exception as e:
        print(f"⚠️  Failed to create ZIP: {e}", flush=True)


# ------------------ CLI ------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate synthetic terrain and imagery datasets"
    )
    ap.add_argument(
        "--biome",
        type=str,
        default="city",
        choices=[
            "forest",
            "mountain",
            "desert",
            "city",
            "farm",
            "coast",
            "island",
            "water",
        ],
        help="Biome type to generate",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=0,
        help="(Deprecated) Width in pixels. 0 = auto from bounds + GSD",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=0,
        help="(Deprecated) Height in pixels. 0 = auto from bounds + GSD",
    )
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Output directory")
    ap.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="Bounding box coordinates (minLon minLat maxLon maxLat)",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default="high",
        choices=["low", "high", "ultra"],
        help="Quality profile (default: high)",
    )
    ap.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="Comma-separated list of resolutions to generate (25cm,1m,10m). Default: all",
    )
    ap.add_argument(
        "--photorealism",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable photorealistic effects (default: true)",
    )
    ap.add_argument(
        "--crs_metric",
        type=str,
        default="utm",
        choices=["utm", "3857"],
        help="Metric CRS for outputs: utm=TRUE ground meters (default), 3857=Web Mercator",
    )
    ap.add_argument(
        "--rgb_wgs84",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Also write RGB in EPSG:4326 (WGS84) for web maps (default: false)",
    )
    ap.add_argument(
        "--ref_map",
        type=str,
        default=None,
        help="Optional RGB reference map (png/jpg/tif) to derive water mask from (blue pixels)",
    )

    args = ap.parse_args()

    bounds = tuple(args.bounds)
    if len(bounds) != 4:
        ap.error("--bounds requires exactly 4 values: minLon minLat maxLon maxLat")

    minLon, minLat, maxLon, maxLat = bounds
    if minLon >= maxLon or minLat >= maxLat:
        ap.error(
            "Invalid bounds: minLon must be < maxLon and minLat must be < maxLat")

    # Parse resolutions
    resolutions = None
    if args.resolutions:
        resolutions = [r.strip() for r in args.resolutions.split(",")]
        # Validate resolutions
        valid_resolutions = ["25cm", "1m", "10m"]
        for res in resolutions:
            if res not in valid_resolutions:
                ap.error(
                    f'Invalid resolution: {res}. Must be one of: {", ".join(valid_resolutions)}'
                )

    # Parse photorealism flag
    enable_photorealism = args.photorealism.lower() == "true"
    rgb_wgs84 = args.rgb_wgs84.lower() == "true"

    # Map CLI choice to CRS
    if args.crs_metric == "utm":
        crs_metric = "UTM"  # Auto-select UTM zone
    else:
        crs_metric = "EPSG:3857"  # Web Mercator

    main(
        biome=args.biome,
        width=args.width,
        height=args.height,
        seed=args.seed,
        outdir=args.outdir,
        bounds=bounds,
        profile=args.profile,
        resolutions=resolutions,
        crs_metric=crs_metric,
        rgb_wgs84=rgb_wgs84,
        enable_photorealism=enable_photorealism,
        ref_map=args.ref_map,
    )
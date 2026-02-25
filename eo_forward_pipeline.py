
#!/usr/bin/env python3
"""
Complete Electro-Optical (EO) Forward Imaging Pipeline
=======================================================

Transforms categorical biome imagery into photorealistic satellite imagery
by simulating the complete EO sensor acquisition chain.

Pipeline stages:
1. Super-sampled generation (4× internal resolution)
2. Intra-biome continuous texture (multi-octave fractal noise)
3. Terrain-aware shading (hillshade from DEM)
4. Edge realism (soft boundaries, no hard edges)
5. EO optics simulation (sensor MTF)
6. Downsample to target resolution (INTER_AREA)
7. EO colour science / ISP (desaturation, tone curve)

Author: Thales EO Simulation Team
"""


# Optional dependencies
OPENCV_AVAILABLE = False
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    pass

RASTERIO_AVAILABLE = False
try:
    import rasterio
    from rasterio.enums import ColorInterp
    RASTERIO_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EOConfig:
    """Configuration for EO forward imaging pipeline."""
    supersample_factor: int = 4          # Internal rendering scale
    fbm_octaves: int = 6                 # Fractal noise octaves
    fbm_persistence: float = 0.5         # Octave amplitude decay
    fbm_lacunarity: float = 2.0          # Octave frequency increase
    # Base spatial scale for FBM texture (larger = coarser features)
    fbm_base_scale: float = 64.0
    sun_azimuth: float = 315.0           # Sun direction (degrees, 0=N, 90=E)
    sun_altitude: float = 45.0           # Sun elevation (degrees)
    hillshade_strength: float = 0.4      # Terrain shading intensity
    edge_blur_sigma: float = 2.0         # Edge softening strength
    optics_blur_sigma: float = 0.6       # Sensor MTF blur
    desaturation_factor: float = 0.9     # ISP desaturation (1.0=none)
    tone_curve_gamma: float = 1.1        # ISP tone curve
    green_yellow_bias: float = 1.05      # Green/yellow channel boost
    seed: int = 42                       # Random seed for reproducibility
    # Tile size for tiled processing (target resolution)
    tile_size: int = 512
    # Overlap between tiles (halo) to avoid seams
    tile_overlap: int = 64
    sensor_noise_sigma: float = 0.003    # Sensor noise standard deviation
    final_smooth_bilateral_d: int = 5    # Bilateral filter diameter
    final_smooth_bilateral_sigma_color: float = 24.0  # Bilateral color sigma
    final_smooth_bilateral_sigma_space: float = 5.0   # Bilateral space sigma
    # Bilateral smoothing mix (0=off, 1=full)
    final_smooth_mix: float = 0.12
    warp_strength_px: float = 0.6        # Subpixel warping strength
    warp_base: int = 96                  # Noise scale for warp field
    # Post-ISP unsharp mask strength (0=off, 0.3=strong)
    sharpen_strength: float = 0.15


# ============================================================================
# STEP 1: FRACTAL NOISE GENERATION (FBM) - PROPER VALUE NOISE
# ============================================================================

def smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth interpolation function (cubic Hermite)."""
    return t * t * (3 - 2 * t)


def hash_lattice(xi: np.ndarray, yi: np.ndarray, seed: int) -> np.ndarray:
    """
    Deterministic lattice hash: pure function of (xi, yi, seed).

    This is the KEY to world-continuous noise. Grid corner values are computed
    deterministically from their integer coordinates, so adjacent tiles sampling
    the same world coordinates get identical values.

    Uses a high-quality integer hash (xxHash-inspired) that:
    - Works with large xi/yi (whole-world coordinates, no clamping)
    - Is fast and vectorized (no per-call RNG or grid allocation)
    - Produces uniform distribution in [0, 1]

    WHY THIS ELIMINATES SEAMS:
    - Tile A at world position (1000, 2000) and Tile B at (1512, 2000)
      both sample lattice point (1500, 2000) and get IDENTICAL hash value
    - No per-tile random grids, no clamping, no boundary discontinuities

    Args:
        xi: Integer X coordinates (any range, vectorized)
        yi: Integer Y coordinates (any range, vectorized)
        seed: Base seed for this noise layer

    Returns:
        Hash values in [0, 1], same shape as xi/yi
    """
    # Cast to int64 to handle large world coordinates safely
    xi = xi.astype(np.int64)
    yi = yi.astype(np.int64)

    # xxHash-inspired mixing (high avalanche, low bias)
    PRIME1 = np.int64(73856093)
    PRIME2 = np.int64(19349663)
    PRIME3 = np.int64(83492791)

    h = xi.astype(np.int64) * PRIME1
    h = h ^ (yi.astype(np.int64) * PRIME2)
    h = h ^ (np.int64(seed) * PRIME3)

    # Avalanche mixing
    h ^= h >> 16
    h *= np.int64(0x85ebca6b)
    h ^= h >> 13
    h *= np.int64(0xc2b2ae35)
    h ^= h >> 16

    # Map to [0, 1] using low 24 bits (uniform distribution)
    return ((h & 0xFFFFFF).astype(np.float32)) / float(0xFFFFFF)


def value_noise(width: int, height: int, grid: int, seed: int,
                offset_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    World-continuous value noise via deterministic lattice hashing.

    CRITICAL FOR SEAM-FREE TILING:
    - Grid corner values are pure functions of world coordinates (xi, yi, seed)
    - No per-call RNG, no local grid allocation, no clamping
    - Adjacent tiles sampling the same world point get identical values

    Args:
        width: Output width in pixels
        height: Output height in pixels
        grid: Grid spacing (larger = coarser features)
        seed: Seed for this noise layer (deterministic)
        offset_xy: World-space pixel offset (for tiling)

    Returns:
        Noise array (H, W) with values in [0, 1]
    """
    ox, oy = offset_xy

    # World-space coordinates (can be arbitrarily large)
    y = (np.arange(height, dtype=np.float32)[:, None] + oy) / grid
    x = (np.arange(width, dtype=np.float32)[None, :] + ox) / grid

    # Integer lattice coordinates (no clamping, works for whole world)
    yi = np.floor(y).astype(np.int32)
    xi = np.floor(x).astype(np.int32)

    # Fractional parts with smoothstep for C1 continuity
    fy = smoothstep(y - yi)
    fx = smoothstep(x - xi)

    # Sample lattice corners via deterministic hash (WORLD-CONTINUOUS)
    g00 = hash_lattice(xi,     yi,     seed)
    g10 = hash_lattice(xi + 1, yi,     seed)
    g01 = hash_lattice(xi,     yi + 1, seed)
    g11 = hash_lattice(xi + 1, yi + 1, seed)

    # Bilinear interpolation
    n0 = g00 * (1 - fx) + g10 * fx
    n1 = g01 * (1 - fx) + g11 * fx

    return (n0 * (1 - fy) + n1 * fy).astype(np.float32)


def fbm_noise(width: int, height: int, octaves: int = 6,
              persistence: float = 0.5, lacunarity: float = 2.0,
              seed: int = 42, offset_xy: Tuple[float, float] = (0.0, 0.0),
              base_scale: float = 64.0) -> np.ndarray:
    """
    World-continuous Fractional Brownian Motion (FBM) via deterministic lattice hashing.

    CRITICAL FOR SEAM-FREE TILING:
    - Each octave uses deterministic value_noise with world-space offset
    - No per-octave random grids, all values are pure functions of (x, y, seed, octave)
    - Adjacent tiles sampling the same world coordinates get identical FBM values

    WHY THIS WORKS:
    - offset_xy is in world pixel coordinates (e.g., tile origin)
    - Each octave scales offset_xy by its frequency for correct world alignment
    - Lattice hash ensures grid corners have identical values across tiles

    Args:
        width: Output width in pixels
        height: Output height in pixels
        octaves: Number of noise octaves to sum
        persistence: Amplitude decay per octave (typically 0.5)
        lacunarity: Frequency increase per octave (typically 2.0)
        seed: Base seed for reproducibility
        offset_xy: World-space pixel offset (tile origin in full image)
        base_scale: Base spatial scale for noise features

    Returns:
        Noise array (H, W) with values approximately in [-1, 1]
    """
    # Initialize output
    noise = np.zeros((height, width), dtype=np.float32)

    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for octave in range(octaves):
        # Calculate grid size for this octave
        grid = max(2, int(base_scale / frequency))

        # Scale offset_xy by frequency for world continuity
        # This ensures octaves align correctly in world space
        scaled_offset = (offset_xy[0] * frequency, offset_xy[1] * frequency)

        # Unique seed per octave (deterministic)
        octave_seed = seed + octave * 1000

        # Generate value noise for this octave (WORLD-CONTINUOUS)
        octave_noise = value_noise(
            width, height, grid, octave_seed, offset_xy=scaled_offset)

        # Convert from [0,1] to approximately [-1,1] for better mixing
        octave_noise = octave_noise * 2.0 - 1.0

        # Add this octave to the result
        noise += octave_noise * amplitude
        max_value += amplitude

        # Update for next octave
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to approximately [-1, 1]
    if max_value > 0:
        noise /= max_value

    return noise


def apply_water_texture(img: np.ndarray, water_mask: np.ndarray, config: EOConfig,
                        origin_xy: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Apply water-specific texture (ripples, sun-aligned specular highlights).

    WHY THIS IMPROVES REALISM:
    - Water has anisotropic ripples (not isotropic vegetation texture)
    - Specular highlights align with sun direction (not random sparkle)
    - Subtle effect (no cartoon glitter)
    - Mimics real satellite water appearance

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        water_mask: Water mask (H, W), float32 [0-1]
        config: EO pipeline configuration
        origin_xy: Pixel origin for world continuity

    Returns:
        Image with water texture applied (H, W, 3), float32 [0-1]
    """
    h, w = img.shape[:2]
    ox, oy = origin_xy

    # Generate anisotropic ripple patterns (elongated in one direction)
    ripple_x = fbm_noise(w, h, octaves=4, persistence=0.6, lacunarity=2.5,
                         seed=config.seed + 100, offset_xy=(ox, oy), base_scale=32.0)
    ripple_y = fbm_noise(w, h, octaves=4, persistence=0.6, lacunarity=2.5,
                         seed=config.seed + 101, offset_xy=(ox * 0.3, oy * 0.3), base_scale=48.0)

    # Combine for anisotropic ripples
    ripples = (ripple_x * 0.7 + ripple_y * 0.3)
    ripples = (ripples + 1.0) / 2.0  # Normalize to [0,1]

    # Generate sun-aligned specular highlights
    # Compute sun reflection direction based on sun azimuth
    sun_azimuth_rad = np.deg2rad(config.sun_azimuth)
    sun_dir_x = np.cos(sun_azimuth_rad)
    sun_dir_y = np.sin(sun_azimuth_rad)

    # Create directional gradient for sun alignment
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sun_alignment = (xx * sun_dir_x + yy * sun_dir_y) / np.sqrt(w**2 + h**2)
    sun_alignment = (sun_alignment + 1.0) / 2.0  # Normalize to [0,1]

    # Generate base specular noise
    specular_base = fbm_noise(w, h, octaves=3, persistence=0.4, lacunarity=3.0,
                              seed=config.seed + 102, offset_xy=(ox, oy), base_scale=64.0)
    specular_base = (specular_base + 1.0) / 2.0  # [0,1]

    # Modulate specular by sun alignment (highlights along sun direction)
    specular = specular_base * sun_alignment
    # Make sparse and bright (reduce cartoon sparkle)
    specular = specular ** 4.0

    # Apply water texture only where water_mask > 0
    water_mask_3ch = water_mask[..., np.newaxis]

    # Subtle brightness modulation from ripples
    brightness_mod = 0.94 + 0.06 * ripples  # Reduced from 0.08 for subtlety
    img = img * (1 - water_mask_3ch) + img * \
        brightness_mod[..., np.newaxis] * water_mask_3ch

    # Add sun-aligned specular highlights (subtle, more in blue channel)
    specular_rgb = np.stack(
        [specular * 0.2, specular * 0.3, specular * 0.5], axis=-1)
    img = img + specular_rgb * water_mask_3ch * \
        0.08  # Reduced from 0.15 for subtlety

    return np.clip(img, 0, 1)


def apply_intra_biome_texture(img: np.ndarray, config: EOConfig, origin_xy: Tuple[int, int] = (0, 0),
                              water_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply continuous fractal texture within each biome to eliminate flat colors.

    This modulates both brightness and hue locally using multi-octave noise,
    creating natural variation within uniform biome regions. Water bodies
    receive special treatment with ripples and specular highlights.

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        config: EO pipeline configuration
        origin_xy: Pixel origin (x0, y0) in world space for seamless tiling
        water_mask: Optional water mask (H, W), float32 [0-1] or bool

    Returns:
        Textured RGB image (H, W, 3), float32 [0-1]
    """
    h, w = img.shape[:2]
    x0, y0 = origin_xy

    # Convert pixel origin to "world" offsets in the noise domain
    # This ensures adjacent tiles sample the same global noise field
    ox = float(x0)
    oy = float(y0)

    # Generate multiple noise layers for different texture scales
    # CRITICAL: Use same offset_xy for all noise layers to maintain world continuity
    # Use config.fbm_base_scale to control feature size (larger = coarser, less pixelation)
    noise_fine = fbm_noise(w, h, octaves=config.fbm_octaves,
                           persistence=config.fbm_persistence,
                           lacunarity=config.fbm_lacunarity,
                           seed=config.seed,
                           offset_xy=(ox, oy),
                           base_scale=config.fbm_base_scale)

    noise_medium = fbm_noise(w, h, octaves=config.fbm_octaves - 2,
                             persistence=config.fbm_persistence,
                             lacunarity=config.fbm_lacunarity,
                             seed=config.seed + 1,
                             offset_xy=(ox, oy),
                             base_scale=config.fbm_base_scale * 1.5)

    noise_coarse = fbm_noise(w, h, octaves=config.fbm_octaves - 4,
                             persistence=config.fbm_persistence,
                             lacunarity=config.fbm_lacunarity,
                             seed=config.seed + 2,
                             offset_xy=(ox, oy),
                             base_scale=config.fbm_base_scale * 2.5)

    # Normalize noise to [0, 1] for modulation
    noise_fine = (noise_fine + 1.0) / 2.0
    noise_medium = (noise_medium + 1.0) / 2.0
    noise_coarse = (noise_coarse + 1.0) / 2.0

    # Create land mask (inverse of water)
    if water_mask is not None:
        # Normalize water mask to float32 [0,1]
        if water_mask.dtype == bool:
            water_mask_norm = water_mask.astype(np.float32)
        else:
            water_mask_norm = water_mask.astype(np.float32)
            if np.max(water_mask_norm) > 1.0:
                water_mask_norm /= np.max(water_mask_norm)

        # Resize if needed
        if water_mask_norm.shape != (h, w):
            if OPENCV_AVAILABLE:
                # Keep coastlines crisp if mask is binary-like
                water_mask_norm = cv2.resize(
                    water_mask_norm, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                # Fallback: PIL resize (no SciPy dependency)
                from PIL import Image
                water_pil = Image.fromarray(
                    (water_mask_norm * 255).astype(np.uint8))
                water_pil = water_pil.resize((w, h), Image.NEAREST)
                water_mask_norm = np.array(
                    water_pil).astype(np.float32) / 255.0

        land_mask = 1.0 - water_mask_norm
    else:
        # No water mask: apply texture to all pixels
        land_mask = np.ones((h, w), dtype=np.float32)
        water_mask_norm = None

    # Apply land texture (vegetation-style modulation)
    # Apply brightness modulation (affects all channels)
    brightness_mod = 0.85 + 0.15 * noise_fine
    img = img * brightness_mod[..., np.newaxis]

    # Apply hue modulation (affects channels differently) - only on land
    # Red channel: medium-scale variation
    img[..., 0] *= (1.0 + (0.90 + 0.10 * noise_medium - 1.0) * land_mask)

    # Green channel: fine-scale variation (vegetation detail)
    img[..., 1] *= (1.0 + (0.88 + 0.12 * noise_fine - 1.0) * land_mask)

    # Blue channel: coarse-scale variation (atmospheric)
    img[..., 2] *= (1.0 + (0.92 + 0.08 * noise_coarse - 1.0) * land_mask)

    img = np.clip(img, 0, 1)

    # Apply water-specific texture if water mask provided
    if water_mask_norm is not None:
        img = apply_water_texture(img, water_mask_norm, config, origin_xy)

    return img


# ============================================================================
# STEP 2: TERRAIN-AWARE SHADING (HILLSHADE)
# ============================================================================

def compute_hillshade(dem: np.ndarray, azimuth: float = 315.0,
                      altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """
    Compute hillshade from Digital Elevation Model (DEM).

    Hillshade simulates illumination of terrain by a directional light source
    (the sun), creating realistic terrain shading.

    Args:
        dem: Digital Elevation Model (H, W), float32
        azimuth: Sun azimuth angle in degrees (0=N, 90=E, 180=S, 270=W)
        altitude: Sun altitude angle in degrees (0=horizon, 90=zenith)
        z_factor: Vertical exaggeration factor

    Returns:
        Hillshade array (H, W), float32 [0-1]
    """
    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)

    # Compute gradients (slope)
    dy, dx = np.gradient(dem * z_factor)

    # Compute slope and aspect
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)

    # Compute hillshade using standard formula
    # hillshade = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
    zenith_rad = np.pi / 2 - altitude_rad

    hillshade = (
        np.cos(zenith_rad) * np.cos(slope) +
        np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )

    # Normalize to [0, 1]
    hillshade = np.clip(hillshade, 0, 1)

    return hillshade.astype(np.float32)


def apply_terrain_shading(img: np.ndarray, dem: Optional[np.ndarray],
                          config: EOConfig) -> np.ndarray:
    """
    Apply terrain-aware photometric shading using hillshade.

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        dem: Digital Elevation Model (H, W), float32, or None
        config: EO pipeline configuration

    Returns:
        Shaded RGB image (H, W, 3), float32 [0-1]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for terrain shading")

    if dem is None:
        return img

    h, w = img.shape[:2]

    # Resize DEM to match image if needed
    if dem.shape != (h, w):
        dem = cv2.resize(dem, (w, h), interpolation=cv2.INTER_LINEAR)

    # Compute hillshade
    hillshade = compute_hillshade(dem,
                                  azimuth=config.sun_azimuth,
                                  altitude=config.sun_altitude)

    # Apply hillshade modulation
    # Formula: img = img * (base + strength * hillshade)
    base = 1.0 - config.hillshade_strength
    shading = base + config.hillshade_strength * hillshade

    img = img * shading[..., np.newaxis]

    return np.clip(img, 0, 1)


# ============================================================================
# STEP 3: EDGE REALISM (EDGE-AWARE BLENDING AT BIOME BOUNDARIES)
# ============================================================================

def derive_edge_mask_from_labels(img: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Derive edge mask from color label discontinuities (biome boundaries).

    WHY THIS IMPROVES REALISM:
    - Real satellite imagery has soft transitions at biome boundaries (not hard vector edges)
    - Detects where categorical colors meet (forest/field, land/water, etc.)
    - Allows targeted edge-aware blending only where needed

    Method:
    1. Compute gradient magnitude of RGB channels
    2. Threshold to find strong discontinuities (biome boundaries)
    3. Morphological dilation to create blending zone

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        threshold: Gradient magnitude threshold for edge detection

    Returns:
        Edge mask (H, W), float32 [0-1], where 1 = edge/boundary
    """
    if not OPENCV_AVAILABLE:
        return np.zeros(img.shape[:2], dtype=np.float32)

    # Compute gradient magnitude for each channel
    edges = np.zeros(img.shape[:2], dtype=np.float32)

    for c in range(3):
        # Sobel gradients
        gx = cv2.Sobel(img[..., c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img[..., c], cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_mag = np.sqrt(gx**2 + gy**2)
        edges = np.maximum(edges, grad_mag)

    # Threshold to get binary edge map
    edge_binary = (edges > threshold).astype(np.uint8)

    # Dilate to create blending zone (3-5 pixel transition)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dilated = cv2.dilate(edge_binary, kernel, iterations=1)

    # Blur edge mask for smooth blending
    edge_mask = cv2.GaussianBlur(edge_dilated.astype(np.float32), (7, 7), 2.0)

    return edge_mask


def soften_edges(img: np.ndarray, edge_mask: Optional[np.ndarray],
                 config: EOConfig) -> np.ndarray:
    """
    Edge-aware blending at biome boundaries using bilateral filtering.

    WHY THIS IMPROVES REALISM:
    - Replaces global Gaussian blur with targeted edge-aware smoothing
    - Bilateral filter preserves edges while smoothing within regions
    - Only applied at detected biome boundaries (not entire image)
    - Mimics real EO sensor integration across boundary pixels

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        edge_mask: Binary edge mask (H, W), or None (will be derived)
        config: EO pipeline configuration

    Returns:
        Edge-softened RGB image (H, W, 3), float32 [0-1]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for edge softening")

    # Derive edge mask from color discontinuities if not provided
    if edge_mask is None:
        edge_mask = derive_edge_mask_from_labels(img, threshold=0.05)

        # If no significant edges detected, skip processing
        if np.max(edge_mask) < 0.01:
            return img

    h, w = img.shape[:2]
    if edge_mask.shape != (h, w):
        edge_mask = cv2.resize(edge_mask.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)

    # Apply bilateral filter (edge-preserving smoothing)
    # Bilateral filter parameters tuned for biome boundary blending
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_bilateral = cv2.bilateralFilter(
        img_u8, d=7, sigmaColor=30, sigmaSpace=7)
    img_bilateral = img_bilateral.astype(np.float32) / 255.0

    # Blend: apply bilateral smoothing only at edges
    # This creates soft transitions at biome boundaries while preserving texture
    edge_mask_3ch = edge_mask[..., np.newaxis]
    img = img * (1 - edge_mask_3ch) + img_bilateral * edge_mask_3ch

    return np.clip(img, 0, 1)


# ============================================================================
# STEP 4: EO OPTICS SIMULATION (SENSOR MTF)
# ============================================================================

def apply_optics_blur(img: np.ndarray, config: EOConfig) -> np.ndarray:
    """
    Apply sensor Modulation Transfer Function (MTF) blur.

    Real EO sensors have finite optical resolution due to diffraction,
    aberrations, and detector pixel size. This is modeled as a Gaussian PSF.

    FIX #3: Anisotropic blur better approximates real EO optics and reduces
    diagonal aliasing artifacts.

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        config: EO pipeline configuration

    Returns:
        Optically blurred RGB image (H, W, 3), float32 [0-1]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for optics blur")

    # FIX #3: Apply slightly anisotropic Gaussian blur
    # Real EO optics are not perfectly circular, and anisotropy reduces diagonal aliasing
    sigmaX = config.optics_blur_sigma
    sigmaY = config.optics_blur_sigma * 0.7  # Slight anisotropy

    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY,
                           borderType=cv2.BORDER_REFLECT)

    return img


# ============================================================================
# STEP 4.5: ATMOSPHERIC EFFECTS (HAZE / VEILING GLARE)
# ============================================================================

def apply_atmospheric_haze(img: np.ndarray, config: EOConfig, origin_xy: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Apply subtle atmospheric haze and veiling glare for photorealism.

    WHY THIS IMPROVES REALISM:
    - Real satellite imagery is affected by atmospheric scattering
    - Lifts shadows (no pure blacks), reduces local contrast
    - Adds very low-frequency color cast (atmospheric path radiance)
    - Mimics veiling glare from internal sensor reflections

    PLACED AFTER OPTICS BLUR, BEFORE DOWNSAMPLE:
    - Operates at super-sampled resolution for smooth gradients
    - Haze is a scene-level effect (before sensor integration)
    - Applied before ISP (which handles sensor-level color science)

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        config: EO pipeline configuration
        origin_xy: World-space origin for continuous haze pattern

    Returns:
        Image with atmospheric effects (H, W, 3), float32 [0-1]
    """
    h, w = img.shape[:2]

    # Generate very low-frequency haze pattern (world-continuous)
    # Large base_scale = smooth, atmospheric-scale variation
    haze_pattern = fbm_noise(w, h, octaves=3, persistence=0.6, lacunarity=2.0,
                             seed=config.seed + 5000, offset_xy=origin_xy, base_scale=256.0)

    # Normalize to [0, 1] and scale to subtle range
    haze_pattern = (haze_pattern + 1.0) / 2.0
    haze_strength = 0.03 + 0.02 * haze_pattern  # 3-5% haze

    # Atmospheric color cast (slight blue-gray from Rayleigh scattering)
    # Cool atmospheric tint
    haze_color = np.array([0.75, 0.80, 0.85], dtype=np.float32)

    # Apply haze: lifts darks, adds color cast
    # Formula: img_haze = img * (1 - strength) + haze_color * strength
    haze_strength_3ch = haze_strength[..., np.newaxis]
    img = img * (1 - haze_strength_3ch) + haze_color * haze_strength_3ch

    # Veiling glare: reduces local contrast (mimics internal reflections)
    # Apply very subtle global lift to shadows
    img = img * 0.97 + 0.03  # Lift blacks slightly

    return np.clip(img, 0, 1)


# ============================================================================
# STEP 5: EO COLOUR SCIENCE / ISP
# ============================================================================

def apply_isp(img: np.ndarray, config: EOConfig) -> np.ndarray:
    """
    Apply Image Signal Processing (ISP) pipeline.

    Simulates the color processing that occurs in real satellite sensors:
    - Desaturation (atmospheric scattering reduces color purity)
    - Green-yellow bias (vegetation spectral response)
    - Tone curve (sensor response non-linearity)

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        config: EO pipeline configuration

    Returns:
        ISP-processed RGB image (H, W, 3), float32 [0-1]
    """
    # 1. Apply tone curve (gamma correction)
    img = np.power(img, config.tone_curve_gamma)

    # 2. Desaturation (move toward grayscale)
    gray = np.mean(img, axis=2, keepdims=True)
    img = img * config.desaturation_factor + \
        gray * (1 - config.desaturation_factor)

    # 3. Green-yellow bias (vegetation spectral signature)
    img[..., 1] *= config.green_yellow_bias  # Green channel
    # Red channel (slight boost)
    img[..., 0] *= (config.green_yellow_bias - 0.05)

    return np.clip(img, 0, 1)


def apply_sensor_sharpening(img: np.ndarray, config: EOConfig) -> np.ndarray:
    """
    Apply weak post-ISP unsharp mask (sensor sharpening).

    WHY THIS IMPROVES REALISM:
    - Real EO sensors apply subtle sharpening in ISP to compensate for optics blur
    - Very weak strength (0.1-0.3) maintains natural look
    - Applied AFTER ISP (final stage before output)
    - Mimics commercial satellite imagery processing

    Uses unsharp mask: sharpened = img + strength * (img - blur(img))

    Args:
        img: Input RGB image (H, W, 3), float32 [0-1]
        config: EO pipeline configuration

    Returns:
        Sharpened RGB image (H, W, 3), float32 [0-1]
    """
    if not OPENCV_AVAILABLE:
        return img

    # Skip if sharpening disabled
    if not hasattr(config, 'sharpen_strength') or config.sharpen_strength <= 0:
        return img

    # Blur for unsharp mask (small radius for subtle sharpening)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8, sigmaY=0.8)

    # Unsharp mask: img + strength * (img - blurred)
    sharpened = img + config.sharpen_strength * (img - blurred)

    return np.clip(sharpened, 0, 1)


# ============================================================================
# ANTI-BLOCK POST-PROCESSING
# ============================================================================

def apply_subpixel_warp(img: np.ndarray, config: EOConfig, origin_xy: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Subpixel warping in a *global/world* coordinate system so tiles remain seam-free.

    Uses world-space coordinate offsets to ensure adjacent tiles sample the same
    global noise field, eliminating seams at tile boundaries.

    Args:
        img: Input image tile
        config: EO configuration
        origin_xy: (x0, y0) pixel origin of this tile in the full image

    Returns:
        Warped image
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for subpixel warping")

    h, w = img.shape[:2]
    x0, y0 = origin_xy

    # Pass pixel coordinates directly to fbm_noise
    # fbm_noise already divides by grid internally, so no need for arbitrary scaling
    # base_scale controls the feature size
    offset_x = float(x0)
    offset_y = float(y0)

    # Generate noise in world space
    # offset_xy is in pixel coordinates, base_scale controls feature size
    dx_noise = fbm_noise(w, h, octaves=4, seed=config.seed + 111,
                         offset_xy=(offset_x, offset_y), base_scale=config.warp_base)
    dy_noise = fbm_noise(w, h, octaves=4, seed=config.seed + 222,
                         offset_xy=(offset_x, offset_y), base_scale=config.warp_base)

    # Scale to pixel displacement
    dx = dx_noise * config.warp_strength_px
    dy = dy_noise * config.warp_strength_px

    # Build remap coordinates in local tile space
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)

    warped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped


def apply_bilateral_smooth(img: np.ndarray, config: EOConfig) -> np.ndarray:
    """
    Apply edge-aware bilateral smoothing to remove blocky structure.

    Blends smoothed version with original to preserve edges while removing blocks.
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for bilateral smoothing")

    if config.final_smooth_mix <= 0:
        return img

    # Convert to uint8 for bilateral filter
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Apply bilateral filter (edge-preserving)
    smoothed_u8 = cv2.bilateralFilter(
        img_u8,
        d=config.final_smooth_bilateral_d,
        sigmaColor=config.final_smooth_bilateral_sigma_color,
        sigmaSpace=config.final_smooth_bilateral_sigma_space
    )

    # Convert back to float
    smoothed = smoothed_u8.astype(np.float32) / 255.0

    # Blend with original
    result = (1 - config.final_smooth_mix) * img + \
        config.final_smooth_mix * smoothed

    return np.clip(result, 0, 1)


# ============================================================================
# TILED PROCESSING (FOR SCALABILITY)
# ============================================================================

def process_tile(rgb_tile: np.ndarray, dem_tile: Optional[np.ndarray],
                 config: EOConfig, tile_x: int = 0, tile_y: int = 0,
                 origin_xy: Tuple[int, int] = (0, 0),
                 water_mask_tile: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Process a single tile through the EO pipeline.

    This function applies all EO effects to a single tile at super-sampled resolution,
    then downsamples back to target resolution. This is the core processing unit
    for tiled processing.

    Args:
        rgb_tile: RGB tile (H, W, 3), float32 [0-1]
        dem_tile: DEM tile (H, W), float32, or None
        config: EO pipeline configuration
        tile_x: Tile X index for deterministic noise
        tile_y: Tile Y index for deterministic noise
        origin_xy: (x0, y0) pixel origin of this tile in the full image (for seam-free warping)
        water_mask_tile: Optional water mask tile (H, W), float32 [0-1] or bool

    Returns:
        Processed RGB tile (H, W, 3), float32 [0-1]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for tile processing")

    h_target, w_target = rgb_tile.shape[:2]

    # Step 1: Super-sample tile
    w_hr = w_target * config.supersample_factor
    h_hr = h_target * config.supersample_factor

    img_hr = cv2.resize(rgb_tile, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

    dem_hr = None
    if dem_tile is not None:
        dem_hr = cv2.resize(dem_tile, (w_hr, h_hr),
                            interpolation=cv2.INTER_CUBIC)

    water_mask_hr = None
    if water_mask_tile is not None:
        water_mask_hr = cv2.resize(water_mask_tile.astype(np.float32), (w_hr, h_hr),
                                   interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply intra-biome texture
    # CRITICAL: Pass origin_xy scaled to super-sampled resolution for world continuity
    origin_xy_hr = (origin_xy[0] * config.supersample_factor,
                    origin_xy[1] * config.supersample_factor)
    img_hr = apply_intra_biome_texture(img_hr, config, origin_xy=origin_xy_hr,
                                       water_mask=water_mask_hr)

    # Step 3: Apply terrain shading
    if dem_hr is not None:
        img_hr = apply_terrain_shading(img_hr, dem_hr, config)

    # Step 4: Edge-aware blending at biome boundaries
    img_hr = soften_edges(img_hr, edge_mask=None, config=config)

    # Step 5: Apply optics blur (sensor MTF)
    img_hr = apply_optics_blur(img_hr, config)

    # Step 5.5: Apply atmospheric haze (BEFORE downsample, at super-sampled resolution)
    # Placed here because haze is a scene-level effect (before sensor integration)
    img_hr = apply_atmospheric_haze(img_hr, config, origin_xy=origin_xy_hr)

    # FIX #4: Half-pixel integration shift before downsampling
    # Prevents pixel-center locking during sensor integration
    shift_matrix = np.float32([[1, 0, 0.5], [0, 1, 0.5]])
    img_hr = cv2.warpAffine(img_hr, shift_matrix, (img_hr.shape[1], img_hr.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # CRITICAL FIX: Only apply subpixel warp if supersampled
    # Warping at native resolution causes visible block deformation
    if config.supersample_factor > 1 and config.warp_strength_px > 0:
        # Subpixel motion jitter BEFORE sensor integration (physically correct)
        img_hr = apply_subpixel_warp(
            img_hr,
            config,
            origin_xy=origin_xy_hr
        )

    # Step 6: Downsample to target resolution (sensor integration)
    img = cv2.resize(img_hr, (w_target, h_target),
                     interpolation=cv2.INTER_AREA)

    # FIX #2: Add subtle stochastic sensor noise AFTER downsampling
    # Deterministic per tile position using prime number offsets
    rng = np.random.default_rng(config.seed + tile_y * 10007 + tile_x * 7919)
    sensor_noise = rng.normal(
        0, config.sensor_noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img + sensor_noise, 0, 1)

    # Step 7: Apply ISP (sensor color processing)
    # CRITICAL FIX: ISP now happens BEFORE bilateral smoothing
    img = apply_isp(img, config)

    # Step 8: Anti-block post-processing (AFTER ISP)
    # CRITICAL FIX: Bilateral smoothing moved after ISP with reduced strength
    # This prevents it from fighting with pixel-scale noise edges
    img = apply_bilateral_smooth(img, config)

    # Step 9: Apply sensor sharpening (post-ISP unsharp mask)
    # Final stage - subtle sharpening matching commercial EO processing
    img = apply_sensor_sharpening(img, config)

    return img


def eo_forward_pipeline_tiled(rgb_base: np.ndarray,
                              dem: Optional[np.ndarray] = None,
                              config: Optional[EOConfig] = None,
                              water_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    EO forward pipeline with TILED processing for scalability.

    WHY TILING IS REQUIRED:
    -----------------------
    Full-frame 4× super-sampling expands a 2539×2130 image into 10156×8520 (~86 MP),
    which causes:
    1. Memory exhaustion (>6 GB for float32 RGB)
    2. QGIS display issues (max recommended: 10000×10000)
    3. Processing hangs on large areas

    TILED PROCESSING SOLUTION:
    --------------------------
    Instead of processing the entire image at 4× resolution:
    1. Divide image into tiles (e.g., 512×512 at target resolution)
    2. For each tile:
       a) Add overlap (halo) to prevent seam artifacts
       b) Super-sample tile to 4× resolution
       c) Apply all EO effects (texture, shading, blur, etc.)
       d) Downsample tile back to target resolution
       e) Crop overlap and stitch into output
    3. Never hold full 4× image in memory

    BENEFITS:
    ---------
    - Memory usage: O(tile_size²) instead of O(image_size²)
    - Scalable to arbitrarily large images
    - QGIS-compatible output (same size as input)
    - Numerically equivalent to full-frame processing (with overlap)

    Args:
        rgb_base: Base RGB image (H, W, 3), float32 [0-1]
        dem: Digital Elevation Model (H, W), float32, or None
        config: Pipeline configuration, or None for defaults
        water_mask: Optional water mask (H, W), float32 [0-1] or bool

    Returns:
        Photorealistic EO RGB image (H, W, 3), float32 [0-1]
    """
    if config is None:
        config = EOConfig()

    h_full, w_full = rgb_base.shape[:2]

    print(
        f"🛰️  EO Forward Pipeline (Tiled): {w_full}×{h_full} → photorealistic")
    print(
        f"  📐 Tile size: {config.tile_size}×{config.tile_size}, overlap: {config.tile_overlap}px")

    # Determine if tiling is needed
    # Use tiling when the image is bigger than ~1 tile in either dimension
    use_tiling = (w_full > config.tile_size) or (h_full > config.tile_size)

    if not use_tiling:
        print(f"  ℹ️  Image small enough for full-frame processing")
        return eo_forward_pipeline_fullframe(rgb_base, dem, config, water_mask=water_mask)

    # Calculate number of tiles
    tile_size = config.tile_size
    overlap = config.tile_overlap

    # Calculate tile grid
    n_tiles_x = int(np.ceil(w_full / tile_size))
    n_tiles_y = int(np.ceil(h_full / tile_size))

    print(
        f"  🔲 Processing {n_tiles_x}×{n_tiles_y} = {n_tiles_x * n_tiles_y} tiles...")

    # Initialize output
    output = np.zeros_like(rgb_base)

    # Process each tile
    tiles_processed = 0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # Calculate tile bounds with overlap
            x_start = max(0, tx * tile_size - overlap)
            y_start = max(0, ty * tile_size - overlap)
            x_end = min(w_full, (tx + 1) * tile_size + overlap)
            y_end = min(h_full, (ty + 1) * tile_size + overlap)

            # Extract tile with overlap
            rgb_tile = rgb_base[y_start:y_end, x_start:x_end].copy()
            dem_tile = dem[y_start:y_end, x_start:x_end].copy(
            ) if dem is not None else None
            water_mask_tile = water_mask[y_start:y_end, x_start:x_end].copy(
            ) if water_mask is not None else None

            # Compute origin for seam-free warping
            origin_xy = (x_start, y_start)

            # Process tile with tile indices for deterministic noise
            processed_tile = process_tile(rgb_tile, dem_tile, config, tile_x=tx, tile_y=ty,
                                          origin_xy=origin_xy, water_mask_tile=water_mask_tile)

            # Calculate crop region (remove overlap)
            crop_x_start = overlap if tx > 0 else 0
            crop_y_start = overlap if ty > 0 else 0
            crop_x_end = processed_tile.shape[1] - \
                (overlap if tx < n_tiles_x - 1 else 0)
            crop_y_end = processed_tile.shape[0] - \
                (overlap if ty < n_tiles_y - 1 else 0)

            # Crop overlap
            cropped_tile = processed_tile[crop_y_start:crop_y_end,
                                          crop_x_start:crop_x_end]

            # Calculate output position
            out_x_start = tx * tile_size
            out_y_start = ty * tile_size
            out_x_end = min(w_full, out_x_start + cropped_tile.shape[1])
            out_y_end = min(h_full, out_y_start + cropped_tile.shape[0])

            # Stitch into output
            output[out_y_start:out_y_end, out_x_start:out_x_end] = \
                cropped_tile[:out_y_end - out_y_start,
                             :out_x_end - out_x_start]

            tiles_processed += 1
            if tiles_processed % 10 == 0 or tiles_processed == n_tiles_x * n_tiles_y:
                print(
                    f"  ✓ Processed {tiles_processed}/{n_tiles_x * n_tiles_y} tiles...")

    print(f"  ✅ Tiled processing complete!")

    return output


def eo_forward_pipeline_fullframe(rgb_base: np.ndarray,
                                  dem: Optional[np.ndarray] = None,
                                  config: Optional[EOConfig] = None,
                                  water_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    EO forward pipeline with FULL-FRAME processing (original implementation).

    This is the original full-frame implementation, used for small images
    where tiling is not necessary.

    Args:
        rgb_base: Base RGB image (H, W, 3), float32 [0-1]
        dem: Digital Elevation Model (H, W), float32, or None
        config: Pipeline configuration, or None for defaults
        water_mask: Optional water mask (H, W), float32 [0-1] or bool

    Returns:
        Photorealistic EO RGB image (H, W, 3), float32 [0-1]
    """
    if config is None:
        config = EOConfig()

    h_target, w_target = rgb_base.shape[:2]

    print(
        f"🛰️  EO Forward Pipeline (Full-frame): {w_target}×{h_target} → photorealistic")

    # Process entire image (tile indices 0,0 for full-frame)
    # Explicitly pass origin_xy=(0,0) to match tiled behavior
    output = process_tile(rgb_base, dem, config, tile_x=0, tile_y=0,
                          origin_xy=(0, 0), water_mask_tile=water_mask)

    print(f"  ✅ Full-frame processing complete!")

    return output


# ============================================================================
# MAIN PIPELINE (DISPATCHER)
# ============================================================================

def eo_forward_pipeline(rgb_base: np.ndarray,
                        dem: Optional[np.ndarray] = None,
                        config: Optional[EOConfig] = None,
                        water_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Complete EO forward imaging pipeline with automatic tiling for large images.

    This is the main entry point that automatically selects between:
    - Tiled processing (for large images to prevent memory issues)
    - Full-frame processing (for small images where tiling is unnecessary)

    Requires OpenCV for image processing operations.

    AUTOMATIC TILING FOR SCALABILITY:
    ----------------------------------
    Large images (e.g., 2539×2130) would require 10156×8520 at 4× super-sampling
    (~86 MP, >6 GB memory). This causes:
    - Memory exhaustion
    - QGIS display issues
    - Processing hangs

    Solution: Automatically tile large images into manageable chunks (e.g., 512×512),
    process each tile independently with overlap to avoid seams, then stitch back together.

    Pipeline stages (applied per-tile for large images):
    1. Super-sample to 4× resolution
    2. Apply intra-biome continuous texture (FBM noise, water-aware)
    3. Apply terrain-aware shading (hillshade)
    4. Soften edges (no hard boundaries)
    5. Apply sensor optics blur (MTF)
    6. Downsample to target resolution (INTER_AREA)
    7. Apply ISP color processing

    Args:
        rgb_base: Base RGB image (H, W, 3), float32 [0-1]
        dem: Digital Elevation Model (H, W), float32, or None
        config: Pipeline configuration, or None for defaults
        water_mask: Optional water mask (H, W), float32 [0-1] or bool

    Returns:
        Photorealistic EO RGB image (H, W, 3), float32 [0-1]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for EO forward pipeline")

    if config is None:
        config = EOConfig()

    # Dispatch to tiled or full-frame processing
    return eo_forward_pipeline_tiled(rgb_base, dem, config, water_mask)


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def process_geotiff(rgb_tif: str, dem_tif: Optional[str] = None,
                    output_tif: str = "eo_photorealistic.tif",
                    config: Optional[EOConfig] = None,
                    output_dtype: Literal["uint8", "float32"] = "uint8") -> None:
    """
    Process GeoTIFF files through the EO forward pipeline.

    Args:
        rgb_tif: Path to input RGB GeoTIFF (3-band)
        dem_tif: Path to input DEM GeoTIFF (1-band), or None
        output_tif: Path to output photorealistic GeoTIFF
        config: Pipeline configuration, or None for defaults
        output_dtype: Output data type, "uint8" for compatibility or "float32" for precision
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required for GeoTIFF processing")

    print("="*70)
    print("EO Forward Pipeline - GeoTIFF Processing")
    print("="*70)

    # Load RGB
    print(f"📂 Loading RGB: {rgb_tif}")
    with rasterio.open(rgb_tif) as src:
        rgb = src.read()  # (bands, H, W)
        profile = src.profile.copy()

        # Convert to (H, W, 3) and normalize
        rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

    # Load DEM if provided
    dem = None
    if dem_tif is not None:
        print(f"📂 Loading DEM: {dem_tif}")
        with rasterio.open(dem_tif) as src:
            dem = src.read(1).astype(np.float32)

    # Run pipeline
    img_out = eo_forward_pipeline(rgb, dem, config)

    # Write output
    print(f"💾 Writing output: {output_tif}")

    # Convert to output dtype
    if output_dtype == "uint8":
        img_out_u8 = (np.clip(img_out, 0, 1) * 255 + 0.5).astype(np.uint8)
        img_out_chw = np.transpose(img_out_u8, (2, 0, 1))
        profile.update(
            dtype='uint8',
            count=3,
            photometric='RGB',
            compress='deflate'
        )
    else:
        # float32 output
        img_out_chw = np.transpose(img_out.astype(np.float32), (2, 0, 1))
        profile.update(
            dtype='float32',
            count=3,
            compress='deflate',
            predictor=2
        )

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(img_out_chw)
        # Set RGB color interpretation for 3-band images
        if output_dtype == "uint8":
            dst.colorinterp = [ColorInterp.red,
                               ColorInterp.green, ColorInterp.blue]

    print("="*70)
    print("✅ Processing complete!")
    print("="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of the EO forward pipeline."""

    # Example 1: Process GeoTIFF files
    config = EOConfig(
        supersample_factor=4,
        fbm_octaves=6,
        sun_azimuth=315.0,
        sun_altitude=45.0,
        optics_blur_sigma=0.6,
        seed=42
    )

    process_geotiff(
        rgb_tif="rgb_base.tif",
        dem_tif="dem.tif",
        output_tif="eo_photorealistic.tif",
        config=config
    )

    # Example 2: Process NumPy arrays directly
    # rgb_base = np.random.rand(512, 512, 3).astype(np.float32)
    # dem = np.random.rand(512, 512).astype(np.float32)
    # result = eo_forward_pipeline(rgb_base, dem, config)


if __name__ == '__main__':
    example_usage()
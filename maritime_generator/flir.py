"""
FLIR (Forward Looking Infrared) image generation
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, List
import random


def generate_vessel_mask(size: Tuple[int, int] = (256, 256), 
                         vessel_type: str = 'generic',
                         seed: int = None) -> np.ndarray:
    """
    Generate a vessel silhouette mask.
    
    Args:
        size: (width, height) of image
        vessel_type: Type of vessel ('generic', 'small', 'large')
        seed: Random seed
        
    Returns:
        Binary mask array (0-1)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    width, height = size
    mask = np.zeros((height, width), dtype=np.float32)
    
    center_x, center_y = width // 2, height // 2
    
    if vessel_type == 'small':
        # Small vessel: ~30% of image
        hull_length = width * 0.3
        hull_width = height * 0.15
    elif vessel_type == 'large':
        # Large vessel: ~60% of image
        hull_length = width * 0.6
        hull_width = height * 0.3
    else:
        # Generic: ~45% of image
        hull_length = width * 0.45
        hull_width = height * 0.22
    
    # Create hull (ellipse)
    y, x = np.ogrid[:height, :width]
    
    # Main hull
    hull_mask = ((x - center_x) / (hull_length / 2))**2 + \
                ((y - center_y) / (hull_width / 2))**2 <= 1.0
    
    # Add superstructure (rectangle on top)
    superstructure_height = hull_width * 0.4
    superstructure_width = hull_length * 0.6
    superstructure_y = center_y - hull_width / 2 - superstructure_height / 2
    
    superstructure_mask = (
        (x >= center_x - superstructure_width / 2) &
        (x <= center_x + superstructure_width / 2) &
        (y >= superstructure_y - superstructure_height / 2) &
        (y <= superstructure_y + superstructure_height / 2)
    )
    
    mask = (hull_mask | superstructure_mask).astype(np.float32)
    
    return mask


def apply_thermal_effect(mask: np.ndarray, 
                        hot_core_temp: float = 0.9,
                        cool_surrounding: float = 0.2,
                        seed: int = None) -> np.ndarray:
    """
    Apply thermal/FLIR effect to vessel mask.
    
    Args:
        mask: Binary vessel mask
        hot_core_temp: Temperature value for hot core (0-1)
        cool_surrounding: Temperature value for cool background (0-1)
        seed: Random seed
        
    Returns:
        Thermal image array (0-1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = mask.shape
    thermal = np.zeros((height, width), dtype=np.float32)
    
    # Background (cool)
    thermal[:] = cool_surrounding
    
    # Vessel body (hot core)
    thermal[mask > 0.5] = hot_core_temp
    
    # Add heat gradient (hotter in center, cooler at edges)
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    
    # Distance from center
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = dist / (max_dist + 1e-6)
    
    # Gradient: hot in center, cooler at edges
    gradient = 1.0 - normalized_dist * 0.5
    gradient = np.clip(gradient, 0, 1)
    
    # Apply gradient to vessel
    thermal[mask > 0.5] = hot_core_temp * gradient[mask > 0.5]
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, (height, width))
    thermal = np.clip(thermal + noise, 0, 1)
    
    # Add Gaussian blur for thermal diffusion
    thermal_pil = Image.fromarray((thermal * 255).astype(np.uint8))
    thermal_pil = thermal_pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    thermal = np.array(thermal_pil).astype(np.float32) / 255.0
    
    return thermal


def apply_thermal_colormap(thermal: np.ndarray) -> np.ndarray:
    """
    Apply thermal colormap (hot = white/yellow, cool = black/blue).
    
    Args:
        thermal: Thermal image (0-1)
        
    Returns:
        RGB image array (0-1)
    """
    height, width = thermal.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    
    # Thermal colormap: black -> blue -> cyan -> yellow -> white
    # Simplified version: black -> blue -> yellow -> white
    
    # Cool (0.0-0.3): black to blue
    cool_mask = thermal <= 0.3
    cool_val = thermal[cool_mask] / 0.3
    rgb[cool_mask, 0] = cool_val * 0.2  # R
    rgb[cool_mask, 1] = cool_val * 0.3  # G
    rgb[cool_mask, 2] = cool_val * 0.8   # B
    
    # Medium (0.3-0.6): blue to cyan
    medium_mask = (thermal > 0.3) & (thermal <= 0.6)
    medium_val = (thermal[medium_mask] - 0.3) / 0.3
    rgb[medium_mask, 0] = medium_val * 0.3  # R
    rgb[medium_mask, 1] = 0.5 + medium_val * 0.3  # G
    rgb[medium_mask, 2] = 0.8 - medium_val * 0.3  # B
    
    # Hot (0.6-1.0): cyan to yellow to white
    hot_mask = thermal > 0.6
    hot_val = (thermal[hot_mask] - 0.6) / 0.4
    rgb[hot_mask, 0] = 0.3 + hot_val * 0.7  # R
    rgb[hot_mask, 1] = 0.8 + hot_val * 0.2  # G
    rgb[hot_mask, 2] = 0.5 - hot_val * 0.5  # B
    
    return np.clip(rgb, 0, 1)


def generate_flir_frame(frame_num: int, vessel_mask: np.ndarray,
                       base_thermal: np.ndarray,
                       rotation_drift: float = 0.5,
                       translation_drift: float = 2.0,
                       seed: int = None) -> np.ndarray:
    """
    Generate a single FLIR frame with movement.
    
    Args:
        frame_num: Frame number (for consistent movement)
        vessel_mask: Base vessel mask
        base_thermal: Base thermal image
        rotation_drift: Max rotation in degrees
        translation_drift: Max translation in pixels
        seed: Random seed
        
    Returns:
        RGB FLIR image array (0-1)
    """
    if seed is not None:
        np.random.seed(seed + frame_num)
    
    height, width = vessel_mask.shape
    
    # Apply rotation
    angle = np.random.uniform(-rotation_drift, rotation_drift)
    
    # Apply translation
    tx = np.random.uniform(-translation_drift, translation_drift)
    ty = np.random.uniform(-translation_drift, translation_drift)
    
    # Use PIL for rotation and translation
    # Convert to PIL Image
    thermal_pil = Image.fromarray((base_thermal * 255).astype(np.uint8))
    
    # Rotate
    rotated_pil = thermal_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=int(0.2 * 255))
    
    # Translate (using numpy)
    rotated_thermal = np.array(rotated_pil).astype(np.float32) / 255.0
    
    # Manual translation using numpy roll
    height, width = rotated_thermal.shape
    tx_int = int(np.round(tx))
    ty_int = int(np.round(ty))
    
    # Roll for translation
    shifted_thermal = np.roll(rotated_thermal, (ty_int, tx_int), axis=(0, 1))
    
    # Fill edges with background value
    if ty_int > 0:
        shifted_thermal[:ty_int, :] = 0.2
    elif ty_int < 0:
        shifted_thermal[ty_int:, :] = 0.2
    
    if tx_int > 0:
        shifted_thermal[:, :tx_int] = 0.2
    elif tx_int < 0:
        shifted_thermal[:, tx_int:] = 0.2
    
    # Add frame-specific noise
    noise = np.random.normal(0, 0.03, (height, width))
    shifted_thermal = np.clip(shifted_thermal + noise, 0, 1)
    
    # Apply thermal colormap
    rgb = apply_thermal_colormap(shifted_thermal)
    
    return rgb


def generate_flir_sequence(num_frames: int, size: Tuple[int, int] = (256, 256),
                          vessel_type: str = 'generic',
                          seed: int = None) -> List[np.ndarray]:
    """
    Generate a sequence of FLIR frames.
    
    Args:
        num_frames: Number of frames to generate
        size: (width, height) of each frame
        vessel_type: Type of vessel
        seed: Random seed
        
    Returns:
        List of RGB image arrays (0-1)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate base vessel mask
    vessel_mask = generate_vessel_mask(size, vessel_type, seed)
    
    # Generate base thermal image
    base_thermal = apply_thermal_effect(vessel_mask, seed=seed)
    
    # Generate sequence
    frames = []
    for i in range(num_frames):
        frame = generate_flir_frame(i, vessel_mask, base_thermal, seed=seed)
        frames.append(frame)
    
    return frames

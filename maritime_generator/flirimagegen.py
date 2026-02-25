import cv2
import numpy as np

def fake_flir(image_path, output_path):
    # Load image
    img = cv2.imread(image_path)

    # Convert to grayscale (heat intensity base)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to simulate heat diffusion
    heat = cv2.GaussianBlur(gray, (21, 21), 0)

    # Normalize contrast
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)

    # Optional: emphasize edges (hot outlines)
    edges = cv2.Canny(gray, 100, 200)
    heat = cv2.addWeighted(heat, 1.0, edges, 0.4, 0)

    # Apply thermal colormap
    thermal = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)

    # Save result
    cv2.imwrite(output_path, thermal)

fake_flir("vessel.jpg", "vessel_flir.jpg")

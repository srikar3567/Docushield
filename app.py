    import cv2
import numpy as np
from PIL import Image

def analyze_fft_with_window(image_pil):
    """
    Applies a 2D Hanning window to avoid false spikes from crop boundaries.
    """
    img_gray = np.array(image_pil.convert('L'), dtype=np.float32)
    h, w = img_gray.shape
    
    # 2D Hanning Window
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    windowed_img = img_gray * window

    # 2D Fast Fourier Transform
    f_transform = np.fft.fft2(windowed_img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)

    center_y, center_x = h // 2, w // 2
    r_inner = min(h, w) // 8
    r_outer = min(h, w) // 2
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    high_freq_mask = (dist_from_center > r_inner) & (dist_from_center < r_outer)
    high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
    total_energy = np.mean(magnitude_spectrum)

    ratio = high_freq_energy / (total_energy + 1e-5)
    normalized_score = np.clip((ratio - 0.7) * 200, 0, 100)
    return round(float(normalized_score), 2)


def analyze_ai_art_residuals(image_pil):
    """
    Detects smooth latent patterns and unnatural color distributions in AI/Anime art.
    """
    img_rgb = np.array(image_pil.convert('RGB'))
    
    # 1. High-frequency stroke noise
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_variance = laplacian.var()
    smoothness_score = np.clip((250 - noise_variance) / 2.5, 0, 100)

    # 2. Color saturation distribution
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat_std = np.std(hsv[:, :, 1])
    saturation_score = np.clip((sat_std - 45) * 2, 0, 100)

    ai_art_score = 0.6 * smoothness_score + 0.4 * saturation_score
    return round(float(np.clip(ai_art_score, 0, 100)), 2)


def get_combined_synthetic_score(image_pil):
    """
    Combines stabilized FFT with AI art noise residuals.
    """
    fft_score = analyze_fft_with_window(image_pil)
    art_score = analyze_ai_art_residuals(image_pil)
    
    # Takes the dominant synthetic factor
    final_score = round(max(fft_score, art_score) * 0.7 + (fft_score + art_score) * 0.15, 2)
    return min(final_score, 100.0)
    

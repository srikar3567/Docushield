import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io

st.set_page_config(page_title="DocuShield - Document Tamper Detection", layout="centered")

st.title("🛡️ DocuShield: Multi-Modal Document Screener")
st.write("Upload an identity document or image to evaluate tampering and synthetic AI patterns.")

# --- Forensic Algorithms ---

def perform_ela(image_pil, quality=90):
    """
    Computes Error Level Analysis (ELA) to highlight pixel-level compression anomalies.
    """
    buffer = io.BytesIO()
    image_pil.convert('RGB').save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved_image = Image.open(buffer)
    
    # Pixel difference
    ela_image = ImageChops.difference(image_pil.convert('RGB'), resaved_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Compute suspicion percentage based on mean difference intensity
    ela_array = np.array(ela_image)
    tamper_score = np.clip((np.mean(ela_array) / 255.0) * 200, 0, 100)
    return ela_image, round(float(tamper_score), 2)


def analyze_fft_with_window(image_pil):
    """
    Applies a 2D Hanning window before FFT to eliminate sharp crop edge artifacts.
    """
    img_gray = np.array(image_pil.convert('L'), dtype=np.float32)
    h, w = img_gray.shape
    
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    windowed_img = img_gray * window

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
    Detects smooth latent patterns and unnatural saturation typical in AI anime/art.
    """
    img_rgb = np.array(image_pil.convert('RGB'))
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_variance = laplacian.var()
    smoothness_score = np.clip((250 - noise_variance) / 2.5, 0, 100)

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
    final_score = round(max(fft_score, art_score) * 0.7 + (fft_score + art_score) * 0.15, 2)
    return min(final_score, 100.0)


# --- Streamlit Dashboard Execution ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Forensic Processing
    ela_img, ela_score = perform_ela(image)
    ai_score = get_combined_synthetic_score(image)
    
    st.subheader("Forensic Analysis (ELA)")
    st.image(ela_img, caption="Error Level Analysis Heatmap", use_container_width=True)
    
    st.subheader("📊 Multi-Factor Forensic Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tamper Suspicion (ELA)", value=f"{int(ela_score)}%")
    with col2:
        st.metric(label="Synthetic AI Likelihood", value=f"{int(ai_score)}%")
    
    # Final Verdict Decision Logic
    if ela_score > 40 or ai_score > 50:
        st.error("🚨 Verdict: High Suspicion of Tampering / AI Generation")
    elif ela_score > 25 or ai_score > 35:
        st.warning("⚠️ Verdict: Needs Manual Verification")
    else:
        st.success("✅ Verdict: Authentic Document")
    

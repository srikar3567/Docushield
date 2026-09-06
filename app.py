import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ExifTags
import io

st.set_page_config(page_title="DocuShield - Document Tamper Detection", layout="centered")

st.title("🛡️ DocuShield: Multi-Modal Document Screener")
st.write("Upload an identity document or image to evaluate tampering and synthetic AI patterns.")

# --- Metadata Audit ---

def extract_metadata_datetime(image_pil):
    try:
        exif = image_pil._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name in ['DateTimeOriginal', 'DateTime']:
                    return str(value)
    except Exception:
        pass
    return "No EXIF Timestamp Found (Stripped/Digital Export)"


# --- Forensic Algorithms ---

def perform_ela(image_pil, quality=90):
    buffer = io.BytesIO()
    image_pil.convert('RGB').save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved_image = Image.open(buffer)
    
    ela_image = ImageChops.difference(image_pil.convert('RGB'), resaved_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    ela_array = np.array(ela_image)
    tamper_score = np.clip((np.mean(ela_array) / 255.0) * 200, 0, 100)
    return ela_image, round(float(tamper_score), 2)


def analyze_fft_with_window(image_pil):
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
    normalized_score = np.clip((ratio - 0.55) * 220, 0, 100)
    return round(float(normalized_score), 2)


def analyze_ai_art_residuals(image_pil):
    """
    Detects AI anime/illustration synthetic characteristics:
    1. Color quantization and hyper-vivid palette variance
    2. Lack of natural optical sensor noise in smooth regions
    """
    rgb_img = image_pil.convert('RGB')
    arr = np.array(rgb_img, dtype=np.float32)

    # 1. Color Palette Purity (AI art exhibits exaggerated channel divergence)
    std_r, std_g, std_b = np.std(arr[:, :, 0]), np.std(arr[:, :, 1]), np.std(arr[:, :, 2])
    color_vibrancy = (std_r + std_g + std_b) / 3.0
    vibrancy_score = np.clip((color_vibrancy - 48) * 1.8, 0, 100)

    # 2. Local gradient smoothness (AI renders skin/background with zero micro-grain)
    gray = image_pil.convert('L')
    blur = gray.filter(ImageFilter.GaussianBlur(radius=1.5))
    residual_noise = np.abs(np.array(gray, dtype=np.float32) - np.array(blur, dtype=np.float32))
    noise_level = np.mean(residual_noise)
    
    # Real camera images maintain high residual sensor noise (> 4.5); AI anime is ultra-clean (< 3.0)
    smooth_score = np.clip((4.2 - noise_level) * 30.0, 0, 100)

    # 3. Saturation distribution
    hsv_img = image_pil.convert('HSV')
    sat_mean = np.mean(np.array(hsv_img)[:, :, 1])
    sat_score = np.clip((sat_mean - 40) * 1.5, 0, 100)

    ai_score = (0.4 * smooth_score) + (0.35 * vibrancy_score) + (0.25 * sat_score)
    return round(float(np.clip(ai_score, 0, 100)), 2)


def get_combined_synthetic_score(image_pil):
    fft_score = analyze_fft_with_window(image_pil)
    art_score = analyze_ai_art_residuals(image_pil)
    
    # Prioritizes whichever detector strongly flags synthetic structure
    final_score = max(art_score, fft_score * 0.75 + art_score * 0.25)
    return round(min(final_score, 100.0), 2)


# --- Streamlit UI Execution ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    timestamp = extract_metadata_datetime(image)
    st.info(f"🕒 **Metadata Timestamp:** {timestamp}")

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
    
    # Verdict logic
    if ela_score >= 40 or ai_score >= 65:
        st.error("🚨 Verdict: High Suspicion of Tampering / AI Generation")
    elif ela_score >= 25 or ai_score >= 50:
        st.warning("⚠️ Verdict: Needs Manual Verification")
    else:
        st.success("✅ Verdict: Authentic Document")
    
        

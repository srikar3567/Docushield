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
    normalized_score = np.clip((ratio - 0.65) * 260, 0, 100)
    return round(float(normalized_score), 2)


def analyze_ai_illustration_signals(image_pil):
    """
    Identifies AI Anime / Digital Illustrations robustly:
    - Anime line art edges with flat cell shading
    - Color quantization & exaggerated vibrant hues
    - Watermark invariant detection
    """
    img_rgb = image_pil.convert('RGB')
    arr = np.array(img_rgb, dtype=np.float32)

    # 1. Anime / Non-photorealistic color distribution (HSV)
    hsv = np.array(image_pil.convert('HSV'), dtype=np.float32)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Anime art typically has vivid, high-saturation pixels with punchy dynamic ranges
    high_sat_ratio = np.mean(sat > 110)
    anime_color_score = np.clip(high_sat_ratio * 250, 0, 100)

    # 2. Cell-shading / Posterization (Flat regions with sharp boundaries)
    gray = image_pil.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32)
    
    # Non-edge flatness (difference between smooth regions and bold ink outlines)
    edge_mask = edge_arr > 50
    if np.sum(~edge_mask) > 0:
        flat_area_variation = np.std(np.array(gray)[~edge_mask])
        # AI anime characters have extremely uniform flat color fills
        cell_shading_score = np.clip((40.0 - flat_area_variation) * 3.0, 0, 100)
    else:
        cell_shading_score = 0.0

    # 3. Channel divergence (Digital synthetic graphics have extreme RGB channel splits)
    channel_diff = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 2]))
    split_score = np.clip((channel_diff - 30) * 2.0, 0, 100)

    synthetic_art_score = (0.4 * anime_color_score) + (0.35 * cell_shading_score) + (0.25 * split_score)
    return round(float(np.clip(synthetic_art_score, 0, 100)), 2)


def get_combined_synthetic_score(image_pil):
    fft_score = analyze_fft_with_window(image_pil)
    art_score = analyze_ai_illustration_signals(image_pil)
    
    # If illustration/anime indicators fire strongly, it overrides neutral FFT
    final_score = max(art_score, (0.5 * fft_score + 0.5 * art_score))
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
    
    # Decision Logic
    if ela_score >= 45 or ai_score >= 60:
        st.error("🚨 Verdict: High Suspicion of Tampering / AI Generation")
    elif ela_score >= 28 or ai_score >= 40:
        st.warning("⚠️ Verdict: Needs Manual Verification")
    else:
        st.success("✅ Verdict: Authentic Document")
        


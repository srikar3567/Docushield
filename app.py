import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ExifTags
import io

st.set_page_config(page_title="DocuShield - Document Tamper Detection", layout="centered")

st.title("🛡️ DocuShield: Multi-Modal Document Screener")
st.write("Upload an identity document or image to evaluate tampering and synthetic AI patterns.")

# --- Metadata Audit ---

def extract_metadata(image_pil):
    timestamp = "No EXIF Timestamp Found (Digital Export / Stripped)"
    software = "None"
    try:
        exif = image_pil._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name in ['DateTimeOriginal', 'DateTime']:
                    timestamp = str(value)
                elif tag_name == 'Software':
                    software = str(value)
    except Exception:
        pass
    return timestamp, software


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
    tamper_score = np.clip((np.mean(ela_array) / 255.0) * 180, 0, 100)
    return ela_image, round(float(tamper_score), 2)


def analyze_spectral_artifacts(image_pil):
    """
    Detects latent diffusion lattice grids in frequency domain.
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
    r_inner = min(h, w) // 6
    r_outer = min(h, w) // 2
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    high_freq_mask = (dist_from_center > r_inner) & (dist_from_center < r_outer)
    high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
    total_energy = np.mean(magnitude_spectrum)

    ratio = high_freq_energy / (total_energy + 1e-5)
    # AI photoreal generators inject synthetic high-frequency energy
    spectral_score = np.clip((ratio - 0.60) * 250, 0, 100)
    return float(spectral_score)


def analyze_photoreal_ai_grain(image_pil):
    """
    Evaluates micro-texture inconsistencies between foreground and blurred background.
    AI portraits produce mathematically sterile smooth backgrounds compared to real camera sensor ISO noise.
    """
    gray = image_pil.convert('L')
    
    # Residual micro-noise
    blur = gray.filter(ImageFilter.GaussianBlur(radius=1.2))
    residual = np.abs(np.array(gray, dtype=np.float32) - np.array(blur, dtype=np.float32))
    
    # Calculate gradient standard deviation
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32)
    
    # Real camera: high background sensor variance even when blurry
    # AI portrait: background has artificially low micro-texture variance
    bg_mask = edge_arr < 15
    if np.sum(bg_mask) > 100:
        bg_noise = np.std(residual[bg_mask])
        # AI smooth bokeh falls below 1.6
        bg_smooth_score = np.clip((2.0 - bg_noise) * 50.0, 0, 100)
    else:
        bg_smooth_score = 0.0

    return float(bg_smooth_score)


def get_combined_synthetic_score(image_pil, software_tag):
    ai_keywords = ['stable diffusion', 'midjourney', 'dall-e', 'photoshop', 'canva']
    if any(kw in software_tag.lower() for kw in ai_keywords):
        return 85.0
        
    spectral = analyze_spectral_artifacts(image_pil)
    grain = analyze_photoreal_ai_grain(image_pil)
    
    # Prioritizes either frequency lattice or unnatural background smoothness
    final_score = 0.55 * spectral + 0.45 * grain
    return round(float(np.clip(final_score, 0, 100)), 2)


# --- Streamlit UI Execution ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    timestamp, software = extract_metadata(image)
    st.info(f"🕒 **Metadata Timestamp:** {timestamp}")

    ela_img, ela_score = perform_ela(image)
    ai_score = get_combined_synthetic_score(image, software)
    
    st.subheader("Forensic Analysis (ELA)")
    st.image(ela_img, caption="Error Level Analysis Heatmap", use_container_width=True)
    
    st.subheader("📊 Multi-Factor Forensic Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tamper Suspicion (ELA)", value=f"{int(ela_score)}%")
    with col2:
        st.metric(label="Synthetic AI Likelihood", value=f"{int(ai_score)}%")
    
    # Strict verdict logic
    if ela_score >= 40 or ai_score >= 60:
        st.error("🚨 Verdict: High Suspicion of Tampering / Synthetic Generation")
    elif ela_score >= 25 or ai_score >= 38:
        st.warning("⚠️ Verdict: Needs Manual Verification")
    else:
        st.success("✅ Verdict: Authentic Document")
        


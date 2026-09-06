import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import io

st.set_page_config(page_title="DocuShield - Document Tamper Detection", layout="centered")

st.title("🛡️ DocuShield: Multi-Modal Document Screener")
st.write("Upload an identity document or image to evaluate tampering and file anomalies.")

# --- Metadata Audit ---

def check_file_metadata(image_pil):
    timestamp = "No EXIF Timestamp Found (Digital Export / Stripped)"
    software = "Standard / Camera"
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


def compute_synthetic_risk(image_pil, software_tag):
    """
    Evaluates synthetic risk without falsely flagging real camera portraits.
    Real photographs have high natural color entropy and balance across channels.
    """
    img_rgb = image_pil.convert('RGB')
    arr = np.array(img_rgb, dtype=np.float32)
    
    # 1. AI Software signature check
    ai_keywords = ['stable diffusion', 'midjourney', 'dall-e', 'photoshop', 'canva', 'gimp']
    software_flag = any(kw in software_tag.lower() for kw in ai_keywords)
    
    # 2. Check for synthetic anime / vector palette
    # Real photos have continuous tone gradients; synthetic art has posterized saturation
    hsv = np.array(image_pil.convert('HSV'), dtype=np.float32)
    sat = hsv[:, :, 1]
    
    # Check extreme unnatural saturation clusters (Anime/CGI)
    hyper_saturated_pixels = np.mean(sat > 160)
    unnatural_palette_score = np.clip((hyper_saturated_pixels - 0.25) * 150, 0, 100)

    # 3. Channel variance imbalance (real human skin/clothes maintain consistent cross-channel ratio)
    r_g_diff = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]))
    g_b_diff = np.mean(np.abs(arr[:, :, 1] - arr[:, :, 2]))
    channel_disparity = abs(r_g_diff - g_b_diff)
    disparity_score = np.clip((channel_disparity - 35) * 1.5, 0, 100)

    base_score = 0.6 * unnatural_palette_score + 0.4 * disparity_score
    
    if software_flag:
        base_score = max(base_score, 65.0)
        
    return round(float(np.clip(base_score, 0, 100)), 2)


# --- Streamlit UI ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    timestamp, software = check_file_metadata(image)
    st.info(f"🕒 **Metadata Timestamp:** {timestamp}")

    ela_img, ela_score = perform_ela(image)
    ai_score = compute_synthetic_risk(image, software)
    
    st.subheader("Forensic Analysis (ELA)")
    st.image(ela_img, caption="Error Level Analysis Heatmap", use_container_width=True)
    
    st.subheader("📊 Multi-Factor Forensic Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tamper Suspicion (ELA)", value=f"{int(ela_score)}%")
    with col2:
        st.metric(label="Synthetic AI Likelihood", value=f"{int(ai_score)}%")
    
    # Verdict Logic
    if ela_score >= 40 or ai_score >= 60:
        st.error("🚨 Verdict: High Suspicion of Tampering / Synthetic Generation")
    elif ela_score >= 25 or ai_score >= 40:
        st.warning("⚠️ Verdict: Needs Manual Verification")
    else:
        st.success("✅ Verdict: Authentic Document")
        


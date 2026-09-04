import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import numpy as np
import os

st.set_page_config(page_title="DocuShield - SIH26188", layout="wide")

st.title("🛡️ DocuShield: Multi-Layer Document & AI Forgery Detector")
st.caption("Smart India Hackathon | Problem Statement: SIH26188")
st.caption("Developed by: **Srikar** | Dept. of ECE, IIIT Nuzvid")
st.markdown("---")

def analyze_ai_synthetic(image):
    """
    Analyzes frequency domain artifacts using Fast Fourier Transform (FFT)
    and High-Pass Laplacian Energy to detect synthetic / AI generation.
    """
    gray_img = image.convert("L").resize((512, 512))
    img_array = np.array(gray_img, dtype=np.float32)

    # 1. 2D Fast Fourier Transform (Frequency Spectrum Analysis)
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)

    # Calculate high-frequency energy ratio
    rows, cols = 512, 512
    crow, ccol = rows // 2, cols // 2
    r = 60
    # Mask low frequencies in center
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - r : crow + r, ccol - r : ccol + r] = 0
    high_freq_energy = np.mean(magnitude_spectrum[mask == 1])

    # 2. Laplacian High-Pass Spatial Filter (Noise & Edge Consistency)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Simple valid convolution for variance
    h, w = img_array.shape
    padded = np.pad(img_array, 1, mode="reflect")
    filtered = (
        padded[:-2, 1:-1] * laplacian_kernel[0, 1]
        + padded[1:-1, :-2] * laplacian_kernel[1, 0]
        + padded[1:-1, 1:-1] * laplacian_kernel[1, 1]
        + padded[1:-1, 2:] * laplacian_kernel[1, 2]
        + padded[2:, 1:-1] * laplacian_kernel[2, 1]
    )
    laplacian_var = np.var(filtered)

    # AI images exhibit unnatural frequency balance and high Laplacian smoothness/anomalies
    ai_prob = 0
    if high_freq_energy < 105 or high_freq_energy > 165:
        ai_prob += 45
    if laplacian_var < 180 or laplacian_var > 3500:
        ai_prob += 40

    # Extra weight if camera EXIF is missing completely (synthetic indicator)
    exif = image.getexif()
    has_cam_data = any(ExifTags.TAGS.get(k) in ["Make", "Model", "FocalLength"] for k in exif.keys()) if exif else False
    if not has_cam_data:
        ai_prob += 15

    ai_prob = min(ai_prob, 99)
    return ai_prob, magnitude_spectrum


def extract_metadata(image):
    meta_info = {
        "Software": "Clean / Original (No Editing Signature)",
        "Modified Date/Time": "Not recorded / Stripped",
        "Camera/Device": "Unknown / Scanned / Digital",
        "Is_AI_Tagged": False,
        "Is_Edited_Tagged": False
    }
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                val_str = str(value).strip().lower()

                # Check AI tags
                ai_signatures = ["midjourney", "stable diffusion", "dall-e", "dreamstudio", "novelai", "comfyui", "leonardo"]
                if any(k in val_str for k in ai_signatures):
                    meta_info["Software"] = f"🤖 AI Engine ({value})"
                    meta_info["Is_AI_Tagged"] = True
                    continue

                # Check Photo editing software
                edit_signatures = ["photoshop", "gimp", "canva", "picsart", "lightroom", "snapseed"]
                if any(k in val_str for k in edit_signatures):
                    meta_info["Software"] = f"⚠️ Photo Editor ({value})"
                    meta_info["Is_Edited_Tagged"] = True
                    continue

                if tag == "Software" and not meta_info["Is_AI_Tagged"] and not meta_info["Is_Edited_Tagged"]:
                    if "android" in val_str:
                        meta_info["Software"] = "Android System / Downloaded File"
                    else:
                        meta_info["Software"] = str(value)
                elif tag in ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]:
                    if meta_info["Modified Date/Time"] == "Not recorded / Stripped":
                        meta_info["Modified Date/Time"] = str(value)
                elif tag in ["Model", "Make"]:
                    if meta_info["Camera/Device"] == "Unknown / Scanned / Digital":
                        meta_info["Camera/Device"] = str(value)
                    else:
                        meta_info["Camera/Device"] += f" ({value})"
    except Exception:
        pass
    return meta_info


def analyze_ela(image):
    temp_filename = "temp_resaved.jpg"
    rgb_img = image.convert("RGB")
    rgb_img.save(temp_filename, "JPEG", quality=90)
    resaved = Image.open(temp_filename)

    diff = ImageChops.difference(rgb_img, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)

    diff_arr = np.array(diff)
    mean_diff = np.mean(diff_arr)
    tamper_score = min(int((mean_diff / 12.0) * 100), 100)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return ela_image, tamper_score


# Upload Section
uploaded_file = st.file_uploader("Upload ID Card / Document (Aadhaar, PAN, Passport)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Input")
        st.image(input_image, use_container_width=True)

    with st.spinner("Executing Multi-Layer Forensic & Frequency Analysis..."):
        ela_result, tamper_score = analyze_ela(input_image)
        ai_score, mag_spectrum = analyze_ai_synthetic(input_image)
        metadata = extract_metadata(input_image)

    with col2:
        st.subheader("Forensic Analysis (ELA)")
        st.image(ela_result, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Multi-Factor Forensic Assessment")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Tamper Suspicion (ELA)", value=f"{tamper_score}%")
    with m2:
        st.metric(label="Synthetic AI Likelihood (FFT)", value=f"{ai_score}%")
    with m3:
        if metadata["Is_AI_Tagged"] or ai_score >= 65:
            st.error("🤖 **Verdict: Synthetic / AI Generated**")
        elif metadata["Is_Edited_Tagged"] or tamper_score > 50:
            st.error("⚠️ **Verdict: Spliced / Tampered Document**")
        else:
            st.success("✅ **Verdict: Authentic Document**")

    # Detailed Forensic Insights
    st.markdown("---")
    st.subheader("🔍 Deep Forensic & Metadata Audit")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"**Software / Origin:**\n\n{metadata['Software']}")
    with c2:
        st.info(f"**Capture / Mod Date:**\n\n{metadata['Modified Date/Time']}")
    with c3:
        st.info(f"**Device Hardware:**\n\n{metadata['Camera/Device']}")

import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import numpy as np
import os

st.set_page_config(page_title="DocuShield - SIH26188", layout="wide")

st.title("🛡️ DocuShield: AI Document Tamper & Fraud Detector")
st.caption("Smart India Hackathon | Problem Statement: SIH26188")
st.caption("Developed by: **Srikar** | Dept. of ECE, IIIT Nuzvid")
st.markdown("---")

def extract_metadata(image):
    meta_info = {
        "Software": "Clean / Original (No Editing Tool Detected)",
        "Modified Date/Time": "Not recorded / Stripped",
        "Camera/Device": "Unknown / Scanned",
        "Is_AI": False,
        "Is_Edited": False
    }
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                val_str = str(value).strip().lower()
                
                # Check for AI Generation Signatures
                ai_signatures = ["midjourney", "stable diffusion", "dall-e", "dreamstudio", "novelai", "comfyui", "leonardo"]
                if any(ai_sig in val_str for ai_sig in ai_signatures):
                    meta_info["Software"] = f"🤖 AI Generated Image Signature Detected ({value})"
                    meta_info["Is_AI"] = True
                    continue

                # Check for Photo Editing Software Signatures
                edit_signatures = ["photoshop", "gimp", "canva", "picsart", "lightroom", "pixlr"]
                if any(edit_sig in val_str for edit_sig in edit_signatures):
                    meta_info["Software"] = f"⚠️ Photo Editor Detected ({value})"
                    meta_info["Is_Edited"] = True
                    continue

                # Normal metadata mapping
                if tag == "Software" and not meta_info["Is_AI"] and not meta_info["Is_Edited"]:
                    meta_info["Software"] = str(value)
                elif tag in ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]:
                    if meta_info["Modified Date/Time"] == "Not recorded / Stripped":
                        meta_info["Modified Date/Time"] = str(value)
                elif tag in ["Model", "Make"]:
                    if meta_info["Camera/Device"] == "Unknown / Scanned":
                        meta_info["Camera/Device"] = str(value)
                    else:
                        meta_info["Camera/Device"] += f" ({value})"
    except Exception:
        pass
    return meta_info

def analyze_document(image):
    # 1. ELA Processing
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
    
    # 2. Tamper Score Calculation
    diff_arr = np.array(diff)
    mean_diff = np.mean(diff_arr)
    tamper_score = min(int((mean_diff / 12.0) * 100), 100)
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return ela_image, tamper_score

# File Upload Section
uploaded_file = st.file_uploader("Upload ID Card / Document (Aadhaar, PAN, Passport)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Document")
        st.image(input_image, use_container_width=True)
        
    with st.spinner("Analyzing document forensic signatures & metadata..."):
        ela_result, score = analyze_document(input_image)
        metadata = extract_metadata(input_image)
        
    with col2:
        st.subheader("Forensic Analysis (ELA)")
        st.image(ela_result, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📊 Forensic Assessment & Verdict")
    
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        st.metric(label="Tamper Suspicion Score", value=f"{score}%")
        
    with v_col2:
        if metadata["Is_AI"]:
            st.error("🚨 **Verdict: Synthetic AI-Generated Image Detected**")
            st.write("Artificial intelligence synthesis footprints were matched in document parameters.")
        elif metadata["Is_Edited"] or score > 50:
            st.error("⚠️ **Verdict: Potential Tampering / Digital Manipulation Detected**")
            st.write("Discrepancies identified in high-frequency compression regions or editing footprints.")
        else:
            st.success("✅ **Verdict: Document Appears Authentic**")
            st.write("Compression artifacts are uniform across all channels with no manipulation signatures.")

    # Display Metadata Section
    st.markdown("---")
    st.subheader("🔍 EXIF Metadata & Forensic Audit")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.info(f"**Software / Origin:**\n\n{metadata['Software']}")
    with m_col2:
        st.info(f"**Modification Date & Time:**\n\n{metadata['Modified Date/Time']}")
    with m_col3:
        st.info(f"**Device / Source:**\n\n{metadata['Camera/Device']}")

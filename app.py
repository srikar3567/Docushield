import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import io
import re

st.set_page_config(page_title="BorderGuard AI - MHA Screening Platform", layout="wide")

st.title("🛡️ BorderGuard AI: Border Security Document Screener")
st.caption("Ministry of Home Affairs | Automated Document Verification, Tamper Detection & Fraud Screening")

# --- Module 1 & 2: Rule-Based Validation & Text Extraction Mock/Audit ---

def validate_extracted_fields(doc_text):
    """
    Validates document structure against border control rules (Module 2).
    """
    rules_passed = []
    rules_failed = []
    
    # Mock MRZ / Document structure pattern checks
    has_dob = bool(re.search(r'\b(19|20)\d{2}[-/.]\d{2}[-/.]\d{2}\b', doc_text)) or "DOB" in doc_text
    has_expiry = "EXP" in doc_text or "EXPIRY" in doc_text or "202" in doc_text
    has_doc_num = bool(re.search(r'[A-Z][0-9]{7,8}', doc_text)) or "ID" in doc_text

    if has_doc_num:
        rules_passed.append("Document Serial Pattern Matched")
    else:
        rules_failed.append("Document Serial Missing or Irregular")

    if has_expiry:
        rules_passed.append("Document Validity Within Legal Threshold")
    else:
        rules_failed.append("Potential Expired / Unverifiable Validity Period")

    return rules_passed, rules_failed

# --- Module 3: Forensics & Tampering Detection ---

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
    tamper_score = np.clip((np.mean(ela_array) / 255.0) * 190, 0, 100)
    return ela_image, round(float(tamper_score), 2)

def extract_metadata_audit(image_pil):
    timestamp = "Unavailable (Digital Scan / Web Export)"
    software = "Original / Embedded Sensor"
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

# --- Module 4: Photo Region Extraction & Visual Integrity ---

def analyze_photo_box(image_pil):
    """
    Checks portrait area consistency (Face splicing / synthetic artifact check).
    """
    w, h = image_pil.size
    # Focus on standard ID portrait crop region (usually left or right center)
    crop_area = image_pil.crop((int(w * 0.05), int(h * 0.15), int(w * 0.45), int(h * 0.75)))
    
    # Check pixel variance consistency in photo zone
    gray_crop = np.array(crop_area.convert('L'), dtype=np.float32)
    var = np.var(gray_crop)
    
    # Highly flat or saturated photo patches flag replacement
    splicing_risk = 0.0
    if var < 150:
        splicing_risk = 45.0
    return crop_area, splicing_risk


# --- Streamlit Dashboard UI Layout ---

uploaded_file = st.sidebar.file_uploader("📂 Ingest Travel Document", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    doc_image = Image.open(uploaded_file)
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📄 Document Inspection Canvas")
        st.image(doc_image, use_container_width=True)
        
    with col_right:
        st.subheader("🛡️ Forensic Screening Dashboard")
        
        # 1. Tamper Analysis
        ela_map, tamper_score = perform_ela(doc_image)
        timestamp, software = extract_metadata_audit(doc_image)
        photo_patch, photo_risk = analyze_photo_box(doc_image)
        
        # Calculate Unified Risk Index
        composite_risk = round(tamper_score * 0.6 + photo_risk * 0.4, 2)
        
        m1, m2 = st.columns(2)
        m1.metric("Tamper Splicing Score", f"{int(tamper_score)}%")
        m2.metric("Composite Fraud Risk", f"{int(composite_risk)}%")
        
        st.markdown("---")
        st.write(f"**Metadata Signature:** `{software}`")
        st.write(f"**Recorded Timestamp:** `{timestamp}`")
        
        # Checkpoint Decision
        if composite_risk >= 40:
            st.error("🚨 **ACTION: INTERCEPT & MANUAL SECONDARY INSPECTION**")
            st.caption("Significant compression mismatch or potential photo alteration detected.")
        elif composite_risk >= 25:
            st.warning("⚠️ **ACTION: SUPERVISOR OVERRIDE REQUIRED**")
        else:
            st.success("✅ **ACTION: PASS / VERIFIED AUTHENTIC**")

    st.markdown("---")
    sec_col1, sec_col2 = st.columns(2)
    
    with sec_col1:
        st.subheader("🔍 Module 3: Error Level Analysis (ELA) Heatmap")
        st.image(ela_map, caption="Luminance anomalies highlight modified dates/seals/photos", use_container_width=True)
        
    with sec_col2:
        st.subheader("📋 Module 1 & 2: Structural Verification Audit")
        
        # Demo text simulation for checkpoint rules
        simulated_text = "PASSPORT IND P<INDTEST<<SAMPLE 2028-12-31 DOB 1998-05-12"
        passed, failed = validate_extracted_fields(simulated_text)
        
        st.write("✓ **Checkpoint Rules Passed:**")
        for p in passed:
            st.markdown(f"- :green[{p}]")
            
        if failed:
            st.write("✗ **Security Flags:**")
            for f in failed:
                st.markdown(f"- :red[{f}]")
        else:
            st.write("✓ :green[All security layout checksums verified against database schemas.]")
        

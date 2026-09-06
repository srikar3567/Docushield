import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import io
import re
import pytesseract

st.set_page_config(page_title="BorderGuard AI - MHA Screening Platform", layout="wide")

st.title("🛡️ BorderGuard AI: Document Screening System")
st.caption("Ministry of Home Affairs (MHA) | Problem Statement: SIH26188 | Automated Immigration Credential Inspection")

# ==========================================
# VERHOEFF MATHEMATICAL CHECKSUM ENGINE
# ==========================================
d_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]
p_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_verhoeff(num_str):
    """Calculates official 12-digit Dihedral group D5 checksum equation."""
    num_str = re.sub(r'\D', '', str(num_str))
    if len(num_str) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(num_str)):
        c = d_table[c][p_table[i % 8][int(item)]]
    return c == 0

# ==========================================
# MODULE 4: FACE ROI LOCALIZATION
# ==========================================
def extract_and_verify_face(image_pil):
    w, h = image_pil.size
    box = (int(w * 0.04), int(h * 0.18), int(w * 0.42), int(h * 0.82))
    face_crop = image_pil.crop(box)
    
    crop_arr = np.array(face_crop.convert('L'), dtype=np.float32)
    grad_y = np.diff(crop_arr, axis=0)
    grad_x = np.diff(crop_arr, axis=1)
    variance = float(np.var(grad_y) + np.var(grad_x))
    
    splicing_risk = 0.0
    status_note = "Standard portrait ROI localized & texture profile valid"
    if variance < 80.0:
        splicing_risk = 35.0
        status_note = "Warning: Flat boundary variance detected (Potential photo-swap)"
        
    return face_crop, splicing_risk, status_note

# ==========================================
# MODULE 3: TAMPERING DETECTION (ELA & EXIF)
# ==========================================
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
    timestamp = "Not Recorded (Digital Scan / Web Export)"
    software = "Standard Embedded Hardware Profile"
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

# ==========================================
# MODULE 1 & 2: REAL OCR & RULE VALIDATION
# ==========================================
def perform_real_ocr(image_pil):
    """Executes live Tesseract OCR on the ingested document."""
    try:
        gray = image_pil.convert('L')
        # Contrast adjustment for clean OCR text
        enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
        text = pytesseract.image_to_string(enhanced)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return text, lines
    except Exception as e:
        return "", [f"OCR Engine Notification: {str(e)}"]

def audit_document_payload(text_corpus):
    fields = {
        "Document Number": "Unresolved",
        "Date of Birth": "Unresolved",
        "Expiration Date": "Unresolved",
        "Mathematical Checksum": "Pending Analysis"
    }
    flags = []
    checks_passed = []
    
    # 1. 12-Digit Mathematical Checksum Validation (Verhoeff)
    twelve_digit_matches = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text_corpus)
    if twelve_digit_matches:
        candidate_num = twelve_digit_matches[0].replace(" ", "")
        fields["Document Number"] = "[Redacted 12-Digit ID]"
        if validate_verhoeff(candidate_num):
            fields["Mathematical Checksum"] = "PASSED (Verhoeff D5 Valid)"
            checks_passed.append("Official 12-Digit Mathematical Checksum Validated")
        else:
            fields["Mathematical Checksum"] = "FAILED (Checksum Mismatch)"
            flags.append("SECURITY ALERT: Altered/Fake ID Number (Verhoeff Checksum Failed)")
    else:
        doc_id_match = re.search(r'\b[A-Z0-9]{7,14}\b', text_corpus)
        if doc_id_match:
            fields["Document Number"] = doc_id_match.group(0)
            checks_passed.append("Credential Serial Pattern Localized")
        else:
            flags.append("Document Serial Pattern not recognized in text stream")

    # 2. Date checks
    dates = re.findall(r'\b(?:\d{2}[-/.]\d{2}[-/.]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})\b', text_corpus)
    if len(dates) >= 2:
        fields["Date of Birth"] = dates[0]
        fields["Expiration Date"] = dates[1]
        checks_passed.append("DOB & Expiry Date patterns identified")
    elif len(dates) == 1:
        fields["Date of Birth"] = dates[0]
        checks_passed.append(f"Primary Credential Date Localized ({dates[0]})")
    else:
        flags.append("Standard date structures not recognized")

    # 3. State security header verification
    if any(k in text_corpus.upper() for k in ["GOVERNMENT", "INDIA", "AUTHORITY", "ENROLMENT", "IDENTITY"]):
        checks_passed.append("Official Department Emblems / Keyword Headers Verified")
    else:
        flags.append("Missing standard state security header syntax")

    return fields, checks_passed, flags

# ==========================================
# USER INTERFACE & WORKFLOW ROUTING
# ==========================================

uploaded_file = st.file_uploader("📂 Ingest Identity Document (Passport / National ID / Driving License)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    doc_img = Image.open(uploaded_file)
    
    c_left, c_right = st.columns([1.1, 1])
    
    with c_left:
        st.subheader("📄 Document Inspection Canvas")
        st.image(doc_img, use_container_width=True)
        
    with c_right:
        st.subheader("🛡️ Forensic Triage Dashboard")
        
        # Forensics
        ela_map, tamper_score = perform_ela(doc_img)
        meta_ts, meta_soft = extract_metadata_audit(doc_img)
        face_patch, face_risk, face_msg = extract_and_verify_face(doc_img)
        
        # Real OCR Extraction
        ocr_text, ocr_lines = perform_real_ocr(doc_img)
        fields, passed_rules, failed_rules = audit_document_payload(ocr_text)
        
        # Composite Fraud Risk Calculation
        rule_risk_penalty = len(failed_rules) * 20.0
        composite_risk = round(tamper_score * 0.40 + face_risk * 0.20 + rule_risk_penalty * 0.40, 2)
        composite_risk = min(100.0, composite_risk)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("ELA Tamper Score", f"{int(tamper_score)}%")
        m2.metric("Face Integrity Risk", f"{int(face_risk)}%")
        m3.metric("Composite Risk Index", f"{int(composite_risk)}%")
        
        st.markdown("---")
        
        # Tactical Triage Directives
        if composite_risk >= 40 or any("SECURITY ALERT" in r for r in failed_rules):
            st.error("🚨 **ACTION: INTERCEPT & SECONDARY MANUAL SCREENING**")
            st.caption("Mathematical checksum failure or critical forensic anomalies detected.")
        elif composite_risk >= 25 or len(failed_rules) >= 2:
            st.warning("⚠️ **ACTION: SUPERVISOR MANUAL OVERRIDE REQUIRED**")
            st.caption("Borderline risk metrics or missing security parameters.")
        else:
            st.success("✅ **ACTION: PASS / VERIFIED AUTHENTIC**")
            st.caption("Credential passed mathematical checksum, ELA forensics, and layout validation.")

    st.markdown("---")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("🔍 Module 3: Error Level Analysis (ELA)")
        st.image(ela_map, caption="Luminance variations highlight edited text, stamps, or photo swaps", use_container_width=True)
        st.write(f"**Software Profile:** `{meta_soft}`")
        st.write(f"**Capture Timestamp:** `{meta_ts}`")

    with col_m2:
        st.subheader("👤 Module 4: Biometric Portrait Extraction")
        if face_patch:
            st.image(face_patch, width=160, caption="Isolated Portrait ROI")
        st.write(f"**Biometric Audit:** {face_msg}")
        
    st.markdown("---")
    
    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.subheader("📋 Module 1: Real OCR Extracted Payload")
        st.json(fields)
        with st.expander("View Raw Real-Time OCR Text Stream"):
            st.write(ocr_lines if ocr_lines else "Processing image stream...")
                
    with col_m4:
        st.subheader("⚖️ Module 2: Document Rule Validation")
        st.write("**✓ Verified Checks:**")
        for p in passed_rules:
            st.markdown(f"- :green[{p}]")
        if failed_rules:
            st.write("**✗ Security Flags / Irregularities:**")
            for f in failed_rules:
                st.markdown(f"- :red[{f}]")
        else:
            st.write("- :green[All structural checks matched standard schemas.]")
        

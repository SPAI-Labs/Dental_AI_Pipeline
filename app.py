import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO
from supabase_client import store_scan_result  # --- CHANGE 1: Supabase import added ---
# --- 1. CONFIGURATION & MODEL LOADING ---
@st.cache_resource
def load_models():
    # Force loading on CPU for cloud stability and memory management
    model_1 = YOLO("weights/stage1_best.pt")
    model_1.to('cpu')
    model_2 = YOLO("weights/stage2_best.pt")
    model_2.to('cpu')
    return model_1, model_2
# --- 2. ROBUST PREPROCESSING HELPER ---
def clean_image(uploaded_file):
    """Handles mobile rotation (EXIF) and standardizes format to RGB."""
    try:
        img = Image.open(uploaded_file)
        # Corrects rotation from mobile camera orientation tags
        img = ImageOps.exif_transpose(img)
        return img.convert('RGB')
    except Exception:
        return None
# --- 3. AI ENGINE (ROTATION & VALIDATION) ---
def process_single_image(image_np, m1, m2, expected_label):
    """Handles ROI detection with 180° rotation check for mirrored data logic."""
    try:
        image_np = np.asanyarray(image_np).astype('uint8')
        # --- STAGE 1: DUAL-TRY ROI DETECTION ---
        # Attempt 1: Standard RGB
        res_orig = m1(image_np, conf=0.1, imgsz=640, augment=True)
        # Attempt 2: BGR Fallback (Failsafe for color-space mismatches)
        if not res_orig[0].boxes:
            res_orig = m1(image_np[..., ::-1], conf=0.1, imgsz=640)
        best_orig = None
        if res_orig[0].boxes:
            best_orig = sorted(res_orig[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        # Rotation Check: If detection fails or view is wrong, check 180° rotation
        needs_rotation_check = True
        if best_orig:
            detected_view = res_orig[0].names[int(best_orig.cls)].lower()
            if detected_view == expected_label.lower():
                needs_rotation_check = False
        final_img = image_np
        final_res = res_orig
        if needs_rotation_check:
            # Rotate image 180 degrees to match potential mirrored training orientation
            img_rotated = np.asanyarray(Image.fromarray(image_np).rotate(180)).astype('uint8')
            res_rot = m1(img_rotated, conf=0.1, imgsz=640)
            if res_rot[0].boxes:
                best_rot = sorted(res_rot[0].boxes, key=lambda x: x.conf, reverse=True)[0]
                # Use rotated version if it provides better anatomical alignment
                if best_orig is None or best_rot.conf > best_orig.conf:
                    final_img = img_rotated
                    final_res = res_rot
        if not final_res[0].boxes:
            return None, {"error": "No mouth detected"}, None, None, None
        best_box = sorted(final_res[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        coords = best_box.xyxy[0].cpu().numpy().astype(int)
        json_1 = {
            "stage": "01_roi_detection",
            "class": final_res[0].names[int(best_box.cls)],
            "conf": round(float(best_box.conf), 4)
        }
        plot_1 = final_res[0].plot()
        # --- STAGE 2: DISEASE DETECTION ---
        x1, y1, x2, y2 = coords
        h, w, _ = final_img.shape
        # Padding to ensure edge findings are not clipped
        x1, y1, x2, y2 = max(0, x1 - 5), max(0, y1 - 5), min(w, x2 + 5), min(h, y2 + 5)
        crop_img = final_img[y1:y2, x1:x2]
        res_rgb = m2(crop_img, conf=0.10)
        res_bgr = m2(crop_img[..., ::-1], conf=0.10)
        # Optimization: Select mode with higher detection count
        final_res2 = res_bgr if len(res_bgr[0].boxes) > len(res_rgb[0].boxes) else res_rgb
        findings = []
        for box in final_res2[0].boxes:
            findings.append({
                "class": final_res2[0].names[int(box.cls)],
                "conf": round(float(box.conf), 4)
            })
        json_2 = {"stage": "02_diagnosis", "findings": findings}
        plot_2 = final_res2[0].plot()
        return coords, json_1, plot_1, json_2, plot_2
    except Exception as e:
        return None, {"error": str(e)}, None, None, None
# --- 4. MAIN APP INTERFACE (FLEXIBLE INPUT LOGIC) ---
st.set_page_config(page_title="Dental AI: Diagnostic Pipeline", layout="wide")
st.title("Dental AI Diagnostic Pipeline")
m1, m2 = load_models()
VALID_MAP = {"Frontal": "frontal", "Maxilla": "maxilla", "Mandible": "mandible"}
# UI Selection Intact
scan_mode = st.sidebar.radio("Select Scan Type", ["Quick Scan (Frontal Only)", "Full Scan (3 Angles)"])
images_to_process = {}
if scan_mode == "Quick Scan (Frontal Only)":
    st.header("1. Quick Scan: Frontal View")
    up = st.file_uploader("Upload Frontal Image", type=["jpg", "png", "jpeg"], key="q")
    if up: images_to_process["Frontal"] = clean_image(up)
else:
    st.header("1. Full Scan: 3-Angle Upload")
    col1, col2, col3 = st.columns(3)
    with col1:
        up_f = st.file_uploader("Frontal View", type=["jpg", "png", "jpeg"], key="f")
        if up_f: images_to_process["Frontal"] = clean_image(up_f)
    with col2:
        up_mx = st.file_uploader("Maxilla", type=["jpg", "png", "jpeg"], key="mx")
        if up_mx: images_to_process["Maxilla"] = clean_image(up_mx)
    with col3:
        up_md = st.file_uploader("Mandible", type=["jpg", "png", "jpeg"], key="md")
        if up_md: images_to_process["Mandible"] = clean_image(up_md)
# FLEXIBLE TRIGGER: Analysis runs if AT LEAST one image is present
is_ready = len(images_to_process) >= 1
if is_ready:
    if st.button(f"Run Analysis ({len(images_to_process)} image(s))", use_container_width=True):
        for label, img in images_to_process.items():
            if img is None: continue
            st.divider()
            st.subheader(f"Results for: {label} View")
            img_np = np.array(img)
            roi, j1, p1, j2, p2 = process_single_image(img_np, m1, m2, label)
            if roi is not None:
                # --- CHANGE 2: Store image and metadata in Supabase ---
                store_scan_result(img_np, label, scan_mode, j1, j2)
                # View Validation Check
                if j1["class"].lower() != VALID_MAP[label].lower():
                    st.warning(f"Note: Detected {j1['class']} in {label} slot.")
                c_img, c_json = st.columns([1, 1])
                with c_img:
                    # COMMENTED OUT: Stage 1 ROI display
                    # st.image(p1, caption=f"{label}: ROI", width=400)
                    # FIX: Correct bluish color (BGR to RGB)
                    p2_rgb = cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)
                    st.image(p2_rgb, caption=f"{label}: Findings", width=400)
                with c_json:
                    # COMMENTED OUT: Stage 1 and Stage 2 JSON output
                    # tab1, tab2 = st.tabs(["Stage 1 JSON", "Stage 2 JSON"])
                    # with tab1: st.json(j1)
                    # with tab2: st.json(j2)
                    pass
            else:
                st.error(f"Error in {label}: {j1['error']}")
else:
    st.info("Please upload at least one image to begin analysis.")
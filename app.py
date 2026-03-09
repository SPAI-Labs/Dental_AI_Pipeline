import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# --- 1. CONFIGURATION & MODEL LOADING ---
@st.cache_resource
def load_models():
    # Load specific weights as per your local setup
    model_1 = YOLO("weights/stage1_best.pt")
    model_2 = YOLO("weights/stage2_best.pt")
    return model_1, model_2


# --- 2. THE AI ENGINE (SINGLE IMAGE PROCESSOR) ---
def process_single_image(image_np, m1, m2):
    """
    Runs the full 2-stage pipeline on a single numpy image.
    Returns: (roi_coords, stage1_json, stage1_plot, stage2_json, stage2_plot)
    """
    # STAGE 1: ROI
    results_1 = m1(image_np, conf=0.25)
    if len(results_1[0].boxes) == 0:
        return None, {"detected": False, "error": "No mouth detected"}, None, None, None

    best_box = sorted(results_1[0].boxes, key=lambda x: x.conf, reverse=True)[0]
    coords = best_box.xyxy[0].cpu().numpy().astype(int)
    class_name_1 = results_1[0].names[int(best_box.cls)]

    json_1 = {
        "stage": "01_roi_detection",
        "detected": True,
        "class_name": class_name_1,
        "roi_bbox": coords.tolist(),
        "confidence": round(float(best_box.conf), 4)
    }
    plot_1 = results_1[0].plot()

    # STAGE 2: DISEASE (RGB/BGR Double Check)
    x1, y1, x2, y2 = coords
    crop_img = image_np[y1:y2, x1:x2]

    res_rgb = m2(crop_img, conf=0.10)
    img_bgr = crop_img[..., ::-1]
    res_bgr = m2(img_bgr, conf=0.10)

    if len(res_bgr[0].boxes) > len(res_rgb[0].boxes):
        final_res = res_bgr
        mode = "BGR (Color Flip)"
        plot_2 = final_res[0].plot(img=crop_img)  # Plot on RGB for correct colors
    else:
        final_res = res_rgb
        mode = "RGB (Standard)"
        plot_2 = final_res[0].plot()

    findings = []
    for box in final_res[0].boxes:
        lx1, ly1, lx2, ly2 = box.xyxy[0].cpu().numpy()
        findings.append({
            "class_id": int(box.cls),
            "class_name": final_res[0].names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox_global": [float(lx1 + x1), float(ly1 + y1), float(lx2 + x1), float(ly2 + y1)]
        })

    json_2 = {
        "stage": "02_disease_detection",
        "mode_used": mode,
        "total_findings": len(findings),
        "findings": findings
    }

    return coords, json_1, plot_1, json_2, plot_2


# --- 3. MAIN APP INTERFACE ---
st.set_page_config(page_title="Dental AI: Multi-Scan", layout="wide")
st.title("🦷 Dental AI Diagnostic Pipeline")

m1, m2 = load_models()

# Sidebar for Scan Selection
scan_mode = st.sidebar.radio("Select Scan Type", ["Quick Scan (Frontal Only)", "Full Scan (3 Angles)"])

images_to_process = {}

if scan_mode == "Quick Scan (Frontal Only)":
    st.header("1. Quick Scan: Frontal View")
    up = st.file_uploader("Upload Frontal Image", type=["jpg", "png", "jpeg"], key="quick")
    if up:
        images_to_process["Frontal"] = Image.open(up).convert('RGB')

    # Requirement: Only 1 image needed
    is_ready = len(images_to_process) == 1
else:
    st.header("1. Full Scan: 3-Angle Upload")
    col1, col2, col3 = st.columns(3)
    with col1:
        up_f = st.file_uploader("Frontal View", type=["jpg", "png", "jpeg"], key="f")
        if up_f: images_to_process["Frontal"] = Image.open(up_f).convert('RGB')
    with col2:
        up_mx = st.file_uploader("Maxilla (Upper)", type=["jpg", "png", "jpeg"], key="mx")
        if up_mx: images_to_process["Maxilla"] = Image.open(up_mx).convert('RGB')
    with col3:
        up_md = st.file_uploader("Mandible (Lower)", type=["jpg", "png", "jpeg"], key="md")
        if up_md: images_to_process["Mandible"] = Image.open(up_md).convert('RGB')

    # Requirement: All 3 images must be present
    is_ready = len(images_to_process) == 3

# --- UPDATED EXECUTION LOGIC ---
if is_ready:
    if st.button("🚀 Run Complete Analysis", use_container_width=True):
        for label, img in images_to_process.items():
            st.divider()
            st.subheader(f"📊 {label} View Results")

            img_np = np.array(img)
            roi, j1, p1, j2, p2 = process_single_image(img_np, m1, m2)

            if roi is not None:
                # Use columns and width parameter to keep images small
                c1, c2 = st.columns([1, 1])
                with c1:
                    # 'width=400' keeps the images from taking up the whole screen
                    st.image(p1, caption=f"{label}: Stage 1 (ROI)", width=400)
                    st.image(p2, caption=f"{label}: Stage 2 (Findings)", width=400)
                with c2:
                    # Single tab set to avoid repetitive JSON labels
                    tab1, tab2 = st.tabs(["Stage 1 Data", "Stage 2 Data"])
                    with tab1:
                        st.json(j1)
                    with tab2:
                        st.json(j2)
            else:
                st.error(f"Analysis failed for {label}: {j1['error']}")
else:
    # Inform the user what is missing
    needed = 3 if scan_mode == "Full Scan (3 Angles)" else 1
    current = len(images_to_process)
    st.info(f"Please upload all required images to begin. ({current}/{needed} uploaded)")
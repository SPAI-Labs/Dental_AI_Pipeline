import os
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configuration
st.set_page_config(page_title="Dental AI Diagnostic Pipeline", layout="wide")


@st.cache_resource
def load_models():
    # Force CPU loading to maintain stability in shared cloud environments
    m1 = YOLO("weights/stage1_best.pt")
    m1.to('cpu')
    m2 = YOLO("weights/stage2_best.pt")
    m2.to('cpu')
    return m1, m2


def run_inference(image_np, m1, m2):
    # Stage 1: ROI Detection
    res1 = m1(image_np, conf=0.25)
    if not res1[0].boxes:
        return None, {"error": "No mouth detected"}, None, None, None

    box = sorted(res1[0].boxes, key=lambda x: x.conf, reverse=True)[0]
    coords = box.xyxy[0].cpu().numpy().astype(int)

    j1 = {
        "stage": "roi_detection",
        "class": res1[0].names[int(box.cls)],
        "bbox": coords.tolist(),
        "conf": round(float(box.conf), 4)
    }
    p1 = res1[0].plot()

    # Stage 2: Diagnosis (Double Check Logic)
    crop = image_np[coords[1]:coords[3], coords[0]:coords[2]]
    res_rgb = m2(crop, conf=0.10)
    res_bgr = m2(crop[..., ::-1], conf=0.10)

    final_res = res_bgr if len(res_bgr[0].boxes) > len(res_rgb[0].boxes) else res_rgb

    findings = []
    for b in final_res[0].boxes:
        lx1, ly1, lx2, ly2 = b.xyxy[0].cpu().numpy()
        findings.append({
            "class": final_res[0].names[int(b.cls)],
            "conf": round(float(b.conf), 4),
            "bbox_global": [float(lx1 + coords[0]), float(ly1 + coords[1]), float(lx2 + coords[0]),
                            float(ly2 + coords[1])]
        })

    j2 = {"stage": "diagnosis", "findings": findings}
    p2 = final_res[0].plot()

    return coords, j1, p1, j2, p2


# --- UI Interface ---
st.title("Dental AI Diagnostic Pipeline")
m1, m2 = load_models()

VALID_MAP = {"Frontal": "frontal", "Maxilla": "maxilla", "Mandible": "mandible"}
mode = st.sidebar.radio("Scan Type", ["Quick Scan (Frontal)", "Full Scan (3 Angles)"])

uploads = {}
if mode == "Quick Scan (Frontal)":
    f = st.file_uploader("Upload Frontal Image", type=["jpg", "jpeg", "png"])
    if f: uploads["Frontal"] = Image.open(f).convert("RGB")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        f = st.file_uploader("Frontal", type=["jpg", "jpeg", "png"])
        if f: uploads["Frontal"] = Image.open(f).convert("RGB")
    with c2:
        mx = st.file_uploader("Maxilla", type=["jpg", "jpeg", "png"])
        if mx: uploads["Maxilla"] = Image.open(mx).convert("RGB")
    with c3:
        md = st.file_uploader("Mandible", type=["jpg", "jpeg", "png"])
        if md: uploads["Mandible"] = Image.open(md).convert("RGB")

ready = (len(uploads) == 1 if mode.startswith("Quick") else len(uploads) == 3)

if ready and st.button("Run Analysis", use_container_width=True):
    for label, img in uploads.items():
        st.divider()
        st.subheader(f"Analysis: {label}")
        img_np = np.array(img)
        roi, j1, p1, j2, p2 = run_inference(img_np, m1, m2)

        if roi is not None:
            if j1["class"].lower() != VALID_MAP[label]:
                st.error(f"Validation Failed: Detected {j1['class']} in {label} slot.")
                continue

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(p1, caption="ROI Detection")
                st.image(p2, caption="Disease Detection")
            with col_b:
                st.json(j2)
        else:
            st.error(f"Processing Failed: {j1['error']}")
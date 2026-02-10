import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# --- CONFIGURATION ---
@st.cache_resource
def load_models():
    # Load your specific weights
    model_1 = YOLO("weights/stage1_best.pt")
    model_2 = YOLO("weights/stage2_best.pt")
    return model_1, model_2


# --- STAGE 1: ROI DETECTION ---
def run_stage_1(model, image_np):
    # Run Stage 1
    results = model(image_np, conf=0.25)

    if len(results[0].boxes) > 0:
        best_box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        coords = best_box.xyxy[0].cpu().numpy().astype(int)

        output_json = {
            "stage": "01_roi_detection",
            "detected": True,
            "roi_bbox": coords.tolist(),
            "confidence": round(float(best_box.conf), 4)
        }
        # Plotting on the original image for visualization
        plot_img = results[0].plot()
        return coords, output_json, plot_img
    else:
        return None, {"stage": "01_roi_detection", "detected": False}, None


# --- STAGE 2: DISEASE DETECTION ---
def run_stage_2(model, crop_img, roi_coords):
    # STRATEGY: "The Double Check" with Color Correction

    # Run 1: Normal RGB
    results_rgb = model(crop_img, conf=0.10)

    # Run 2: BGR (OpenCV format)
    img_bgr = crop_img[..., ::-1]
    results_bgr = model(img_bgr, conf=0.10)

    # Pick the result with MORE findings
    if len(results_bgr[0].boxes) > len(results_rgb[0].boxes):
        final_results = results_bgr
        used_mode = "BGR (Color Flip)"
        # CRITICAL FIX: Plot the BGR detections onto the RGB image
        # This ensures the user sees "Pink" gums, not Blue
        plot_img = final_results[0].plot(img=crop_img)
    else:
        final_results = results_rgb
        used_mode = "RGB (Standard)"
        plot_img = final_results[0].plot()

    origin_x, origin_y = roi_coords[0], roi_coords[1]
    findings = []

    for box in final_results[0].boxes:
        lx1, ly1, lx2, ly2 = box.xyxy[0].cpu().numpy()

        # Translation
        gx1 = lx1 + origin_x
        gy1 = ly1 + origin_y
        gx2 = lx2 + origin_x
        gy2 = ly2 + origin_y

        cls_id = int(box.cls)
        cls_name = final_results[0].names[cls_id]

        findings.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(float(box.conf), 4),
            "bbox_global": [float(gx1), float(gy1), float(gx2), float(gy2)]
        })

    output_json = {
        "stage": "02_disease_detection",
        "mode_used": used_mode,
        "total_findings": len(findings),
        "findings": findings
    }

    return output_json, plot_img


# --- MAIN APP INTERFACE ---
st.set_page_config(page_title="Dental AI Final", layout="wide")
st.title("Dental AI: Final Deployment Pipeline")

model_1, model_2 = load_models()

# Section 1: Upload
st.header("1. Upload Image")
uploaded_file = st.file_uploader("Upload Patient Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Show Original
    st.image(image, caption="Original Input", width=300)

    if st.button("Run Complete Analysis"):

        # --- STAGE 1 EXECUTION ---
        roi_coords, json_1, plot_1 = run_stage_1(model_1, image_np)

        if roi_coords is not None:
            st.divider()
            st.header("2. Stage 1 Results (ROI)")

            # Create two columns for Stage 1
            col1_a, col1_b = st.columns(2)
            with col1_a:
                st.image(plot_1, caption="Stage 1: Detected Mouth", use_container_width=True)
            with col1_b:
                st.subheader("Stage 1 JSON")
                st.json(json_1)

            # --- STAGE 2 EXECUTION ---
            x1, y1, x2, y2 = roi_coords
            crop_img = image_np[y1:y2, x1:x2]

            json_2, plot_2 = run_stage_2(model_2, crop_img, roi_coords)

            st.divider()
            st.header("3. Stage 2 Results (Disease)")

            # Create two columns for Stage 2
            col2_a, col2_b = st.columns(2)
            with col2_a:
                st.image(plot_2, caption=f"Stage 2: Findings ({json_2['mode_used']})", use_container_width=True)
            with col2_b:
                st.subheader("Stage 2 JSON")
                st.json(json_2)

        else:
            st.error("Error: Could not detect mouth (Stage 1 Failed).")
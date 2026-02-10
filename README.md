# Dental_AI_Pipeline
Stage_01 and Stage_02

**Integration Guide**

Here is the refined **Integration README** for your mobile application developer. It focuses strictly on how to use the repository to set up the backend, connect the mobile app, and implement the frontend visualization with your specific color requirements.

---

# Mobile Integration Guide: Dental AI Pipeline

## 1. Project Overview

This repository contains the trained models and inference logic for a 2-Stage Dental Disease Detection System.

* **Stage 01:** Detects the teeth region (ROI) 5 classes.
* **Stage 02:** Detects 13 specific dental diseases within that region.

**Goal:**

1. Host the AI models as a REST API (using the weights and logic in this repo).
2. Connect the Mobile App (React Native) to this API.
3. Visualize the returned JSON data on the mobile screen using specific color codes.

---

## 2. Backend Setup (How to use this Repo)

**Do not** connect the mobile app to the Streamlit URL (`.streamlit.app`). That is for visual testing only.
You must wrap the inference logic in a simple API (e.g., FastAPI) to communicate with the mobile app.

### **Step 2.1: Files You Need**

Clone this repository and locate:

* `weights/stage1_best.pt` (ROI Model)
* `weights/stage2_best.pt` (Disease Model)
* `app.py` (Contains the inference logic functions: `run_stage_1` and `run_stage_2`)

### **Step 2.2: The API Logic**

Create a generic API endpoint (e.g., `POST /predict`) that accepts an image and runs the following Python logic (reference `app.py`):

1. **Load Image:** Convert uploaded file to numpy array.
2. **Run Stage 1:** Get ROI coordinates `[x1, y1, x2, y2]`.
3. **Crop:** Crop the image using those coordinates.
4. **Run Stage 2:** Run inference on the crop.
5. **Translate Coordinates:** Convert Stage 2 local crop coordinates to global image coordinates:
* `Global_X = Local_X + ROI_x1`
* `Global_Y = Local_Y + ROI_y1`


6. **Return JSON:** Structure the response as defined below.

---

## 3. API Response Structure (JSON)

The mobile app expects a response containing two distinct data objects: one for the ROI crop (Stage 1) and one for disease findings (Stage 2).

**Expected JSON Format:**

```json
{
  "status": "success",
  "meta": {
    "image_width": 3000,
    "image_height": 4000
  },
  "stage_01_roi": {
    "detected": true,
    "class_name": "maxilla",       
    "confidence": 0.98,
    "bbox": [100, 500, 2900, 2000] // [x1, y1, x2, y2]
  },
  "stage_02_disease": {
    "total_findings": 3,
    "findings": [
      {
        "class_id": 1,
        "class_name": "calculus",
        "confidence": 0.85,
        "bbox": [1200, 1600, 1400, 1800] // Global Coordinates (for overlay)
      },
      {
        "class_id": 2,
        "class_name": "caries",
        "confidence": 0.72,
        "bbox": [2100, 1550, 2250, 1700]
      }
    ]
  }
}

```

---

## 4. Frontend Implementation (Mobile App)

### **4.1 Workflow**

1. **Capture:** User takes photo.
2. **Send:** App uploads photo to your API endpoint.
3. **Receive:** App gets the JSON above.
4. **Render Stage 1:** Draw a dashed boundary box using `stage_01_roi.bbox` (Visual confirmation of detection).
5. **Render Stage 2:** Iterate through `stage_02_disease.findings` and draw colored bounding boxes.

### **4.2 Stage 02 Color Mapping**

You must implement the specific color scheme below for the disease classes.
*Note: The backend may use BGR (OpenCV format), but React Native requires **Hex** or **RGB**. The conversion is provided below.*

**Color Reference Table:**

| Class ID | Class Name | BGR (Backend) | **Hex Code (Frontend Use)** | Color Description |
| --- | --- | --- | --- | --- |
| 0 | `stain` | (0, 0, 255) | **#FF0000** | Red |
| 1 | `calculus` | (0, 255, 0) | **#00FF00** | Bright Green |
| 2 | `caries` | (235, 206, 135) | **#87CEEB** | Sky Blue |
| 3 | `pitt_n_fissure_caries` | (0, 255, 255) | **#FFFF00** | Yellow |
| 4 | `fracture` | (255, 0, 255) | **#FF00FF** | Magenta |
| 5 | `root_stump` | (255, 255, 0) | **#00FFFF** | Cyan |
| 6 | `restoration` | (0, 140, 255) | **#FF8C00** | Deep Orange |
| 7 | `prosthetics` | (226, 43, 138) | **#8A2BE2** | Lavender/Orchid |
| 8 | `attrition` | (180, 105, 255) | **#FF69B4** | Bright Pink |
| 9 | `tooth_gap` | (128, 128, 0) | **#008080** | Teal |
| 10 | `defective_prosthesis` | (0, 128, 128) | **#808000** | Olive |
| 11 | `gingival_recession` | (19, 69, 139) | **#8B4513** | Brown |
| 12 | `missing_tooth` | (60, 60, 60) | **#3C3C3C** | Dark Grey |

### **4.3 Implementation Snippet (React Native Concept)**

```javascript
const DISEASE_COLORS = {
  "stain": "#FF0000",
  "calculus": "#00FF00",
  "caries": "#87CEEB",
  // ... map all 13 classes
};

// Rendering Logic
{findings.map((item, index) => (
  <View
    key={index}
    style={{
      position: 'absolute',
      borderColor: DISEASE_COLORS[item.class_name] || 'white', // Default fallback
      borderWidth: 2,
      left: scale(item.bbox[0]), // You must scale server coords to screen size
      top: scale(item.bbox[1]),
      width: scale(item.bbox[2] - item.bbox[0]),
      height: scale(item.bbox[3] - item.bbox[1]),
    }}
  />
))}

```

---

## 5. Do's and Don'ts

### **Do:**

* **Coordinate Scaling:** The API returns coordinates for the *original resolution* image (e.g., 3000px wide). You **must** calculate the scale factor (`PhoneScreen_Width / Image_Width`) and multiply all coordinates by this factor before rendering.
* **Handle Empty Results:** If `stage_01_roi.detected` is `false`, prompt the user to retake the photo. Do not attempt to render Stage 2.
* **Use Class Names:** Map colors based on `class_name` (string) rather than `class_id` (int) to avoid issues if the ID order changes in future model updates.

### **Don't:**

* **Do Not Use BGR:** Ensure you use the **Hex** codes provided in Section 4.2. BGR codes (like `0,0,255`) will appear as Blue instead of Red on mobile screens.
* **Do Not Hardcode:** Do not hardcode the API URL to `localhost`. Use your machine's local IP (e.g., `192.168.x.x`) for testing on physical devices.

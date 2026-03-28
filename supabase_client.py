import streamlit as st
from supabase import create_client
import uuid
from datetime import datetime
import json
import cv2

# --- CONNECTION ---
@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

BUCKET_NAME = "intraoral-images"

# --- FILE NAMING ---
def generate_file_name(scan_type):
    """
    Generates a unique file name using timestamp + UUID.
    Example: frontal/2026-03-28_14-32-01_a3f9c1.jpg
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())[:6]
    folder = scan_type.lower()
    file_name = f"{timestamp}_{unique_id}.jpg"
    storage_path = f"{folder}/{file_name}"
    return file_name, storage_path

# --- IMAGE UPLOAD ---
def upload_image_to_supabase(image_np, scan_type):
    """
    Converts numpy image to JPEG bytes and uploads to Supabase Storage.
    Returns (file_name, storage_path) on success, (None, None) on failure.
    """
    try:
        supabase = get_supabase_client()
        file_name, storage_path = generate_file_name(scan_type)

        # Convert numpy array to JPEG bytes
        # success, buffer = cv2.imencode(".jpg", image_np)
        # Convert RGB to BGR before encoding (cv2.imencode expects BGR)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", image_bgr)

        if not success:
            return None, None
        image_bytes = buffer.tobytes()

        # Upload to Supabase Storage bucket
        supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        return file_name, storage_path

    except Exception as e:
        st.warning(f"Image upload failed: {e}")
        return None, None

# --- METADATA SAVE ---
def save_metadata_to_supabase(scan_type, file_name, storage_path,
                               stage1_class, stage1_conf,
                               stage2_findings, scan_mode):
    """
    Saves a metadata row to the image_uploads table in Supabase Database.
    """
    try:
        supabase = get_supabase_client()

        # Convert findings list to JSON string for storage
        findings_str = json.dumps(stage2_findings) if stage2_findings else None

        row = {
            "scan_type": scan_type,
            "file_name": file_name,
            "storage_path": storage_path,
            "stage1_class": stage1_class,
            "stage1_conf": stage1_conf,
            "stage2_findings": findings_str,
            "scan_mode": scan_mode
        }

        supabase.table("image_uploads").insert(row).execute()

    except Exception as e:
        st.warning(f"Metadata save failed: {e}")

# --- COMBINED MASTER FUNCTION ---
def store_scan_result(image_np, scan_type, scan_mode, j1, j2):
    """
    Master function called from app.py after every successful inference.
    Handles both image upload and metadata saving in one call.
    """
    # Upload image
    file_name, storage_path = upload_image_to_supabase(image_np, scan_type)
    if file_name is None:
        return  # Upload failed, skip metadata too

    # Extract Stage 1 info
    stage1_class = j1.get("class") if j1 else None
    stage1_conf = j1.get("conf") if j1 else None

    # Extract Stage 2 findings
    stage2_findings = j2.get("findings") if j2 else None

    # Save metadata row
    save_metadata_to_supabase(
        scan_type=scan_type,
        file_name=file_name,
        storage_path=storage_path,
        stage1_class=stage1_class,
        stage1_conf=stage1_conf,
        stage2_findings=stage2_findings,
        scan_mode=scan_mode
    )
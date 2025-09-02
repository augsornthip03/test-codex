import os
import json
import tempfile
from typing import Optional, Dict, List

import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

import cv2
import numpy as np
import re

# === Hardcoded Google Document AI Config ===
PROJECT_ID = "170160636715"
LOCATION = "us"
PROCESSOR_ID = "137ba540097e229d"


import cv2
import numpy as np
from PIL import Image

def preprocess_for_checkboxes(image: Image.Image):
    """Preprocess image to get morph and contours, like dev code."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.dilate(morph, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return morph, contours


def detect_checkboxes_in_roi(roi, morph, contours, margin=4):
    """
    Detect checkbox status inside ROI using same logic as dev script.
    Returns ("Checked"/"Unchecked", debug_dict).
    """
    x0, y0, x1, y1 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    w0, h0 = x1 - x0, y1 - y0

    checked = False
    found_box = False
    debug_info = {"ratios": [], "rectangles_found": 0}

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if 20 < w < 80 and 20 < h < 80 and 0.8 < w/h < 1.2:
            if x0 <= x and y0 <= y and (x+w) <= x1 and (y+h) <= y1:
                found_box = True
                roi_box = morph[y:y+h, x:x+w]
                inner = roi_box[margin:-margin, margin:-margin]
                if inner.size == 0:
                    continue

                filled_ratio = cv2.countNonZero(inner) / float(inner.size)
                debug_info["ratios"].append(filled_ratio)
                debug_info["rectangles_found"] += 1

                if filled_ratio > 0.9:   # threshold same as dev
                    checked = True

    if not found_box:
        roi_mask = morph[y0:y1, x0:x1]
        ink_ratio = cv2.countNonZero(roi_mask) / float(roi_mask.size)
        debug_info["ratios"].append(ink_ratio)
        if ink_ratio > 0.05:
            checked = True

    return ("Checked" if checked else "Unchecked"), debug_info


# ==============================
# Helper: extract numbers
# ==============================
def extract_numbers(text: str) -> List[str]:
    thai_to_arabic = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
    text = text.translate(thai_to_arabic)
    matches = re.findall(r"\d+(?:[\d,]*\d)?(?:\.\d+)?", text)
    return [m.replace(",", "") for m in matches]

# ==============================
# ROI Masking
# ==============================
def apply_roi_mask(image: Image.Image, rois: List[Dict]) -> Image.Image:
    w, h = image.size
    white_bg = Image.new("RGB", (w, h), color=(255, 255, 255))
    for r in rois:
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        roi = image.crop((x1, y1, x2, y2))
        white_bg.paste(roi, (x1, y1))
    return white_bg

def bbox_to_pixel(bounding_poly, img_width, img_height):
    xs = [v.x for v in bounding_poly.normalized_vertices]
    ys = [v.y for v in bounding_poly.normalized_vertices]
    return (int(min(xs) * img_width), int(min(ys) * img_height),
            int(max(xs) * img_width), int(max(ys) * img_height))

def inside_roi(bbox, roi):
    bx1, by1, bx2, by2 = bbox
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return (roi["x1"] <= cx <= roi["x2"]) and (roi["y1"] <= cy <= roi["y2"])

# ==============================
# OpenCV checkbox detector
# ==============================
# ==============================
# OpenCV checkbox detector (with debug)
# ==============================
def check_checkboxes_in_roi(roi_box: np.ndarray) -> str:
    """Check a cropped ROI box and decide if it's checked.
       Returns 'Checked' or 'Unchecked' only."""
    gray = cv2.cvtColor(roi_box, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.dilate(morph, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    rect_statuses = []
    rectangles_found = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 20 < w < 80 and 20 < h < 80 and 0.8 < w/h < 1.2:
            inner = thresh[y+4:y+h-4, x+4:x+w-4]
            if inner.size == 0:
                continue
            filled_ratio = cv2.countNonZero(inner) / float(inner.size)
            rectangles_found += 1
            if filled_ratio > 0.09:
                rect_statuses.append("Checked")
            else:
                rect_statuses.append("Unchecked")

    # Stricter rule
    if rectangles_found == 0:
        ink_ratio = cv2.countNonZero(thresh) / float(thresh.size)
        return "Checked" if ink_ratio > 0.05 else "Unchecked"
    elif len(rect_statuses) > 1 and "Unchecked" in rect_statuses:
        return "Unchecked"
    else:
        return "Checked" if "Checked" in rect_statuses else "Unchecked"


def process_with_documentai(image_path: str,
                            mime_type: str,
                            rois: List[Dict],
                            field_mask: Optional[str] = None,
                            processor_version_id: Optional[str] = None
                            ) -> Dict[str, str]:
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if processor_version_id:
        name = client.processor_version_path(PROJECT_ID, LOCATION,
                                             PROCESSOR_ID, processor_version_id)
    else:
        name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    with open(image_path, "rb") as image:
        image_content = image.read()

    raw_document = documentai.RawDocument(content=image_content,
                                          mime_type=mime_type)

    request = documentai.ProcessRequest(
        name=name, raw_document=raw_document, field_mask=field_mask,
        process_options=documentai.ProcessOptions(
            individual_page_selector=documentai.ProcessOptions.
            IndividualPageSelector(pages=[1])
        ),
    )

    result = client.process_document(request=request)
    document = result.document

    img_width = document.pages[0].dimension.width
    img_height = document.pages[0].dimension.height

    # Load image for checkbox detection
    img_pil = Image.open(image_path).convert("RGB")
    img_cv = np.array(img_pil)

    # Group ROIs
    grouped = {}
    for roi in rois:
        gname = roi.get("group") or roi["id"]
        grouped.setdefault(gname, []).append(roi)

    roi_results = {}

    for gname, group_rois in grouped.items():
        roi_type = group_rois[0]["type"]

        if roi_type == "checkbox":
            checked_values = []
            for roi in group_rois:
                roi_box = img_cv[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
                status = check_checkboxes_in_roi(roi_box)

                if status == "Checked":
                    collected = []
                    for page in document.pages:
                        for line in page.lines:
                            bbox_line = bbox_to_pixel(line.layout.bounding_poly,
                                                      img_width, img_height)
                            if inside_roi(bbox_line, roi):
                                line_text = "".join(
                                    [document.text[seg.start_index:seg.end_index]
                                     for seg in line.layout.text_anchor.text_segments]
                                ).strip()
                                if line_text:
                                    # Clean markers
                                    line_text = line_text.replace("x", "").replace("X", "")
                                    for sym in ["✓", "✔", "☑", "■"]:
                                        line_text = line_text.replace(sym, "")
                                    collected.append(line_text.strip())
                    if collected:
                        checked_values.append(" ".join(collected))
            roi_results[gname] = " | ".join(checked_values)

        else:
            collected = []
            for roi in group_rois:
                for page in document.pages:
                    for line in page.lines:
                        bbox_line = bbox_to_pixel(line.layout.bounding_poly,
                                                  img_width, img_height)
                        if inside_roi(bbox_line, roi):
                            line_text = "".join(
                                [document.text[seg.start_index:seg.end_index]
                                 for seg in line.layout.text_anchor.text_segments]
                            ).strip()
                            if not line_text:
                                continue
                            if roi_type == "number":
                                collected.extend(extract_numbers(line_text))
                            else:
                                collected.append(line_text)
            roi_results[gname] = " ".join(collected)

    return roi_results

# ==============================
# Process Multiple Images
# ==============================
def process_images_to_excel(files, rois: List[Dict],
                            mime_type="image/jpeg") -> pd.DataFrame:
    rows = []
    for file in files:
        image = Image.open(file).convert("RGB")
        masked_img = apply_roi_mask(image, rois)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            masked_img.save(tmpfile.name)
            tmp_path = tmpfile.name

        roi_outputs = process_with_documentai(tmp_path, mime_type, rois)

        row = {"image": file.name}
        for key, val in roi_outputs.items():
            row[key] = val
        rows.append(row)

        os.remove(tmp_path)

    return pd.DataFrame(rows)

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="ROI OCR App", layout="wide")
tabs = st.tabs(["📑 OCR Page", "⚙️ Config Page"])

# OCR Page
with tabs[0]:
    st.header("Run OCR on Images with ROI JSON")
    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    json_file = st.file_uploader("Upload ROI JSON", type=["json"])

    if files and json_file:
        if st.button("▶️ Run OCR"):
            with st.spinner("Processing images..."):
                rois = json.load(json_file)["boxes"]
                df = process_images_to_excel(files, rois)

            st.success("✅ OCR complete!")
            st.dataframe(df)

            output_path = tempfile.NamedTemporaryFile(suffix=".xlsx",
                                                      delete=False).name
            df.to_excel(output_path, index=False)

            with open(output_path, "rb") as f:
                st.download_button("📥 Download Excel", f,
                                   file_name="ocr_results.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Config Page
with tabs[1]:
    st.header("Configure ROI JSON + Run OCR")
    if "rois" not in st.session_state:
        st.session_state["rois"] = []

    config_files = st.file_uploader("Upload images (multiple allowed)",
                                    type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True)

    if config_files:
        first_img = Image.open(config_files[0]).convert("RGB")
        st.write("Draw ROIs on the first image 👇")
        canvas_result = st_canvas(
            background_image=first_img,
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="lime",
            update_streamlit=True,
            height=first_img.height,
            width=first_img.width,
            drawing_mode="rect",
            key="canvas_config",
        )

        if canvas_result.json_data:
            rois = []
            for i, obj in enumerate(canvas_result.json_data["objects"]):
                if obj["type"] == "rect":
                    x1, y1 = int(obj["left"]), int(obj["top"])
                    x2, y2 = int(obj["left"] + obj["width"]), int(obj["top"] + obj["height"])
                    rois.append({"id": f"ROI_{i+1}", "type": "text",
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "width": x2 - x1, "height": y2 - y1})
            st.session_state["rois"] = rois

        if st.session_state["rois"]:
            preview_img = first_img.copy()
            draw = ImageDraw.Draw(preview_img)
            for r in st.session_state["rois"]:
                draw.rectangle([(r["x1"], r["y1"]), (r["x2"], r["y2"])],
                               outline="red", width=3)
                draw.text((r["x1"], r["y1"] - 10), r["id"], fill="yellow")
            st.image(preview_img, caption="ROI Preview with IDs",
                     use_column_width=True)

            updated_rois = []
            st.subheader("ROI Settings")
            for r in st.session_state["rois"]:
                cols = st.columns([3, 2, 2])
                new_id = cols[0].text_input(f"Rename {r['id']}", value=r["id"], key=f"rename_{r['id']}")
                options = ["Text", "Number", "Checkbox"]
                current_type = r.get("type", "text").capitalize()
                if current_type not in options:
                    current_type = "Text"
                rtype = cols[1].selectbox("Type", options,
                                        index=options.index(current_type),
                                        key=f"type_{r['id']}").lower()
                group = cols[2].text_input("Group (optional)", value=r.get("group", ""), key=f"group_{r['id']}")

                updated_rois.append({**r, "id": new_id, "type": rtype, "group": group})


            if st.button("🗑 Undo Last ROI"):
                if st.session_state["rois"]:
                    st.session_state["rois"].pop()
                    st.experimental_rerun()

            if st.button("✅ Confirm and Run OCR"):
                payload = {"image_path": config_files[0].name,
                           "boxes": updated_rois}
                json_str = json.dumps(payload, ensure_ascii=False, indent=2)

                st.download_button("📥 Download ROI JSON", json_str,
                                   file_name="rect_coords.json",
                                   mime="application/json")

                with st.spinner("Processing all images with Document AI..."):
                    df = process_images_to_excel(config_files, updated_rois)

                st.success("✅ OCR complete!")
                st.dataframe(df)

                output_path = tempfile.NamedTemporaryFile(suffix=".xlsx",
                                                          delete=False).name
                df.to_excel(output_path, index=False)

                with open(output_path, "rb") as f:
                    st.download_button("📥 Download OCR Excel", f,
                                       file_name="ocr_results.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

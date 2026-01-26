from ultralytics import YOLO
import cv2
import os
from ocr_pipeline import run_ocr_on_image

# ---------------- CONFIG ----------------
MODEL_PATH = "../models/best.pt"
IMAGE_PATH = "../data/train/images/bus_video5_294.jpg"
OUTPUT_DIR = "../cropped_outputs"
CONF_THRESH = 0.25
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)
assert img is not None, f"Failed to load image: {IMAGE_PATH}"

results = model(img, conf=CONF_THRESH)

class_names = model.names

final_results = {
    "route_number": [],
    "destination": []
}

crop_id = 0

for r in results:
    boxes = r.boxes
    if boxes is None:
        continue

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = class_names[cls_id]

        crop = img[y1:y2, x1:x2]

        crop_filename = f"{cls_name}_{crop_id}_conf{conf:.2f}.jpg"
        crop_path = os.path.join(OUTPUT_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)

        print(f"[CROP SAVED] {crop_path}")

        # -------- OCR --------
        ocr_out = run_ocr_on_image(crop_path)
        print(f"[OCR] {cls_name}: {ocr_out['text']}")

        final_results[cls_name].append({
            #"crop": crop_filename,
            #"det_conf": conf,
            "ocr_text": ocr_out["text"],
            #"ocr_scores": ocr_out["scores"]
        })

        crop_id += 1

print("\n================ FINAL PIPELINE OUTPUT ================\n")
print(final_results)

import cv2
import os
import pyttsx3
from rapidfuzz import process, fuzz

from detect_and_crop import BusDetector
from ocr_pipeline import run_ocr_on_image


# ================= CONFIG =================
MODEL_PATH = "../models/best.pt"
IMAGE_PATH = "../data/test/bus_video6_321.jpg"
TEMP_DIR = "../temp_crops"
CONF_THRESH = 0.4

ROUTES_FILE = "../data/routes.txt"
LEVENSHTEIN_THRESHOLD = 40
# =========================================

os.makedirs(TEMP_DIR, exist_ok=True)


# ================= LOAD DESTINATIONS =================
with open(ROUTES_FILE, "r") as f:
    VALID_DESTINATIONS = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded {len(VALID_DESTINATIONS)} valid destinations")
# ====================================================


# ================= TTS SETUP =================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 125)
tts_engine.setProperty("volume", 1.0)
# ============================================


def correct_destination_levenshtein(raw_text):
    """
    Correct OCR destination using Levenshtein similarity.
    Returns (corrected_text, score)
    """
    if not raw_text:
        return raw_text, 0

    match, score, _ = process.extractOne(
        raw_text,
        VALID_DESTINATIONS,
        scorer=fuzz.ratio
    )

    if score >= LEVENSHTEIN_THRESHOLD:
        return match, score

    return raw_text, score


def speak(text):
    if text:
        tts_engine.say(text)
        tts_engine.runAndWait()


print("[INFO] Loading detector...")
detector = BusDetector(MODEL_PATH, CONF_THRESH)

print("[INFO] Loading image...")
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Failed to load image"

print("[INFO] Running detection...")
results = detector.detect(img)

print("[INFO] Cropping detections...")
crops = detector.crop_boxes(img, results, TEMP_DIR)

final_output = []


for i, bus in enumerate(crops["bus_front"]):
    bus_x1, bus_y1, bus_x2, bus_y2 = bus["bbox"]

    bus_routes = []
    bus_dests = []

    for r in crops["route_number"]:
        x1, y1, x2, y2 = r["bbox"]
        if bus_x1 <= x1 <= bus_x2:
            bus_routes.append(r)

    for d in crops["destination"]:
        x1, y1, x2, y2 = d["bbox"]
        if bus_x1 <= x1 <= bus_x2:
            bus_dests.append(d)

    route_raw = ""
    dest_raw = ""
    dest_corrected = ""
    dest_score = 0

    if bus_routes:
        route_raw = run_ocr_on_image(bus_routes[0]["path"])["text"]

    if bus_dests:
        dest_raw = run_ocr_on_image(bus_dests[0]["path"])["text"]
        dest_corrected, dest_score = correct_destination_levenshtein(dest_raw)

    final_output.append({
        "bus_id": i,
        "route": route_raw,
        "dest_raw": dest_raw,
        "dest_corrected": dest_corrected,
        "score": dest_score
    })


print("\n================ FINAL BUS OUTPUT ================\n")
for bus in final_output:
    print(f"BUS {bus['bus_id']}")
    print(f"ROUTE (RAW)        : {bus['route']}")
    print(f"DESTINATION (RAW)  : {bus['dest_raw']}")
    print(f"DESTINATION (FIX)  : {bus['dest_corrected']}")
    print(f"LEVENSHTEIN SCORE  : {bus['score']}")
    print("--------------------------------------------------")

    # Speak corrected output
    if bus["route"] and bus["dest_corrected"]:
        speak(f"Bus number {bus['route']} to {bus['dest_corrected']}")

import cv2
import os
import time
import threading
from collections import defaultdict, deque
from queue import Queue
from datetime import datetime

from ultralytics import YOLO
import pyttsx3
from rapidfuzz import process, fuzz

from ocr_pipeline import run_ocr_on_image


# ================= CONFIG =================
MODEL_PATH = "../models/best.pt"
VIDEO_SOURCE = "../data/videos/test_video.mp4"

CONF_THRESH = 0.4
TEMP_DIR = "../temp_crops"

TEMPORAL_WINDOW = 7
STABLE_THRESHOLD = 4

BUS_STALE_TIME = 10
LOG_FILE = "../bus_announcements.log"

ROUTES_FILE = "../data/routes.txt"
LEVENSHTEIN_THRESHOLD = 80   # similarity %
# =========================================

os.makedirs(TEMP_DIR, exist_ok=True)


# ================= LOAD ROUTES =================
with open(ROUTES_FILE, "r") as f:
    VALID_DESTINATIONS = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded {len(VALID_DESTINATIONS)} valid destinations")


# ================= TTS SETUP =================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 140)
tts_engine.setProperty("volume", 1.0)

announcement_queue = Queue()


def tts_worker():
    while True:
        text = announcement_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        announcement_queue.task_done()


tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
# ============================================


# ================= HELPERS =================
def majority_vote(seq):
    if not seq:
        return None, 0
    counts = {}
    for s in seq:
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts[best]


def correct_destination_levenshtein(raw_text):
    """
    Match OCR output against known destinations
    using Levenshtein similarity.
    """
    if not raw_text:
        return None

    match, score, _ = process.extractOne(
        raw_text,
        VALID_DESTINATIONS,
        scorer=fuzz.ratio
    )

    if score >= LEVENSHTEIN_THRESHOLD:
        return match
    return raw_text   # fallback to raw OCR


def build_announcement(route, dest):
    return f"Attention please. Bus number {route} to {dest} has arrived."


def log_announcement(bus_id, route, dest):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] BUS {bus_id} | Route {route} -> {dest}\n")
# ===========================================


print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
class_names = model.names

with open(LOG_FILE, "w") as f:
    f.write("=== BUS ANNOUNCEMENT LOG ===\n\n")

bus_memory = defaultdict(lambda: {
    "route_hist": deque(maxlen=TEMPORAL_WINDOW),
    "dest_hist": deque(maxlen=TEMPORAL_WINDOW),
    "announced": False,
    "last_seen": time.time()
})

print("[INFO] Starting video pipeline...")

results = model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    conf=CONF_THRESH
)

for r in results:
    frame = r.orig_img
    if r.boxes is None:
        continue

    bus_fronts, route_boxes, dest_boxes = [], [], []

    for box in r.boxes:
        cls_name = class_names[int(box.cls[0])]
        if cls_name == "bus_front":
            bus_fronts.append(box)
        elif cls_name == "route_number":
            route_boxes.append(box)
        elif cls_name == "destination":
            dest_boxes.append(box)

    for bus_box in bus_fronts:
        if bus_box.id is None:
            continue

        bus_id = int(bus_box.id[0])
        bus_memory[bus_id]["last_seen"] = time.time()

        bx1, _, bx2, _ = bus_box.xyxy[0].cpu().numpy().astype(int)

        matched_route, matched_dest = None, None

        for rb in route_boxes:
            x1, _, _, _ = rb.xyxy[0].cpu().numpy().astype(int)
            if bx1 <= x1 <= bx2:
                matched_route = rb
                break

        for db in dest_boxes:
            x1, _, _, _ = db.xyxy[0].cpu().numpy().astype(int)
            if bx1 <= x1 <= bx2:
                matched_dest = db
                break

        if matched_route:
            rx1, ry1, rx2, ry2 = matched_route.xyxy[0].cpu().numpy().astype(int)
            crop = frame[ry1:ry2, rx1:rx2]
            path = os.path.join(TEMP_DIR, f"bus{bus_id}_route.jpg")
            cv2.imwrite(path, crop)
            text = run_ocr_on_image(path)["text"]
            if text:
                bus_memory[bus_id]["route_hist"].append(text)

        if matched_dest:
            dx1, dy1, dx2, dy2 = matched_dest.xyxy[0].cpu().numpy().astype(int)
            crop = frame[dy1:dy2, dx1:dx2]
            path = os.path.join(TEMP_DIR, f"bus{bus_id}_dest.jpg")
            cv2.imwrite(path, crop)
            raw_dest = run_ocr_on_image(path)["text"]
            corrected_dest = correct_destination_levenshtein(raw_dest)
            if corrected_dest:
                bus_memory[bus_id]["dest_hist"].append(corrected_dest)

        route_final, route_count = majority_vote(bus_memory[bus_id]["route_hist"])
        dest_final, dest_count = majority_vote(bus_memory[bus_id]["dest_hist"])

        print(f"[BUS {bus_id}] ROUTE={route_final} ({route_count}) "
              f"DEST={dest_final} ({dest_count})")

        if (route_count >= STABLE_THRESHOLD and
            dest_count >= STABLE_THRESHOLD and
            not bus_memory[bus_id]["announced"]):

            announcement = build_announcement(route_final, dest_final)
            announcement_queue.put(announcement)
            log_announcement(bus_id, route_final, dest_final)
            bus_memory[bus_id]["announced"] = True

    now = time.time()
    for bid in list(bus_memory.keys()):
        if now - bus_memory[bid]["last_seen"] > BUS_STALE_TIME:
            del bus_memory[bid]

print("[INFO] Video finished")

announcement_queue.put(None)
tts_thread.join()

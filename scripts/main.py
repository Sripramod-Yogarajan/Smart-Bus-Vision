# ================= main.py =================
import cv2
import os
import time
import threading
from collections import defaultdict, deque
from queue import Queue
from datetime import datetime

import torch
from ultralytics import YOLO
from rapidfuzz import process, fuzz
from gtts import gTTS
import pygame

from ocr_pipeline import run_ocr_on_image

# ================= CONFIG =================
MODEL_PATH = "../models/best.pt"
VIDEO_SOURCE = "../data/data-test/test_video.mp4"

CONF_THRESH = 0.4
IMG_SIZE = 640

FRAME_SKIP = 3
TEMPORAL_WINDOW = 7
STABLE_THRESHOLD = 4

BUS_STALE_TIME = 10
COOLDOWN_TIME = 8

LOG_FILE = "bus_announcements.log"
ROUTES_FILE = "../data/routes.txt"
LEVENSHTEIN_THRESHOLD = 80
# =========================================

# ================= GPU INFO =================
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# ============================================

# ================= LOAD ROUTES =================
with open(ROUTES_FILE, "r") as f:
    VALID_DESTINATIONS = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded {len(VALID_DESTINATIONS)} valid destinations")

# ================= TTS SYSTEM =================
pygame.mixer.init()
announcement_queue = Queue()

def speak_text(text):
    try:
        filename = "temp_audio.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.stop()
        os.remove(filename)
        print("[AUDIO SPOKEN]")
    except Exception as e:
        print("[AUDIO ERROR]", e)

def speak_text(text):
    try:
        filename = f"temp_audio_{int(time.time())}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.stop()
        os.remove(filename)
        print("[AUDIO SPOKEN]")
    except Exception as e:
        print("[AUDIO ERROR]", e)

def tts_worker():
    while True:
        text = announcement_queue.get()
        if text is None:
            break
        print("[QUEUE RECEIVED]", text)  # Debugging line to ensure the queue is getting the announcement
        speak_text(text)
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

def correct_destination(raw_text):
    if not raw_text:
        return None
    match, score, _ = process.extractOne(raw_text, VALID_DESTINATIONS, scorer=fuzz.ratio)
    return match if score >= LEVENSHTEIN_THRESHOLD else raw_text

def build_announcement(route, dest):
    return f"Attention please. Bus number {route} to {dest} has arrived."

def log_announcement(bus_id, route, dest):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] BUS {bus_id} | Route {route} -> {dest}\n")
# ============================================

# ================= LOAD YOLO =================
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH).to("cuda")  # <--- GPU ENABLED
class_names = model.names

with open(LOG_FILE, "w") as f:
    f.write("=== BUS ANNOUNCEMENT LOG ===\n\n")
# ============================================

# ================= BUS MEMORY =================
bus_memory = defaultdict(lambda: {
    "route_hist": deque(maxlen=TEMPORAL_WINDOW),
    "dest_hist": deque(maxlen=TEMPORAL_WINDOW),
    "last_seen": time.time(),
    "last_route": None,
    "last_dest": None,
    "last_announcement_time": 0
})
# ============================================

print("[INFO] Starting video pipeline...")

results = model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    conf=CONF_THRESH,
    imgsz=IMG_SIZE
)

frame_count = 0

for r in results:
    frame_count += 1

    # -------- Frame Skipping (Speed Boost) --------
    if frame_count % FRAME_SKIP != 0:
        continue

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

        # ===== ROUTE OCR (NO DISK WRITE) =====
        if matched_route:
            rx1, ry1, rx2, ry2 = matched_route.xyxy[0].cpu().numpy().astype(int)
            crop = frame[ry1:ry2, rx1:rx2]
            text = run_ocr_on_image(crop)["text"]
            if text:
                bus_memory[bus_id]["route_hist"].append(text)

        # ===== DEST OCR =====
        if matched_dest:
            dx1, dy1, dx2, dy2 = matched_dest.xyxy[0].cpu().numpy().astype(int)
            crop = frame[dy1:dy2, dx1:dx2]
            raw_dest = run_ocr_on_image(crop)["text"]
            corrected = correct_destination(raw_dest)
            if corrected:
                bus_memory[bus_id]["dest_hist"].append(corrected)

        route_final, route_count = majority_vote(bus_memory[bus_id]["route_hist"])
        dest_final, dest_count = majority_vote(bus_memory[bus_id]["dest_hist"])

        print(f"[BUS {bus_id}] ROUTE={route_final} ({route_count}) "
              f"DEST={dest_final} ({dest_count})")

        current_time = time.time()

        if route_count >= STABLE_THRESHOLD and dest_count >= STABLE_THRESHOLD:
            if (bus_memory[bus_id]["last_route"] != route_final or
                bus_memory[bus_id]["last_dest"] != dest_final):
                if current_time - bus_memory[bus_id]["last_announcement_time"] > COOLDOWN_TIME:
                    announcement = build_announcement(route_final, dest_final)
                    print("[ANNOUNCING]", announcement)
                    announcement_queue.put(announcement)
                    log_announcement(bus_id, route_final, dest_final)
                    bus_memory[bus_id]["last_route"] = route_final
                    bus_memory[bus_id]["last_dest"] = dest_final
                    bus_memory[bus_id]["last_announcement_time"] = current_time

    # -------- Cleanup Stale Buses --------
    now = time.time()
    for bid in list(bus_memory.keys()):
        if now - bus_memory[bid]["last_seen"] > BUS_STALE_TIME:
            del bus_memory[bid]

print("[INFO] Video finished")

announcement_queue.put(None)
tts_thread.join()

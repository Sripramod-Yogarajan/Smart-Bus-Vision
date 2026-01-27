import cv2
import os
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from ocr_pipeline import run_ocr_on_image
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "../models/best.pt"
VIDEO_SOURCE = "../data/videos/test_video.mp4"

CONF_THRESH = 0.4
TEMP_DIR = "../temp_crops"

TEMPORAL_WINDOW = 7
STABLE_THRESHOLD = 4

LOG_FILE = "../bus_announcements.log"
# =========================================

os.makedirs(TEMP_DIR, exist_ok=True)

print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
class_names = model.names

# Clear old log file
with open(LOG_FILE, "w") as f:
    f.write("=== BUS ANNOUNCEMENT LOG ===\n\n")

# Per-bus memory (key = track_id from bus_front)
bus_memory = defaultdict(lambda: {
    "route_hist": deque(maxlen=TEMPORAL_WINDOW),
    "dest_hist": deque(maxlen=TEMPORAL_WINDOW),
    "announced": False,
    "last_seen": time.time()
})

def majority_vote(seq):
    if not seq:
        return None, 0
    counts = {}
    for s in seq:
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None, 0
    best = max(counts, key=counts.get)
    return best, counts[best]

def log_announcement(bus_id, route, dest):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] BUS {bus_id} | Route {route} -> {dest}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)

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

    # Separate detections
    bus_fronts = []
    route_boxes = []
    dest_boxes = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]

        if cls_name == "bus_front":
            bus_fronts.append(box)
        elif cls_name == "route_number":
            route_boxes.append(box)
        elif cls_name == "destination":
            dest_boxes.append(box)

    # Process each bus_front
    for bus_box in bus_fronts:
        if bus_box.id is None:
            continue

        bus_id = int(bus_box.id[0])
        bus_memory[bus_id]["last_seen"] = time.time()

        bx1, by1, bx2, by2 = bus_box.xyxy[0].cpu().numpy().astype(int)

        matched_route = None
        matched_dest = None

        # Find route inside bus_front
        for rbox in route_boxes:
            x1, y1, x2, y2 = rbox.xyxy[0].cpu().numpy().astype(int)
            if bx1 <= x1 <= bx2:
                matched_route = rbox
                break

        # Find destination inside bus_front
        for dbox in dest_boxes:
            x1, y1, x2, y2 = dbox.xyxy[0].cpu().numpy().astype(int)
            if bx1 <= x1 <= bx2:
                matched_dest = dbox
                break

        # OCR route
        if matched_route:
            rx1, ry1, rx2, ry2 = matched_route.xyxy[0].cpu().numpy().astype(int)
            r_crop = frame[ry1:ry2, rx1:rx2]
            r_path = os.path.join(TEMP_DIR, f"bus{bus_id}_route.jpg")
            cv2.imwrite(r_path, r_crop)
            route_text = run_ocr_on_image(r_path)["text"]
            if route_text:
                bus_memory[bus_id]["route_hist"].append(route_text)

        # OCR destination
        if matched_dest:
            dx1, dy1, dx2, dy2 = matched_dest.xyxy[0].cpu().numpy().astype(int)
            d_crop = frame[dy1:dy2, dx1:dx2]
            d_path = os.path.join(TEMP_DIR, f"bus{bus_id}_dest.jpg")
            cv2.imwrite(d_path, d_crop)
            dest_text = run_ocr_on_image(d_path)["text"]
            if dest_text:
                bus_memory[bus_id]["dest_hist"].append(dest_text)

        # Temporal fusion
        route_final, route_count = majority_vote(bus_memory[bus_id]["route_hist"])
        dest_final, dest_count = majority_vote(bus_memory[bus_id]["dest_hist"])

        print(f"[BUS {bus_id}] ROUTE={route_final} ({route_count}) "
              f"DEST={dest_final} ({dest_count})")

        # Announce ONCE + LOG
        if (route_count >= STABLE_THRESHOLD and
            dest_count >= STABLE_THRESHOLD and
            not bus_memory[bus_id]["announced"]):

            print("\n>>> NEW BUS ARRIVED <<<")
            print(f"ANNOUNCE: Route {route_final} to {dest_final}")
            print("========================================\n")

            log_announcement(bus_id, route_final, dest_final)
            bus_memory[bus_id]["announced"] = True

    # Cleanup old buses
    now = time.time()
    stale_ids = [
        bid for bid, data in bus_memory.items()
        if now - data["last_seen"] > 10
    ]
    for bid in stale_ids:
        print(f"[INFO] Bus {bid} left scene")
        del bus_memory[bid]

print(f"\n[INFO] Video finished.")
print(f"[INFO] Announcements saved to: {LOG_FILE}")

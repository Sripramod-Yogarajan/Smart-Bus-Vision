# ocr_pipeline.py
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def preprocess_for_led(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return enhanced

def run_ocr_on_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"text": ""}

    pre = preprocess_for_led(img)
    results = reader.readtext(pre)

    texts = []
    for _, text, score in results:
        if score > 0.3:
            texts.append(text)

    final_text = " ".join(texts).strip()

    return {"text": final_text}

import cv2
import easyocr

# Initialize once (heavy model)
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA

def preprocess_for_led(img):
    """
    Preprocess for LED boards to improve OCR
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Resize up for better OCR
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return enhanced

def run_ocr_on_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    pre = preprocess_for_led(img)

    # EasyOCR
    results = reader.readtext(pre)

    texts = []
    scores = []

    for bbox, text, score in results:
        texts.append(text)
        scores.append(score)

    final_text = " ".join(texts).strip()

    return {
        "text": final_text,
        "texts": texts,
        "scores": scores
    }



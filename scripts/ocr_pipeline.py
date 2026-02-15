# ocr_pipeline.py
import cv2
import easyocr

# Enable GPU for EasyOCR if your RTX 3070 Ti is available
reader = easyocr.Reader(['en'], gpu=True)

def preprocess_for_led(img):
    """
    Enhance LED-style images for OCR
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Upscale for better OCR
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return enhanced

def run_ocr_on_image(img_or_path):
    """
    Accepts either:
      - a numpy array (cv2 image)
      - a file path string
    Returns recognized text
    """
    # Load image if a path is provided
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            return {"text": ""}
    else:
        img = img_or_path

    pre = preprocess_for_led(img)
    results = reader.readtext(pre)

    texts = [text for _, text, score in results if score > 0.3]
    final_text = " ".join(texts).strip()

    return {"text": final_text}

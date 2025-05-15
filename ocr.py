import pytesseract
import cv2
import numpy as np
import re

def read_text(cropped):
    cropped = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 45, 15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    text = pytesseract.image_to_string(processed, config=config)

    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    return text.strip()
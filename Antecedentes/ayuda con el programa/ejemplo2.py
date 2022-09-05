from PIL import Image
import re
import cv2
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = cv2.imread(r"C:\Users\joe\invoice-sample.jpg")
d = pytesseract.image_to_data(img, output_type=Output.DICT)
keys = list(d.keys())

date_pattern = "Invoice"

n_boxes = len(d["text"])
for i in range(n_boxes):
    if float(d["conf"][i]) > 60:
    	if re.match(date_pattern, d["text"][i]):
	        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
	        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
import cv2

def normalize_text(text):
    return text.lower().strip()

def normalize_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
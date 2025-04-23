import cv2

def normalize_text(text):
    return text.lower().strip()

def normalize_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    return img
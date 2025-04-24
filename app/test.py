import cv2
import time
import base64
import requests

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return "data:image/jpeg;base64," + base64_image

def call_face_recognition_api(base64_img):
    url = "http://192.168.30.210:8000/api/v1/face/recognize"
    payload = {"image": base64_img}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("predictions", [])
        else:
            print(f"Lỗi từ API: {response.status_code}")
            return []
    except Exception as e:
        print(f"Lỗi gọi API: {e}")
        return []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

t0 = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera")
        break
    
    t1 = time.time()
    if t1 - t0 > 3:
        base64_img = encode_frame_to_base64(frame)
        predictions = call_face_recognition_api(base64_img)
        t0 = t1

    cv2.imshow('Nhan dien khuon mat', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()


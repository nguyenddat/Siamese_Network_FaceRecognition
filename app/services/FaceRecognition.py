from ai.FaceRecognition.FaceRecogntion import FaceRecognition
from helpers.DataLoader import data_loader
from helpers.ImageLoader import load_image

face_recognition = FaceRecognition()

def re_train():
    data_loader.load_data()
    x_a, x_b, y = data_loader.preprocess_data_for_training(data_loader.x_for_training, data_loader.y_for_training)
    face_recognition.backup(x_a, x_b, y)


def predict(img):
    img, _ = load_image(img)
    faces = face_recognition.embed_for_prediction(img)
    predictions = face_recognition.predict(faces, data_loader.x_for_training, data_loader.y_for_training)
    return predictions

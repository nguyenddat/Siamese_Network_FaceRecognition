from ai.FaceRecognition.FaceRecogntion import FaceRecognition
from helpers.DataLoader import data_loader
from helpers.ImageLoader import load_image

face_recognition = FaceRecognition()

def re_train():
    data_loader.load_data()
    x_a, x_b, y = data_loader.preprocess_data_for_training(data_loader.x_for_training, data_loader.y_for_training)
    face_recognition.train(x_a, x_b, y)


def predict(img):
    img, _ = load_image(img)
    x_min, x_max = img.shape[1] // 2, img.shape[1] * 3 // 4
    y_min, y_max = 0, img.shape[0]

    faces = face_recognition.embed_for_prediction(img)
    predictions = face_recognition.predict(
        [face.embedding for face in faces], 
        data_loader.x_for_training, 
        data_loader.y_for_training
    )

    resp_objs = {}
    main_faces = []
    others = []
    for face, prediction in zip(faces, predictions):
        is_main_face, area = check_main_face(face.bbox, (x_min, x_max, y_min, y_max))
        if is_main_face:
            main_faces.append((prediction, area))
        else:
            others.append(prediction)

    main_faces.sort(key=lambda x: x[1], reverse=True)
    if len(main_faces) > 0:
        resp_objs['main_face'] = main_faces[0][0]
        resp_objs['others'] = others
        resp_objs['others'] += [main_face[0] for main_face in main_faces[1:]]
    else:
        resp_objs['others'] = others

    return resp_objs


def check_main_face(bbox, img):
    x, y, w, h = bbox
    x_min, x_max, y_min, y_max = img

    if x + (w // 2) > x_max or x + (w // 2) < x_min or y + (h // 2) > y_max or y + (h // 2) < y_min:
        return False, 0
    
    else:
        return True, w * h
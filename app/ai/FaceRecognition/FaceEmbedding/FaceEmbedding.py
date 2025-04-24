import numpy as np
from insightface.app import FaceAnalysis

class FaceEmbedding:
    def __init__(self):
        self.model = FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=0)

    def embed_for_training(self, img):            
        faces = self.model.get(img)
        if len(faces) == 0:
            return np.zeros((512,))
        else:
            return faces[0].embedding

    def embed_for_prediction(self, img):
        faces = self.model.get(img)
        return faces

face_embedding = FaceEmbedding()
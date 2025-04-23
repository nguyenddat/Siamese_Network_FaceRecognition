import os
import glob
from datetime import datetime
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ai.FaceRecognition.FaceEmbedding.FaceEmbedding import face_embedding
from ai.FaceRecognition.SiameseNetwork.SiameseNetwork import SiameseNetwork
from ai.FaceRecognition.DecisionTree.DecisionTree import threshold_optimize
from core.Settings import settings
from helpers.DataLoader import data_loader

class FaceRecognition:
    def __init__(self):
        self.embedding = face_embedding
        self._load_model()
            
    def predict(self, imgs, X, y):
        resp_objs = []
        for img in imgs:
            img_req = np.vstack([img] * len(X))
            distances_pred = self.model.predict([img_req, X], batch_size=32, verbose=0)
            distances = distances_pred.flatten().reshape(-1, 1)

            sorted_idx = np.argsort(distances, axis=0)[:3].flatten()
            sorted_distances = distances[sorted_idx].flatten()
            sorted_labels = [y[i] for i in sorted_idx]
            print(self.threshold)
            print(sorted_distances)

            matched_idx = np.where(sorted_distances < self.threshold)[0]
            matched_labels = [sorted_labels[i] for i in matched_idx]

            if len(matched_labels) == 0:
                resp_objs.append({"name": "Khách"})
            
            else:
                label_counts = Counter(matched_labels)
                most_common_label = max(label_counts.items(), key=lambda x: (x[1], -matched_labels.index(x[0])))

                resp_objs.append({"name": most_common_label[0]})

        return resp_objs


    def embed_for_prediction(self, img):
        return self.embedding.embed_for_prediction(img)
    

    def embed_for_training(self, img):
        return self.embedding.embed_for_training(img)


    def train(self):
        # Load dữ liệu từ DataLoader
        x_a, x_b, y = data_loader.preprocess_data_for_training()
        x_train_a, x_test_a, x_train_b, x_test_b, y_train, y_test = train_test_split(x_a, x_b, y, test_size=0.2, random_state=42)

        # Train mô hình backup
        backup_model = SiameseNetwork()
        backup_model.fit(x_train_a, x_train_b, y_train)

        # Tính threshold cho mô hình backup
        distances_pred = backup_model.model.predict([x_test_a, x_test_b], batch_size=32)
        distances = distances_pred.flatten().reshape(-1, 1)
        y_test = np.array(y_test)
        
        threshold = threshold_optimize(distances, y_test) * (7.7/10)

        # Tạo tên file có timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = os.path.join(settings.FACE_RECOGNITION_WEIGHTS_PATH, f"{timestamp}_siamese_model.keras")
        threshold_path = os.path.join(settings.FACE_RECOGNITION_WEIGHTS_PATH, f"{timestamp}_siamese_threshold.npz")

        # Lưu trọng số và load lại
        backup_model.model.save(weights_path)
        np.savez_compressed(threshold_path, threshold=threshold)
        print(f"Weights saved to {weights_path}.")
        print(f"Threshold saved to {threshold_path}.")
        
        # Load lại trọng số và threshold vào mô hình chính
        self.model = tf.keras.models.load_model(weights_path)
        self.threshold = np.load(threshold_path)["threshold"]
        print(f"Weights saved and loaded into self.model from {weights_path}.")
        return self


    def _load_model(self):
        weight_files = glob.glob(os.path.join(settings.FACE_RECOGNITION_WEIGHTS_PATH, "*_siamese_model.keras"))
        threshold_files = glob.glob(os.path.join(settings.FACE_RECOGNITION_WEIGHTS_PATH, "*_siamese_threshold.npz"))

        if weight_files and threshold_files:
            latest_threshold = max(threshold_files, key=os.path.getctime)
            threshold_data = np.load(latest_threshold)
            self.threshold = threshold_data['threshold']
            print(f"Threshold loaded from {latest_threshold}.")

            latest_weights = max(weight_files, key=os.path.getctime)
            self.model = tf.keras.models.load_model(latest_weights)
            print(f"Weights loaded from {latest_weights}.")

        else:
            print("❌ No weights or threshold files found. Please train the model first.")
            self.train()
        
        return self
import os
import random
import itertools
from collections import defaultdict

import cv2
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from helpers.FaceRecognitionHelpers import normalize_img, normalize_text
from ai.FaceRecognition.FaceEmbedding.FaceEmbedding import face_embedding

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_saved_data()

    def load_saved_data(self):
        saved_file = os.path.join(self.data_path, "saved_data.pkl")
        if os.path.exists(saved_file):
            with open(saved_file, "rb") as f:
                data = pickle.load(f)

            self.x_for_training = data["x_for_training"]
            self.y_for_training = data["y_for_training"]
            return self
        
        else:
            self.load_data()
        

    def load_data(self):
        labels = list(os.scandir(self.data_path))

        x_for_training = []
        y_for_training = []

        for label in tqdm(labels, desc = "Loadin data for training..."):
            imgs = os.listdir(label.path)

            for i, img in enumerate(imgs):
                img = cv2.imread(os.path.join(label.path, img))
                img = normalize_img(img)

                x_for_training.append(face_embedding.embed_for_training(img))
                y_for_training.append(normalize_text(label.name))

        x_for_training = np.stack(x_for_training)

        self.x_for_training = x_for_training
        self.y_for_training = y_for_training
        self.save_data()
        return self

    def save_data(self):
        saved_file = os.path.join(self.data_path, "saved_data.pkl")
        data = {
            "x_for_training": self.x_for_training,
            "y_for_training": self.y_for_training,
        }
        with open(saved_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {saved_file}.")

    def preprocess_data_for_training(self):
        encoder = LabelEncoder()
        y_encoded = np.array(self.y_for_training)
        y_encoded = encoder.fit_transform(y_encoded)

        class_to_indices = defaultdict(list)
        for idx, y_enc in enumerate(y_encoded):
            class_to_indices[y_enc].append(idx)

        x_a, x_b, labels = [], [], []
        
        # Positive pairs
        for label, indices in class_to_indices.items():
            if len(indices) >= 2:
                pos_combs = list(itertools.combinations(indices, 2))
                for pos_comb in pos_combs:
                    x_a.append(self.x_for_training[pos_comb[0]])
                    x_b.append(self.x_for_training[pos_comb[1]])
                    labels.append(1)
        
        pos_count = len(labels)
        
        # Negative pairs with improved quality
        # 1. Pre-compute feature similarities to find harder negatives
        features = np.array(self.x_for_training)
        # Normalize features for cosine similarity
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Generate negative pairs (same quantity as positive pairs)
        neg_pairs_added = 0
        labels_list = list(class_to_indices.keys())
        
        # Try to find semi-hard negatives first
        for label1 in labels_list:
            for label2 in labels_list:
                if label1 != label2:
                    indices1 = class_to_indices[label1]
                    indices2 = class_to_indices[label2]
                    
                    for i in indices1:
                        if neg_pairs_added >= pos_count:
                            break
                        
                        # Compute similarities with samples from other class
                        sims = np.dot(features_norm[i:i+1], features_norm[indices2].T)[0]
                        
                        # Sort indices by similarity (highest first - hardest negatives)
                        sorted_indices = np.argsort(-sims)
                        
                        # Pick semi-hard negatives (not the hardest, not the easiest)
                        # Take from 25-75% range of similarity distribution
                        semi_hard_idx = len(sorted_indices) // 4
                        if semi_hard_idx < len(sorted_indices):
                            j_idx = sorted_indices[semi_hard_idx]
                            j = indices2[j_idx]
                            
                            x_a.append(self.x_for_training[i])
                            x_b.append(self.x_for_training[j])
                            labels.append(0)
                            neg_pairs_added += 1
                    
                    if neg_pairs_added >= pos_count:
                        break
                
                if neg_pairs_added >= pos_count:
                    break
        
        # If we still need more negative samples, fill with random negatives
        while neg_pairs_added < pos_count:
            label1, label2 = random.sample(labels_list, 2)
            i = random.choice(class_to_indices[label1])
            j = random.choice(class_to_indices[label2])
            
            # Kiểm tra xem cặp này đã được thêm chưa (tránh trùng lặp)
            if (i, j) not in [(x_a[k], x_b[k]) for k in range(len(x_a)) if labels[k] == 0]:
                x_a.append(self.x_for_training[i])
                x_b.append(self.x_for_training[j])
                labels.append(0)
                neg_pairs_added += 1
        
        x_a = np.array(x_a)
        x_b = np.array(x_b)
        y = np.array(labels)
        return x_a, x_b, y    

data_path = os.path.join(os.getcwd(), "app", "data")
data_loader = DataLoader(data_path)

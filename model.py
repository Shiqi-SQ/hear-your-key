import os
import pickle
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
class KeyboardModel:
    def __init__(self, model_path='data/keyboard_model.pkl'):
        self.model = None
        self.features = defaultdict(list)
        self.model_path = model_path
        self.is_trained = False
        self.load_model()
    def add_sample(self, key, features):
        self.features[key].append(features)
    def get_sample_count(self):
        return {k: len(v) for k, v in self.features.items()}
    def _prepare_data(self):
        X = []
        y = []
        for key, feature_list in self.features.items():
            for features in feature_list:
                feature_vector = self._features_to_vector(features)
                X.append(feature_vector)
                y.append(key)
        return np.array(X), np.array(y)
    def _features_to_vector(self, features):
        feature_vector = []
        feature_vector.extend(features['mfcc'])
        feature_vector.append(features['spectral_centroid'])
        feature_vector.append(features['spectral_bandwidth'])
        feature_vector.append(features['spectral_rolloff'])
        feature_vector.append(features['zero_crossing_rate'])
        feature_vector.append(features['rms'])
        return feature_vector
    def train(self):
        if not self.features:
            return False
        X, y = self._prepare_data()
        if len(np.unique(y)) < 2:
            return False
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        self.save_model()
        return True
    def predict(self, features):
        if not self.is_trained or not self.model:
            return None, 0
        feature_vector = self._features_to_vector(features)
        feature_vector = np.array(feature_vector).reshape(1, -1)
        prediction = self.model.predict(feature_vector)[0]
        confidence = np.max(self.model.predict_proba(feature_vector))
        return prediction, confidence
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'features': dict(self.features),
                'is_trained': self.is_trained
            }, f)
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.features = defaultdict(list, data['features'])
                self.is_trained = data['is_trained']
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
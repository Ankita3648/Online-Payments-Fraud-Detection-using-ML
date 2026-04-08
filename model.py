import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

def train_model():
    X, y = load_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return train_model()

def predict(model, data):
    return model.predict([data])[0]

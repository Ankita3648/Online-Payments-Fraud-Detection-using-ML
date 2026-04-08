import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "creditcard.csv")

def load_data():
    data = pd.read_csv(DATA_PATH)

    # Basic cleaning
    data = data.drop_duplicates()

    # Scale Amount
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])

    X = data.drop("Class", axis=1)
    y = data["Class"]

    return X, y

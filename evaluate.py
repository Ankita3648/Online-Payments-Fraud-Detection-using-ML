from sklearn.metrics import classification_report
from model import load_model
from preprocess import load_data

X, y = load_data()
model = load_model()

y_pred = model.predict(X)

print("\nModel Performance:\n")
print(classification_report(y, y_pred))

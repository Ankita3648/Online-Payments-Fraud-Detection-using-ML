from flask import Flask, render_template, request
from model import load_model, predict

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_result():
    try:
        # Expect 30 features; pad if fewer provided
        values = [float(x) for x in request.form.values() if x.strip()!='']
        if len(values) < 30:
            values = values + [0.0]*(30-len(values))
        else:
            values = values[:30]

        result = predict(model, values)

        if result == 1:
            output = "⚠️ Fraud Transaction Detected"
        else:
            output = "✅ Legitimate Transaction"

        return render_template('index.html', prediction_text=output)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

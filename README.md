# Online Payments Fraud Detection (Flask + ML)

## Steps
1. Put dataset at: dataset/creditcard.csv
2. Install deps: pip install -r requirements.txt
3. Run: python src/app.py
4. Open: http://127.0.0.1:5000

## Notes
- Model auto-trains on first run and saves to model.pkl
- Expects 30 features (V1..V28, Time, Amount)

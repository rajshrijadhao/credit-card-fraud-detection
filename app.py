
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model and scalers
model = joblib.load("model.pkl")
time_scaler = joblib.load("time_scaler.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "Please upload a CSV file."

    file = request.files["file"]

    if file.filename == "":
        return "No file selected."

    if not file.filename.endswith(".csv"):
        return "Please upload a CSV file."

    data = pd.read_csv(file)

    required_columns = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "V7", "V8", "V9", "V10", "V11", "V12",
    "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28",
    "Amount"
]

    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        return "Invalid dataset. Required columns are missing."

    X = data[required_columns].copy()

    X["Time"] = time_scaler.transform(X[["Time"]]).ravel()
    X["Amount"] = amount_scaler.transform(X[["Amount"]]).ravel()

    predictions = model.predict(X)

    data["Prediction"] = predictions
    data["Prediction"] = data["Prediction"].map({
        0: "Genuine",
        1: "Fraud"
    })

    total = len(data)
    fraud_count = int((predictions == 1).sum())
    genuine_count = int((predictions == 0).sum())

    return render_template(
        "result.html",
        total=total,
        fraud=fraud_count,
        genuine=genuine_count,
        tables=data[["Time", "Amount", "Prediction"]].head(100).to_html(
            classes="table",
            index=False
        )
    )


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=True)

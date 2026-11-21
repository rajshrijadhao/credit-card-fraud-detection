import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Separate scalers
time_scaler = StandardScaler()
amount_scaler = StandardScaler()

X_train["Time"] = time_scaler.fit_transform(X_train[["Time"]]).ravel()
X_train["Amount"] = amount_scaler.fit_transform(X_train[["Amount"]]).ravel()

X_test["Time"] = time_scaler.transform(X_test[["Time"]]).ravel()
X_test["Amount"] = amount_scaler.transform(X_test[["Amount"]]).ravel()

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Training Completed!")
print(classification_report(y_test, predictions))

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(time_scaler, "time_scaler.pkl")
joblib.dump(amount_scaler, "amount_scaler.pkl")

print("All model files saved successfully!")


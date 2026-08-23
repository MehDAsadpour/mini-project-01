import json
import joblib
import pandas as pd


MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD = 0.3


def load_model():
    return joblib.load(MODEL_PATH)


def load_scaler():
    return joblib.load(SCALER_PATH)


def load_input(path):
    with open(path, "r") as file:
        data = json.load(file)

    return pd.DataFrame([data])


def predict(model,scaler, X):
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[:, 1][0]

    class_id = int(probability >= THRESHOLD)

    if class_id == 1:
        prediction = "Fraud"
    else:
        prediction = "Normal"
        probability = 1 - probability

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability": probability,
        "threshold": THRESHOLD,
        "status": "success"
    }

def save_output(result, path):
    with open(path, "w") as file:
        json.dump(result, file, indent=4)


def main():
    model = load_model()

    scaler = load_scaler()

    X = load_input("examples/input.json")

    result = predict(model,scaler, X)

    save_output(result, "examples/output.json")

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
from pathlib import Path

import joblib


MODEL_PATH = Path("models/iris_model.pkl")

CLASS_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


def predict(sample):
    model = joblib.load(MODEL_PATH)

    prediction = model.predict([sample])

    return CLASS_NAMES[prediction[0]]


def main():
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict(sample)

    print(f"Predicted class: {result}")


if __name__ == "__main__":
    main()
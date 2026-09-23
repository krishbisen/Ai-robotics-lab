from pathlib import Path

import joblib


MODEL_PATH = Path("models/iris_model.pkl")

CLASS_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]

def main():
    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # New flower measurement
    sample = [[5.1, 3.5, 1.4, 0.2]]

    # Make prediction
    prediction = model.predict(sample)
    predicted_class = CLASS_NAMES[prediction[0]]


    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
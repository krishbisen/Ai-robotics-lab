from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ai_robotics.ml.evaluation import evaluate_classification


CONFIG_PATH = Path("configs/ml_config.json")


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)

    
def main():

    config = load_config()

    MODEL_PATH = Path(config["model_path"])


    # 1. Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    # 3. Create model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=config["max_iter"]))
    ])

    # 4. Train
    model.fit(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_classification(
        model,
        X_test,
        y_test,
    )
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


    # 6. Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
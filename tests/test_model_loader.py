from sklearn.linear_model import LogisticRegression

from ai_robotics.ml.model_loader import save_model, load_model


def test_save_and_load_model(tmp_path):
    model = LogisticRegression()
    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    path = tmp_path / "model.pkl"

    save_model(model, path)
    loaded_model = load_model(path)

    assert loaded_model.predict([[0]])[0] == model.predict([[0]])[0]
    assert loaded_model.predict([[3]])[0] == model.predict([[3]])[0]
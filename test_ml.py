
# TODO: add necessary import
import numpy as np
from ml.model import compute_model_metrics, inference, train_model
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics_perfect_predictions():
    """
    # When predictions exactly match the true labels, precision, recall, and the F1 score should be 1.
    """
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 1, 0, 1])
    p, r, fb = compute_model_metrics(y, preds)
    assert fb == 1
    assert p == 1
    assert r == 1


def make_fake_data():
    """Create a small, random dataset for testing."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 5))
    y = np.array([0,1] * 10)
    return X, y

# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_random_forest():
    """
    train_model should return a RandomForestClassifier.
    """
    X, y = make_fake_data()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)



# TODO: implement the third test. Change the function name and input as needed
def test_inference_returns_one_prediction_per_row():
    """
    inference should return a prediction for every input row, and each prediction should be a 0 or 1.
    """
    X, y = make_fake_data()
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

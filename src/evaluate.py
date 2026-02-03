from typing import Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": acc,
        "report": report,
    }


def pick_best(results: Dict[str, Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
    best_name = max(results, key=lambda k: results[k]["accuracy"])  # type: ignore
    return best_name, results[best_name]

from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return tuple(train_test_split(X, y, test_size=test_size, random_state=random_state))


def get_models(random_state: int) -> Dict[str, object]:
    """Return a dict of model name -> model instance ** with hyperparamaters tuning."""
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=100),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", random_state=random_state),
        "Naive Bayes": GaussianNB(),
    }


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)
    return model

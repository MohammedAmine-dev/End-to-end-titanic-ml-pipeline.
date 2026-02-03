import pandas as pd


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split features/target.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

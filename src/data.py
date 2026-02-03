from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

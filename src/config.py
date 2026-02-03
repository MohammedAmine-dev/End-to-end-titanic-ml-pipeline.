from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class PipelineConfig:
    train_path: Path = DATA_DIR / "train_engineered.csv"
    test_path: Path | None = None
    target_col: str = "Survived"
    test_size: float = 0.2
    random_state: int = 42

    # Model selection flags
    use_hyperparameter_tuning: bool = False

    # Save artifacts
    save_model: bool = True
    model_output_path: Path = RESULTS_DIR / "titanic_best_model.pkl"

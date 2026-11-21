from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"source", "target", "interaction_type"}


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load and validate one network's interaction records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Interaction table is empty.")

    frame = frame.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or (frame[column] == "").any():
            raise ValueError(f"Column '{column}' contains missing or blank values.")

    return frame.reset_index(drop=True)

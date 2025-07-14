import joblib
from pathlib import Path
from api.domain.entities import ModelArtifacts

ARTIFACTS_PATH = Path("training/artifacts")

def load_artifacts() -> ModelArtifacts:
    """
    Loads the model artifacts from the artifacts directory.
    """
    try:
        return {
            'model': joblib.load(ARTIFACTS_PATH / "model.cbm"),
            'scaler': joblib.load(ARTIFACTS_PATH / "scaler.pkl"),
            'preset_risk': joblib.load(ARTIFACTS_PATH / "preset_risk.pkl"),
            'features': joblib.load(ARTIFACTS_PATH / "features.pkl")
        }
    except FileNotFoundError as e:
        raise RuntimeError("Model artifacts not found") from e

def get_model_artifacts() -> ModelArtifacts:
    """
    Returns the model artifacts.
    """
    if not hasattr(get_model_artifacts, 'artifacts'):
        get_model_artifacts.artifacts = load_artifacts()
    return get_model_artifacts.artifacts
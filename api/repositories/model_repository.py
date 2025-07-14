import joblib
from pathlib import Path
from typing import Dict, Any

ARTIFACTS_PATH = Path("training/artifacts")

class ModelRepository:
    def __init__(self):
        self.artifacts = None
        
    def load_artifacts(self) -> Dict[str, Any]:
        """
        Loads the model artifacts from the artifacts directory.
        """
        if self.artifacts is None:
            try:
                self.artifacts = {
                    'model': joblib.load(ARTIFACTS_PATH / "model.cbm"),
                    'scaler': joblib.load(ARTIFACTS_PATH / "scaler.pkl"),
                    'metadata': joblib.load(ARTIFACTS_PATH / "metadata.pkl")
                }
            except FileNotFoundError as e:
                raise RuntimeError("Model artifacts not found. Train model first") from e
        return self.artifacts
    
    def save_artifacts(self, artifacts: Dict[str, Any]):
        """
        Saves the model artifacts to the artifacts directory.
        """
        ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts['model'], ARTIFACTS_PATH / "model.cbm")
        joblib.dump(artifacts['scaler'], ARTIFACTS_PATH / "scaler.pkl")
        joblib.dump(artifacts['metadata'], ARTIFACTS_PATH / "metadata.pkl")
        self.artifacts = artifacts
import pandas as pd
import numpy as np
from api.domain.entities import EquipmentData, PredictionResult, RiskLevel
from api.utils.feature_engineering import FeatureEngineer
from api.utils.sequence_utils import SequenceProcessor
from api.repositories.model_repository import ModelRepository

class InferenceService:
    def __init__(self, model_repo: ModelRepository):
        self.model_repo = model_repo
        self.feature_engineer = FeatureEngineer()
        self.sequence_processor = SequenceProcessor()
        
    def predict_single(self, input_data: EquipmentData) -> PredictionResult:
        """
        Returns the prediction result for a single equipment data.
        """
        artifacts = self.model_repo.load_artifacts()
        metadata = artifacts['metadata']
        required_features = metadata['features']
        
        input_df = pd.DataFrame([input_data.dict()])
        input_df['Fail'] = 0
        
        processed_df = self.feature_engineer.add_features(input_df)
        
        processed_df['preset_risk'] = processed_df.apply(
            lambda row: metadata['preset_risk'].get(
                (row['Preset_1'], row['Preset_2']), 0
            ), axis=1
        )
        
        for feature in required_features:
            if feature not in processed_df.columns:
                processed_df[feature] = 0
        
        processed_df = processed_df[required_features]
        
        scaled_values = artifacts['scaler'].transform(processed_df)
        processed_df[required_features] = scaled_values
        
        sequence = np.tile(
            processed_df[required_features].values,
            (metadata['window_size'], 1)
        )
        
        stats = np.concatenate([
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.max(axis=0),
            sequence.min(axis=0),
            np.diff(sequence, axis=0).mean(axis=0)
        ]).reshape(1, -1)
        
        proba = artifacts['model'].predict_proba(stats)[0, 1]
        
        risk_level = self._determine_risk_level(proba)
        
        return PredictionResult(
            failure_probability=proba,
            risk_level=risk_level
        )
    
    def _determine_risk_level(self, probability: float) -> RiskLevel:
        """
        Returns the risk level based on the probability.
        """
        if probability < 0.3:
            return RiskLevel.LOW
        elif probability < 0.7:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH
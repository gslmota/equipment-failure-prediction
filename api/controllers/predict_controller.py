from fastapi import Depends
from api.services.inference_service import InferenceService
from api.domain.entities import EquipmentData, PredictionResult
from api.repositories.model_repository import ModelRepository

class PredictController:
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        
    def predict(self, input_data: EquipmentData) -> PredictionResult:
        """
        Returns the prediction result for a single equipment data.
        """
        return self.inference_service.predict_single(input_data)
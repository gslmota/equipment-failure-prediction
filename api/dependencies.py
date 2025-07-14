from api.repositories.model_repository import ModelRepository
from api.services.inference_service import InferenceService
from api.services.training_service import TrainingService
from api.controllers.predict_controller import PredictController
from api.controllers.train_controller import TrainController
from api.controllers.health_controller import HealthController
from fastapi import Depends

def get_model_repository() -> ModelRepository:
    """
    Returns the model repository.
    """
    return ModelRepository()

def get_inference_service(repo: ModelRepository = Depends(get_model_repository)) -> InferenceService:
    """
    Returns the inference service.
    """
    return InferenceService(repo)

def get_training_service(repo: ModelRepository = Depends(get_model_repository)) -> TrainingService:
    """
    Returns the training service.
    """
    return TrainingService(repo)

def get_predict_controller(service: InferenceService = Depends(get_inference_service)) -> PredictController:
    """
    Returns the predict controller.
    """
    return PredictController(service)

def get_train_controller(service: TrainingService = Depends(get_training_service)) -> TrainController:
    """
    Returns the train controller.
    """
    return TrainController(service)

def get_health_controller() -> HealthController:
    """
    Returns the health controller.
    """
    return HealthController()
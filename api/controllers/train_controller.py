from fastapi import BackgroundTasks
from api.services.training_service import TrainingService
from api.domain.entities import TrainingParameters, TrainingResult

class TrainController:
    def __init__(self, training_service: TrainingService):
        self.training_service = training_service
        
    def train(self, params: TrainingParameters, background_tasks: BackgroundTasks) -> TrainingResult:
        """
        Trains a new model in the background.
        """
        background_tasks.add_task(
            self.training_service.train_model,
            params
        )
        
        return TrainingResult(
            status="started",
            metrics={},
            message="Training process started in background"
        )
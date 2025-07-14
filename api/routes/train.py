from fastapi import APIRouter, BackgroundTasks, Depends
from api.controllers.train_controller import TrainController
from api.dependencies import get_train_controller
from api.domain.entities import TrainingParameters, TrainingResult

router = APIRouter()

@router.post("/", response_model=TrainingResult)
def train_model(
    params: TrainingParameters,
    background_tasks: BackgroundTasks,
    controller: TrainController = Depends(get_train_controller)
):
    """
    Trains a new model in the background.
    """
    return controller.train(params, background_tasks)
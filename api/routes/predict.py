from fastapi import APIRouter, Depends
from api.controllers.predict_controller import PredictController
from api.dependencies import get_predict_controller
from api.domain.entities import EquipmentData, PredictionResult

router = APIRouter()

@router.post("/single", response_model=PredictionResult)
def predict_single(
    input_data: EquipmentData,
    controller: PredictController = Depends(get_predict_controller)
):
    """
    Returns the prediction result for a single equipment data.
    """
    return controller.predict(input_data)
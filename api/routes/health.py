from fastapi import APIRouter, Depends
from api.controllers.health_controller import HealthController
from api.dependencies import get_health_controller
from api.domain.entities import HealthStatus

router = APIRouter()

@router.get("/", response_model=HealthStatus)
def health_check(
    controller: HealthController = Depends(get_health_controller)
):
    """
    Returns the health status of the API.
    """
    return controller.health_check()
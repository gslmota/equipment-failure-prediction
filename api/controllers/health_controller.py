# api/controllers/health_controller.py

from api.domain.entities import HealthStatus
from datetime import datetime

class HealthController:
    def health_check(self) -> HealthStatus:
        """
        Returns the health status of the API.
        """
        return HealthStatus(
            status="ok",
            version="1.0.0",
            timestamp=datetime.utcnow()
        )

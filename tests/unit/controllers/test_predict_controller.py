from unittest.mock import MagicMock
from api.controllers.predict_controller import PredictController
from api.services.inference_service import InferenceService
from api.domain.entities import EquipmentData

def test_predict_controller():
    mock_service = MagicMock(spec=InferenceService)
    mock_service.predict_single.return_value = {
        'failure_probability': 0.75,
        'risk_level': 'high'
    }
    controller = PredictController(mock_service)
    input_data = EquipmentData(
        Cycle=100,
        Temperature=25.0,
        Pressure=100.0,
        VibrationX=10.0,
        VibrationY=10.0,
        VibrationZ=10.0,
        Frequency=50.0,
        Preset_1=1,
        Preset_2=1
    )
    
    result = controller.predict(input_data)
    
    mock_service.predict_single.assert_called_once_with(input_data)
    assert result['failure_probability'] == 0.75
    assert result['risk_level'] == 'high'
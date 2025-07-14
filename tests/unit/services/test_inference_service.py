import pytest
from unittest.mock import MagicMock
import numpy as np
from api.services.inference_service import InferenceService
from api.repositories.model_repository import ModelRepository
from api.domain.entities import EquipmentData, RiskLevel

@pytest.fixture
def mock_repo():
    repo = MagicMock(spec=ModelRepository)
    repo.load_artifacts.return_value = {
        'model': MagicMock(),
        'scaler': MagicMock(),
        'metadata': {
            'features': ['feature1', 'feature2'],
            'window_size': 5,
            'preset_risk': {(1, 1): 0.1}
        }
    }
    return repo

def test_predict_single(mock_repo):
    service = InferenceService(mock_repo)
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

    mock_model = mock_repo.load_artifacts.return_value['model']
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    mock_scaler = mock_repo.load_artifacts.return_value['scaler']
    mock_scaler.transform.return_value = np.array([[0.5, 0.5]])
    
    result = service.predict_single(input_data)
    
    assert result.failure_probability == 0.8
    assert result.risk_level == RiskLevel.HIGH
    mock_model.predict_proba.assert_called_once()
    
    mock_scaler.transform.assert_called_once()

def test_missing_features_handling(mock_repo):
    service = InferenceService(mock_repo)
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
    
    mock_model = mock_repo.load_artifacts.return_value['model']
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    mock_scaler = mock_repo.load_artifacts.return_value['scaler']
    mock_scaler.transform.return_value = np.array([[0.1, 0.2]])
    
    result = service.predict_single(input_data)
    
    assert result is not None
    assert result.failure_probability == 0.8

def test_determine_risk_level():
    service = InferenceService(MagicMock())
    assert service._determine_risk_level(0.2) == RiskLevel.LOW
    assert service._determine_risk_level(0.5) == RiskLevel.MEDIUM
    assert service._determine_risk_level(0.8) == RiskLevel.HIGH
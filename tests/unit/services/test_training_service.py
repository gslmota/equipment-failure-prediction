from unittest.mock import MagicMock, patch
import pandas as pd
from api.services.training_service import TrainingService
from api.repositories.model_repository import ModelRepository
from api.domain.entities import TrainingParameters


@patch('api.services.training_service.DataPipeline')
@patch('api.services.training_service.TrainingPipeline')
def test_train_model(mock_training_pipeline, mock_data_pipeline):
    mock_repo = MagicMock(spec=ModelRepository)
    service = TrainingService(mock_repo)
    
    mock_data = pd.DataFrame({
        'Temperature': [25, 26],
        'Pressure': [100, 101],
        'VibrationX': [10, 11],
        'VibrationY': [5, 6],
        'VibrationZ': [3, 4],
        'Frequency': [50, 51],
        'Fail': [0, 1]
    })
    
    mock_data_pipeline.return_value.load_data.return_value = mock_data
    
    mock_data_pipeline.return_value.split_data.return_value = (
        mock_data.iloc[:1].copy(),
        mock_data.iloc[1:].copy()
    )
    
    mock_data_pipeline.return_value.full_preprocessing.return_value = (
        mock_data.iloc[:1].copy(), 
        mock_data.iloc[1:].copy(), 
        ['feat1', 'feat2'], 
        MagicMock(),
        ['p1_1', 'p2_1']
    )
    
    mock_training_pipeline.return_value.execute_training.return_value = (
        'model_obj', {'metric': 0.95}
    )
    
    params = TrainingParameters(
        data_path="path/to/data.xlsx",
        sheet_name="Sheet1"
    )
    
    metrics = service.train_model(params)
    
    assert metrics == {'metric': 0.95}
    mock_repo.save_artifacts.assert_called_once()
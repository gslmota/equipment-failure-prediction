import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from api.repositories.model_repository import ModelRepository

@patch('api.repositories.model_repository.joblib')
def test_load_artifacts_success(mock_joblib):
    mock_joblib.load.side_effect = [
        'model', 'scaler', {'key': 'value'}  
    ]
    repo = ModelRepository()
    
    artifacts = repo.load_artifacts()
    
    assert artifacts['model'] == 'model'
    assert artifacts['scaler'] == 'scaler'
    assert artifacts['metadata'] == {'key': 'value'}

@patch('api.repositories.model_repository.joblib')
def test_load_artifacts_file_not_found(mock_joblib):
    mock_joblib.load.side_effect = FileNotFoundError
    repo = ModelRepository()
    
    with pytest.raises(RuntimeError, match="Model artifacts not found"):
        repo.load_artifacts()

@patch('api.repositories.model_repository.joblib')
@patch('api.repositories.model_repository.Path.mkdir')
def test_save_artifacts(mock_mkdir, mock_joblib):
    artifacts = {
        'model': 'model_obj',
        'scaler': 'scaler_obj',
        'metadata': {'data': 'value'}
    }
    repo = ModelRepository()
    
    repo.save_artifacts(artifacts)
    
    mock_joblib.dump.assert_any_call('model_obj', Path('training/artifacts/model.cbm'))
    mock_joblib.dump.assert_any_call('scaler_obj', Path('training/artifacts/scaler.pkl'))
    mock_joblib.dump.assert_any_call({'data': 'value'}, Path('training/artifacts/metadata.pkl'))
    assert repo.artifacts == artifacts
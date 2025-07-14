from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_predict_endpoint():
    # Dados de exemplo
    payload = {
        "Cycle": 100,
        "Temperature": 25.0,
        "Pressure": 100.0,
        "VibrationX": 10.0,
        "VibrationY": 10.0,
        "VibrationZ": 10.0,
        "Frequency": 50.0,
        "Preset_1": 1,
        "Preset_2": 1
    }
    
    response = client.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["low", "medium", "high"]

@patch('api.services.training_service.TrainingService.train_model')
def test_train_endpoint(mock_train):
    mock_train.return_value = {"status": "success"}
    
    response = client.post("/train/", json={
        "data_path": "path/to/data.xlsx",
        "sheet_name": "Sheet1"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert "Training process started" in data["message"]


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"  # Atualizado para "ok"
    assert "version" in data
    assert "timestamp" in data
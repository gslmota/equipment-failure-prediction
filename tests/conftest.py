import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)
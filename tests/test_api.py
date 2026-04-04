from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_api_status():
    """Verifica que la API responda con el 404 esperado en la raíz (porque solo usamos /predict)"""
    response = client.get("/")
    assert response.status_code == 404
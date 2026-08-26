def test_health_returns_expected_response(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "SmartDoc AI backend running",
    }

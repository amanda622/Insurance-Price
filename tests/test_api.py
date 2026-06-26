"""Tests for the HTTP API."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid(client):
    payload = {
        "age": 35,
        "sex": "female",
        "bmi": 27.5,
        "children": 2,
        "smoker": "no",
        "region": "southwest",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_charge"] > 0
    assert body["currency"] == "USD"


def test_predict_invalid_age_returns_422(client):
    payload = {
        "age": 200,  # out of allowed range
        "sex": "female",
        "bmi": 27.5,
        "children": 2,
        "smoker": "no",
        "region": "southwest",
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_invalid_enum_returns_422(client):
    payload = {
        "age": 35,
        "sex": "female",
        "bmi": 27.5,
        "children": 2,
        "smoker": "maybe",  # not a valid enum value
        "region": "southwest",
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_logs_to_database(client):
    payload = {
        "age": 41,
        "sex": "male",
        "bmi": 29.1,
        "children": 1,
        "smoker": "yes",
        "region": "southeast",
    }
    client.post("/predict", json=payload)
    history = client.get("/predictions").json()
    assert any(row["age"] == 41 and row["smoker"] == "yes" for row in history)


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Insurance Price Prediction" in response.text

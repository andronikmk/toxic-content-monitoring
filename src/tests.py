from starlette.testclient import TestClient

from main import app

client = TestClient(app)

def test_root_predict():
    json_blob = {"text": "string"}
    resp = client.post("/predict/", json=json_blob)
    assert resp.status_code == 200

def test_predict_sentence():
    json_blob = {"text": "Is this toxic"}
    resp = client.post("/predict/", json=json_blob)
    assert resp.status_code == 200

def test_wrong_root():
    json_blob = {"text": "This is a sentence."}
    resp = client.post("/toxic/", json=json_blob)
    assert resp.status_code != 200
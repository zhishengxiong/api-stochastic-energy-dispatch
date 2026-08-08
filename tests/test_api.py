from contextlib import contextmanager
from datetime import datetime

from fastapi.testclient import TestClient

from stochastic_energy_dispatch import api
from stochastic_energy_dispatch.case_schemas import TSSPRunResult

client = TestClient(api.app)


@contextmanager
def _fake_connection():
    yield object()


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "TSSP API is running"}


def test_create_run_endpoint(monkeypatch):
    monkeypatch.setattr(
        api,
        "run_case",
        lambda config: TSSPRunResult(
            run_id=37,
            T=config.T,
            num_samples=config.num_samples,
            objective_value=21081.05,
        ),
    )

    response = client.post(
        "/runs",
        json={
            "T": 4,
            "num_samples": 5,
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "run_id": 37,
        "T": 4,
        "num_samples": 5,
        "objective_value": 21081.05,
    }


def test_create_run_returns_422_for_invalid_request():
    response = client.post(
        "/runs",
        json={
            "T": 0,
            "num_samples": 5,
        },
    )

    assert response.status_code == 422


def test_create_run_hides_internal_error(monkeypatch):
    def _raise_error(config):
        raise RuntimeError("database password is secret")

    monkeypatch.setattr(api, "run_case", _raise_error)

    response = client.post(
        "/runs",
        json={
            "T": 4,
            "num_samples": 5,
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "TSSP optimization run failed."}
    assert "secret" not in response.text


def test_read_run_endpoint(monkeypatch):
    monkeypatch.setattr(api, "get_connection", _fake_connection)
    monkeypatch.setattr(
        api.input_repository,
        "read_optimization_run",
        lambda connection, run_id: {
            "run_id": run_id,
            "T": 4,
            "num_samples": 5,
            "objective_value": 21081.05,
            "created_at": datetime(2026, 8, 8, 12, 0, 0),
        },
    )

    response = client.get("/runs/37")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == 37
    assert body["T"] == 4
    assert body["num_samples"] == 5
    assert body["objective_value"] == 21081.05


def test_read_run_returns_404(monkeypatch):
    monkeypatch.setattr(api, "get_connection", _fake_connection)
    monkeypatch.setattr(
        api.input_repository,
        "read_optimization_run",
        lambda connection, run_id: None,
    )

    response = client.get("/runs/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Run 999 not found."}


def test_read_run_rejects_non_positive_run_id():
    response = client.get("/runs/0")

    assert response.status_code == 422

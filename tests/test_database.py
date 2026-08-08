from stochastic_energy_dispatch import database


def test_get_connection_uses_environment_password(
    monkeypatch,
):
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("DB_HOST", "db-host")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("DB_USER", "test_user")

    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        database.psycopg,
        "connect",
        fake_connect,
    )

    # Support both the cached-password and non-cached versions.
    if hasattr(database, "_password"):
        monkeypatch.setattr(database, "_password", None)

    database.get_connection()

    assert captured == {
        "host": "db-host",
        "port": 5433,
        "dbname": "test_db",
        "user": "test_user",
        "password": "secret",
    }


def test_get_connection_uses_environment_variables(monkeypatch):
    monkeypatch.setenv("DB_HOST", "test-host")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "test-db")
    monkeypatch.setenv("DB_USER", "test-user")
    monkeypatch.setenv("DB_PASSWORD", "test-password")

    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return "connection"

    monkeypatch.setattr(database.psycopg, "connect", fake_connect)

    result = database.get_connection()

    assert result == "connection"
    assert captured == {
        "host": "test-host",
        "port": 5433,
        "dbname": "test-db",
        "user": "test-user",
        "password": "test-password",
    }

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


def test_get_connection_prompts_when_password_missing(
    monkeypatch,
):
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    if hasattr(database, "_password"):
        monkeypatch.setattr(database, "_password", None)

    monkeypatch.setattr(
        database,
        "getpass",
        lambda prompt: "typed-password",
    )

    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        database.psycopg,
        "connect",
        fake_connect,
    )

    database.get_connection()

    assert captured["password"] == "typed-password"

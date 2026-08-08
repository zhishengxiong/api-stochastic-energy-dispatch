import requests


BASE_URL = "http://127.0.0.1:8888"


def main() -> None:
    payload = {
        "T": 4,
        "num_samples": 5,
    }

    create_response = requests.post(
        f"{BASE_URL}/runs",
        json=payload,
        timeout=300,
    )
    create_response.raise_for_status()

    created_run = create_response.json()
    print("POST /runs")
    print(created_run)

    run_id = created_run["run_id"]

    get_response = requests.get(
        f"{BASE_URL}/runs/{run_id}",
        timeout=30,
    )
    get_response.raise_for_status()

    stored_run = get_response.json()
    print("\nGET /runs/{run_id}")
    print(stored_run)


if __name__ == "__main__":
    main()

# API Stochastic Energy Dispatch

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimizer-EE3524)](https://www.gurobi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A database-backed, containerized, testable, and API-accessible software service for **two-stage stochastic energy dispatch**. The current reference implementation targets optimal dispatch of distribution networks under uncertain PV generation and load demand, while the engineering architecture can be adapted to other scheduling and optimization applications under uncertainty.

---

## Overview

This project implements a **two-stage stochastic programming (TSSP)** workflow for distribution-network operation and exposes it through a structured backend architecture.

The project integrates:

- **PostgreSQL** for persistent model inputs and outputs
- **psycopg** for database access
- **Gurobi** for optimization modeling and solution
- **FastAPI** for REST endpoints
- **Docker** for containerized execution
- **Docker Compose** for API/database orchestration
- **pytest** for automated testing
- **Ruff** for linting and formatting
- **GitHub Actions** for continuous integration

---

## Purpose

Starting from an existing TSSP benchmark used as a comparison method in our research on distribution-network operation under uncertainty, the codebase was refactored to integrate relational database design, REST APIs, environment-based configuration, containerization, automated testing, code-quality checks, and CI. The underlying research application is described in:

> Zhisheng Xiong, Dimitris Boskos, Bo Zeng, Peter Palensky, Pedro P. Vergara,
> **“Robust Operation of Distribution Networks: Generalized Uncertainty Modelling in Confidence-Level-Based Information Gap Decision,”**
> arXiv:2604.23252, 2026.
> https://arxiv.org/abs/2604.23252

The final result is a **deployable, production-style optimization backend** that retains the original stochastic-dispatch logic while adding the software infrastructure needed to run, store, expose, and validate optimization workflows in a structured way.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Database | PostgreSQL | Persistent storage for inputs and results |
| DB Client | psycopg | Python–PostgreSQL communication |
| Optimization | Gurobi / `gurobipy` | Builds and solves the TSSP model |
| API | FastAPI | Creates and retrieves optimization runs |
| Validation | Pydantic | Request, response, and configuration validation |
| Application | Python 3.12 | Core implementation |
| Containerization | Docker | Reproducible API runtime |
| Orchestration | Docker Compose | Runs API and database services together |
| Testing | pytest | Automated testing |
| Code Quality | Ruff | Linting and formatting |
| CI | GitHub Actions | Automated checks on push and pull request |

---

## System Architecture

The codebase separates the main responsibilities into layers.

### API and Workflow

- FastAPI exposes HTTP endpoints for creating and retrieving optimization runs.
- The workflow layer coordinates data loading, optimization execution, and result persistence.
- The optimization workflow is independent from the HTTP layer, so the same core logic can also be executed directly from Python.

### Optimization

- Gurobi constructs and solves the two-stage stochastic dispatch model.
- The optimization layer is isolated from API and database-access details.
- The current model is specific to distribution-network stochastic dispatch.

### Database and Persistence

- PostgreSQL stores system data, stochastic scenarios, optimization-run metadata, and detailed output results.
- Each optimization execution is assigned a persistent `run_id`.
- Database access is isolated in repository modules.
- SQL operations use parameterized queries rather than constructing statements from raw user input.
- PostgreSQL data persist across normal container restarts through a named Docker volume.

### Containerization and Configuration

- Docker packages the API runtime.
- Docker Compose runs the API and PostgreSQL as connected services.
- Database configuration is supplied through environment variables.
- Local credentials are stored in `.env`, which is excluded from version control.
- The Gurobi license remains an external runtime dependency and is mounted into the API container.

### Quality Assurance

- pytest provides automated tests.
- Ruff enforces linting and formatting.
- GitHub Actions runs checks on pushes and pull requests.
- CI also builds the Docker image, starts the API container, and performs an API health-endpoint smoke test.

---

## Data

The repository includes the IEEE 33-node reference case and the data required to run the stochastic dispatch example.

The historical load and PV data were generated following the methodology used in:

> H. Shengren, P. P. Vergara, E. M. Salazar Duque, P. Palensky,
> **“Optimal energy system scheduling using a constraint-aware reinforcement learning algorithm,”**
> *International Journal of Electrical Power & Energy Systems*, 152, 109230, 2023.
> DOI: 10.1016/j.ijepes.2023.109230

During normal execution, optimization inputs are loaded from PostgreSQL.

---

## Repository Structure

```text
api-stochastic-energy-dispatch/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- data/
|-- examples/
|   |-- run_case.py
|   `-- call_api.py
|-- scripts/
|   |-- init_schema.sql
|   `-- load_seed_data.py
|-- src/
|   `-- stochastic_energy_dispatch/
|       |-- api.py
|       |-- api_schemas.py
|       |-- case_schemas.py
|       |-- database.py
|       |-- optimization_workflow.py
|       |-- input_repository.py
|       |-- output_repository.py
|       |-- system_data_preprocessing.py
|       |-- ders_data_preprocessing.py
|       |-- samples_preprocessing.py
|       `-- tssp_model.py
|-- tests/
|-- .dockerignore
|-- .editorconfig
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- compose.yaml
|-- LICENSE
|-- pyproject.toml
`-- README.md
```

---

# Run the Project

The project supports two entry points:

1. **Direct Python execution**
2. **Containerized REST API**

---

## Quick Start with Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/zhishengxiong/api-stochastic-energy-dispatch.git
cd api-stochastic-energy-dispatch
```

### 2. Create the local environment file

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Example `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=energy_optimization_db
DB_USER=postgres
DB_PASSWORD=your_password
GUROBI_LICENSE_PATH=path/to/your/gurobi.lic
```

The real `.env` file is ignored by Git and should not be committed.

### 3. Configure the Gurobi license mount

A valid Gurobi license is required and is **not included in this repository**.

Set `GUROBI_LICENSE_PATH` in your local `.env` file to the path of your Gurobi license. Docker Compose uses this value to mount the license into the API container.

### 4. Start the services

```bash
docker compose up --build -d
```

This starts:

- `postgres_database`
- `tssp_api`

Check service status:

```bash
docker compose ps
```

### 5. Load the reference data

After the PostgreSQL database is initialized, load the seed data once:

```bash
docker compose exec tssp_api python scripts/load_seed_data.py
```

### 6. Open the API

API base URL:

```text
http://127.0.0.1:8888
```

Interactive FastAPI documentation:

```text
http://127.0.0.1:8888/docs
```

### 7. Stop the services

```bash
docker compose down
```

This keeps the PostgreSQL named volume and stored data.

To also remove the database volume:

```bash
docker compose down -v
```

If the volume is deleted, the seed-data command must be run again after recreating the services.

---

## Direct Python Execution

For local execution, install the project from the repository root:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[test]"
```

Then run:

```bash
python examples/run_case.py
```

This entry point executes the optimization workflow directly without using the REST API.

Direct execution assumes that PostgreSQL has already been initialized with `scripts/init_schema.sql` and populated with the reference data using `scripts/load_seed_data.py`.

Local execution requires PostgreSQL, Gurobi Optimizer, a valid Gurobi license, and database settings supplied through environment variables.

---

## Environment and Database Configuration

Database configuration is supplied through environment variables rather than being hard-coded in the application.

The committed `.env.example` documents the expected keys, while the real `.env` remains local and is excluded through `.gitignore`.

For local execution, the default database host is:

```text
DB_HOST=localhost
```

Inside Docker Compose, the API connects to PostgreSQL through the Compose service name:

```text
DB_HOST=postgres_database
```

The PostgreSQL schema is defined in:

```text
scripts/init_schema.sql
```

For Docker Compose, the script runs automatically when the database volume is created for the first time.

Reference-case data are loaded through:

```bash
docker compose exec tssp_api python scripts/load_seed_data.py
```

The PostgreSQL service uses the named volume:

```text
postgres_data
```

This allows stored input data and optimization results to persist across normal container restarts.

---

## API Usage

The current API exposes three endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API health/root endpoint |
| `POST` | `/runs` | Create and execute a new optimization run |
| `GET` | `/runs/{run_id}` | Retrieve a stored optimization run |

### Create a Run

Example request body:

```json
{
  "T": 4,
  "num_samples": 5
}
```

Example Python client:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8888/runs",
    json={
        "T": 4,
        "num_samples": 5,
    },
    timeout=300,
)

response.raise_for_status()
print(response.json())
```

A complete example is available in `examples/call_api.py`.

---

## Testing and CI

Run all local quality checks:

```bash
ruff check .
ruff format --check .
pytest
```

Automatically format the repository with:

```bash
ruff format .
```

GitHub Actions runs automatically on pushes and pull requests to `main`.

The CI pipeline verifies:

- package installation in a clean Python 3.12 environment
- Ruff linting
- Ruff format compliance
- pytest
- Docker image build
- API container startup
- API health-endpoint smoke test

The full optimization model is intentionally not solved in CI because optimization execution depends on an external Gurobi license.

---

## Related Work

For an algorithm-focused repository emphasizing uncertainty modeling and optimization methodology rather than backend engineering, see:

[`cl-igdt-optimization-algorithm`](https://github.com/zhishengxiong/cl-igdt-optimization-algorithm)

The two repositories intentionally emphasize different skills:

- **CL-IGDT repository** — optimization algorithm development and uncertainty modeling
- **this repository** — optimization software engineering, API development, database persistence, containerization, testing, and CI

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

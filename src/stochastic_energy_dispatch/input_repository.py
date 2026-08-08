import pandas as pd


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _fetch_dataframe(connection, query, parameters=None):
    """Execute a SELECT query and return the result as a DataFrame."""
    with connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def fetch_nodes(connection):
    query = """
        SELECT
            node_id,
            is_slack,
            pd_base
        FROM nodes
        ORDER BY node_id;
    """
    return _fetch_dataframe(connection, query)


def fetch_lines(connection):
    query = """
        SELECT
            line_id,
            from_node,
            to_node,
            resistance,
            reactance
        FROM lines
        ORDER BY line_id;
    """
    return _fetch_dataframe(connection, query)


def fetch_demand_forecasts(connection, T: int):
    _validate_positive_int(T, "T")

    query = """
        SELECT
            node_id,
            time_step,
            demand_scale
        FROM demand_forecasts
        WHERE time_step < %s
        ORDER BY node_id, time_step;
    """
    return _fetch_dataframe(connection, query, (T,))


def fetch_generators(connection):
    query = """
        SELECT
            generator_id,
            node_id,
            p_max,
            p_min,
            q_max,
            q_min,
            ramp_up,
            ramp_down,
            generation_cost,
            startup_cost
        FROM generators
        ORDER BY generator_id;
    """
    return _fetch_dataframe(connection, query)


def fetch_energy_storage(connection):
    query = """
        SELECT
            storage_id,
            node_id,
            power_capacity,
            energy_capacity,
            initial_energy,
            efficiency
        FROM energy_storage
        ORDER BY storage_id;
    """
    return _fetch_dataframe(connection, query)


def fetch_electricity_prices(connection, T: int):
    _validate_positive_int(T, "T")

    query = """
        SELECT
            time_step,
            price
        FROM electricity_prices
        WHERE time_step < %s
        ORDER BY time_step;
    """
    return _fetch_dataframe(connection, query, (T,))


def fetch_pv_units(connection):
    query = """
        SELECT
            pv_id,
            node_id,
            capacity
        FROM pv_units
        ORDER BY pv_id;
    """
    return _fetch_dataframe(connection, query)


def fetch_pv_forecasts(connection, T: int):
    _validate_positive_int(T, "T")

    query = """
        SELECT
            time_step,
            pv_scale
        FROM pv_forecasts
        WHERE time_step < %s
        ORDER BY time_step;
    """
    return _fetch_dataframe(connection, query, (T,))


def fetch_reserve_units(connection):
    query = """
        SELECT
            reserve_id,
            node_id,
            reserve_cost
        FROM reserve_units
        ORDER BY reserve_id;
    """
    return _fetch_dataframe(connection, query)


def fetch_historical_scenarios(
    connection,
    T: int,
    num_samples: int,
):
    _validate_positive_int(T, "T")
    if not 1 <= num_samples <= 500:
        raise ValueError("num_samples must be between 1 and 500.")

    query = """
        SELECT
            sample_id,
            time_step,
            pv_scale,
            load_scale
        FROM historical_scenarios
        WHERE sample_id IN (
            SELECT DISTINCT sample_id
            FROM historical_scenarios
            ORDER BY sample_id
            LIMIT %s
        )
          AND time_step < %s
        ORDER BY sample_id, time_step;
    """
    return _fetch_dataframe(
        connection,
        query,
        (num_samples, T),
    )




def read_optimization_run(connection, run_id: int):
    _validate_positive_int(run_id, "run_id")

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                run_id,
                time_horizon,
                num_samples,
                objective_value,
                created_at
            FROM optimization_runs
            WHERE run_id = %s;
            """,
            (run_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "run_id": row[0],
        "T": row[1],
        "num_samples": row[2],
        "objective_value": float(row[3]),
        "created_at": row[4],
    }

def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _validate_tssp_result(tssp_result) -> dict:
    required_result_keys = {"cost", "x_hat"}
    missing_result_keys = required_result_keys - tssp_result.keys()

    if missing_result_keys:
        raise KeyError(f"Missing TSSP result keys: {sorted(missing_result_keys)}")

    x_hat = tssp_result["x_hat"]

    required_x_hat_keys = {
        "G_p",
        "G_u",
        "G_s",
        "R_pmax",
        "ESS_pch",
        "ESS_pdis",
        "ESS_E",
        "ESS_u",
        "P_net",
        "P_flow",
        "flow_buy",
        "flow_sell",
        "flow_u",
    }

    missing_x_hat_keys = required_x_hat_keys - x_hat.keys()

    if missing_x_hat_keys:
        raise KeyError(f"Missing first-stage result keys: {sorted(missing_x_hat_keys)}")

    return x_hat


def _validate_ders_ids(ders_data) -> None:
    if len(ders_data.G_id) != len(ders_data.G_node):
        raise ValueError("Generator IDs and generator nodes have different lengths.")

    if len(ders_data.R_id) != len(ders_data.R_node):
        raise ValueError("Reserve IDs and reserve nodes have different lengths.")

    if len(ders_data.ESS_id) != len(ders_data.ESS_node):
        raise ValueError("Storage IDs and storage nodes have different lengths.")


def save_tssp_results(
    connection,
    tssp_result,
    ders_data,
    num_nodes,
    T,
    num_samples,
):
    """Save one TSSP run and its detailed results to PostgreSQL."""
    _validate_positive_int(num_nodes, "num_nodes")
    _validate_positive_int(T, "T")
    _validate_positive_int(num_samples, "num_samples")

    x_hat = _validate_tssp_result(tssp_result)
    _validate_ders_ids(ders_data)

    with connection.cursor() as cursor:
        # ---------- Optimization run ----------
        cursor.execute(
            """
            INSERT INTO optimization_runs (
                time_horizon,
                num_samples,
                objective_value
            )
            VALUES (%s, %s, %s)
            RETURNING run_id;
            """,
            (
                T,
                num_samples,
                float(tssp_result["cost"]),
            ),
        )

        run_id = cursor.fetchone()[0]

        # ---------- Generator results ----------
        generator_rows = [
            (
                run_id,
                generator_id,
                time_step,
                float(x_hat["G_p"][generator_index, time_step]),
                float(x_hat["G_u"][generator_index, time_step]),
                float(x_hat["G_s"][generator_index, time_step]),
            )
            for generator_index, generator_id in enumerate(ders_data.G_id)
            for time_step in range(T)
        ]

        cursor.executemany(
            """
            INSERT INTO generator_results (
                run_id,
                generator_id,
                time_step,
                p_output,
                commitment,
                startup
            )
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            generator_rows,
        )

        # ---------- Reserve results ----------
        reserve_rows = [
            (
                run_id,
                reserve_id,
                time_step,
                float(x_hat["R_pmax"][reserve_index, time_step]),
            )
            for reserve_index, reserve_id in enumerate(ders_data.R_id)
            for time_step in range(T)
        ]

        cursor.executemany(
            """
            INSERT INTO reserve_results (
                run_id,
                reserve_id,
                time_step,
                reserved_power
            )
            VALUES (%s, %s, %s, %s);
            """,
            reserve_rows,
        )

        # ---------- Storage results ----------
        storage_rows = [
            (
                run_id,
                storage_id,
                time_step,
                float(x_hat["ESS_pch"][storage_index, time_step]),
                float(x_hat["ESS_pdis"][storage_index, time_step]),
                float(x_hat["ESS_E"][storage_index, time_step]),
                float(x_hat["ESS_u"][storage_index, time_step]),
            )
            for storage_index, storage_id in enumerate(ders_data.ESS_id)
            for time_step in range(T)
        ]

        cursor.executemany(
            """
            INSERT INTO storage_results (
                run_id,
                storage_id,
                time_step,
                charging_power,
                discharging_power,
                energy_level,
                charging_status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            storage_rows,
        )

        # ---------- Node results ----------
        node_rows = [
            (
                run_id,
                node_index + 2,
                time_step,
                float(x_hat["P_net"][node_index, time_step]),
            )
            for node_index in range(num_nodes - 1)
            for time_step in range(T)
        ]

        cursor.executemany(
            """
            INSERT INTO node_results (
                run_id,
                node_id,
                time_step,
                net_injection
            )
            VALUES (%s, %s, %s, %s);
            """,
            node_rows,
        )

        # ---------- Line results ----------
        line_rows = [
            (
                run_id,
                line_index + 1,
                time_step,
                float(x_hat["P_flow"][line_index, time_step]),
            )
            for line_index in range(num_nodes - 1)
            for time_step in range(T)
        ]

        cursor.executemany(
            """
            INSERT INTO line_results (
                run_id,
                line_id,
                time_step,
                active_flow
            )
            VALUES (%s, %s, %s, %s);
            """,
            line_rows,
        )

        # ---------- Grid-exchange results ----------
        grid_exchange_rows = [
            (
                run_id,
                time_step,
                float(x_hat["flow_buy"][time_step]),
                float(x_hat["flow_sell"][time_step]),
                float(x_hat["flow_u"][time_step]),
            )
            for time_step in range(T)
        ]

        cursor.executemany(
            """
            INSERT INTO grid_exchange_results (
                run_id,
                time_step,
                power_bought,
                power_sold,
                buy_status
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            grid_exchange_rows,
        )

    return run_id

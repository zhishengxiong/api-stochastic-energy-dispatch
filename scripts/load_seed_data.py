from pathlib import Path

import numpy as np
import pandas as pd

from stochastic_energy_dispatch.database import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

NETWORK_FILE = DATA_DIR / "IEEE33.xlsx"
DERS_FILE = DATA_DIR / "DERs_Data_33.xlsx"
PV_HISTORY_FILE = DATA_DIR / "historical_data_PV_33.xlsx"
LOAD_HISTORY_FILE = DATA_DIR / "historical_data_demand_33.xlsx"

NUM_HISTORICAL_SAMPLES = 500


def validate_source_files():
    required_files = [
        NETWORK_FILE,
        DERS_FILE,
        PV_HISTORY_FILE,
        LOAD_HISTORY_FILE,
    ]

    missing_files = [str(file_path) for file_path in required_files if not file_path.is_file()]

    if missing_files:
        raise FileNotFoundError("Missing seed-data files: " + ", ".join(missing_files))


def validate_numeric_values(dataframe, name):
    numeric_values = dataframe.select_dtypes(include="number").to_numpy()

    if numeric_values.size and not np.isfinite(numeric_values).all():
        raise ValueError(f"{name} contains missing or non-finite numeric values.")


def clear_tables(cursor):
    cursor.execute(
        """
        TRUNCATE TABLE
            reserve_units,
            pv_forecasts,
            pv_units,
            electricity_prices,
            energy_storage,
            generators,
                        historical_scenarios,
            demand_forecasts,
            lines,
            nodes
        RESTART IDENTITY CASCADE;
        """
    )


def load_nodes(cursor):
    df = pd.read_excel(NETWORK_FILE, sheet_name="Nodes")

    rows = [
        (
            int(row["N"]),
            bool(row["Tn"]),
            float(row["PD_base"]),
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(
        """
        INSERT INTO nodes (node_id, is_slack, pd_base)
        VALUES (%s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into nodes.")


def load_lines(cursor):
    df = pd.read_excel(NETWORK_FILE, sheet_name="Lines")

    rows = [
        (
            line_id,
            int(row["FROM"]),
            int(row["TO"]),
            float(row["R"]),
            float(row["X"]),
        )
        for line_id, (_, row) in enumerate(df.iterrows(), start=1)
    ]

    cursor.executemany(
        """
        INSERT INTO lines (
            line_id,
            from_node,
            to_node,
            resistance,
            reactance
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into lines.")


def load_demand_forecasts(cursor):
    df = pd.read_excel(
        NETWORK_FILE,
        sheet_name="Predictive_demand",
        header=None,
    )

    rows = []

    for node_offset, demand_profile in df.iterrows():
        node_id = node_offset + 2

        for time_step, demand_scale in enumerate(demand_profile):
            rows.append(
                (
                    node_id,
                    time_step,
                    float(demand_scale),
                )
            )

    cursor.executemany(
        """
        INSERT INTO demand_forecasts (
            node_id,
            time_step,
            demand_scale
        )
        VALUES (%s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into demand_forecasts.")


def load_generators(cursor):
    df = pd.read_excel(DERS_FILE, sheet_name="Generators")

    rows = [
        (
            generator_id,
            int(row["Node"]),
            float(row["Pmax"]),
            float(row["Pmin"]),
            float(row["Qmax"]),
            float(row["Qmin"]),
            float(row["RU"]),
            float(row["RD"]),
            float(row["Cost"]),
            float(row["UCost"]),
        )
        for generator_id, (_, row) in enumerate(df.iterrows(), start=1)
    ]

    cursor.executemany(
        """
        INSERT INTO generators (
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
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into generators.")


def load_energy_storage(cursor):
    df = pd.read_excel(DERS_FILE, sheet_name="ESS")

    rows = [
        (
            storage_id,
            int(row["Node"]),
            float(row["Power"]),
            float(row["Energy"]),
            float(row["Eini"]),
            float(row["Eff"]),
        )
        for storage_id, (_, row) in enumerate(df.iterrows(), start=1)
    ]

    cursor.executemany(
        """
        INSERT INTO energy_storage (
            storage_id,
            node_id,
            power_capacity,
            energy_capacity,
            initial_energy,
            efficiency
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into energy_storage.")


def load_electricity_prices(cursor):
    df = pd.read_excel(DERS_FILE, sheet_name="Prices")

    rows = [(time_step, float(price)) for time_step, price in enumerate(df["1"])]

    cursor.executemany(
        """
        INSERT INTO electricity_prices (time_step, price)
        VALUES (%s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into electricity_prices.")


def load_pv_units(cursor):
    location_df = pd.read_excel(DERS_FILE, sheet_name="PVLocation")
    capacity_df = pd.read_excel(DERS_FILE, sheet_name="PVCap")

    if len(location_df) != len(capacity_df):
        raise ValueError("PV locations and capacities have different lengths.")

    rows = [
        (
            pv_id,
            int(location_df.iloc[index]["Node"]),
            float(capacity_df.iloc[index]["Node"]),
        )
        for pv_id, index in enumerate(range(len(location_df)), start=1)
    ]

    cursor.executemany(
        """
        INSERT INTO pv_units (pv_id, node_id, capacity)
        VALUES (%s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into pv_units.")


def load_pv_forecasts(cursor):
    df = pd.read_excel(DERS_FILE, sheet_name="PVPredictive")

    rows = [(time_step, float(pv_scale)) for time_step, pv_scale in enumerate(df["1"])]

    cursor.executemany(
        """
        INSERT INTO pv_forecasts (time_step, pv_scale)
        VALUES (%s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into pv_forecasts.")


def load_reserve_units(cursor):
    df = pd.read_excel(DERS_FILE, sheet_name="Reserve")

    rows = [
        (
            reserve_id,
            int(row["Node"]),
            float(row["Cost"]),
        )
        for reserve_id, (_, row) in enumerate(df.iterrows(), start=1)
    ]

    cursor.executemany(
        """
        INSERT INTO reserve_units (
            reserve_id,
            node_id,
            reserve_cost
        )
        VALUES (%s, %s, %s);
        """,
        rows,
    )

    print(f"Loaded {len(rows)} rows into reserve_units.")


def build_scenario_rows(pv_df, load_df, first_sample, last_sample):
    if first_sample <= 0:
        raise ValueError("first_sample must be greater than 0.")

    if last_sample < first_sample:
        raise ValueError("last_sample must be greater than or equal to first_sample.")

    if len(pv_df) != len(load_df):
        raise ValueError(
            "PV and load scenario data have different time lengths: "
            f"{len(pv_df)} and {len(load_df)}."
        )

    expected_index = list(range(len(pv_df)))

    if pv_df.index.tolist() != expected_index:
        raise ValueError(
            f"PV scenario rows must use consecutive time indices 0 to {len(pv_df) - 1}."
        )

    if load_df.index.tolist() != expected_index:
        raise ValueError(
            f"Load scenario rows must use consecutive time indices 0 to {len(load_df) - 1}."
        )

    validate_numeric_values(pv_df, "PV scenario data")
    validate_numeric_values(load_df, "Load scenario data")

    rows = []

    for sample_id in range(first_sample, last_sample + 1):
        pv_column = str(sample_id)

        if pv_column not in pv_df.columns:
            raise ValueError(f"PV sample {sample_id} does not exist.")

        if sample_id not in load_df.columns:
            raise ValueError(f"Load sample {sample_id} does not exist.")

        for time_step in range(len(pv_df)):
            rows.append(
                (
                    sample_id,
                    time_step,
                    float(pv_df.loc[time_step, pv_column]),
                    float(load_df.loc[time_step, sample_id]),
                )
            )

    return rows


def load_scenarios(cursor):
    pv_df = pd.read_excel(PV_HISTORY_FILE, sheet_name="PVData")
    load_df = pd.read_excel(LOAD_HISTORY_FILE, sheet_name="LoadData")

    historical_rows = build_scenario_rows(
        pv_df,
        load_df,
        first_sample=1,
        last_sample=NUM_HISTORICAL_SAMPLES,
    )

    cursor.executemany(
        """
        INSERT INTO historical_scenarios (
            sample_id,
            time_step,
            pv_scale,
            load_scale
        )
        VALUES (%s, %s, %s, %s);
        """,
        historical_rows,
    )

    print(f"Loaded {len(historical_rows)} rows into historical_scenarios.")


def load_seed_data():
    validate_source_files()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            clear_tables(cursor)

            load_nodes(cursor)
            load_lines(cursor)
            load_demand_forecasts(cursor)
            load_generators(cursor)
            load_energy_storage(cursor)
            load_electricity_prices(cursor)
            load_pv_units(cursor)
            load_pv_forecasts(cursor)
            load_reserve_units(cursor)
            load_scenarios(cursor)

    print("All seed data loaded successfully.")


if __name__ == "__main__":
    load_seed_data()

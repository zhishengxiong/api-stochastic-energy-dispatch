-- TSSP database schema
-- Intended for PostgreSQL initialization on a fresh database.

CREATE TABLE nodes (
    node_id INTEGER PRIMARY KEY,
    is_slack BOOLEAN NOT NULL,
    pd_base DOUBLE PRECISION NOT NULL CHECK (pd_base >= 0),
    CHECK (node_id > 0)
);

-- The current TSSP model expects exactly one slack node.
CREATE UNIQUE INDEX one_slack_node
    ON nodes (is_slack)
    WHERE is_slack = TRUE;


CREATE TABLE lines (
    line_id INTEGER PRIMARY KEY,
    from_node INTEGER NOT NULL REFERENCES nodes(node_id),
    to_node INTEGER NOT NULL REFERENCES nodes(node_id),
    resistance DOUBLE PRECISION NOT NULL CHECK (resistance >= 0),
    reactance DOUBLE PRECISION NOT NULL CHECK (reactance >= 0),
    CHECK (line_id > 0),
    CHECK (from_node <> to_node)
);


CREATE TABLE demand_forecasts (
    node_id INTEGER NOT NULL REFERENCES nodes(node_id),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    demand_scale DOUBLE PRECISION NOT NULL CHECK (demand_scale >= 0),
    PRIMARY KEY (node_id, time_step)
);


CREATE TABLE historical_scenarios (
    sample_id INTEGER NOT NULL CHECK (sample_id > 0),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    pv_scale DOUBLE PRECISION NOT NULL CHECK (pv_scale >= 0),
    load_scale DOUBLE PRECISION NOT NULL CHECK (load_scale >= 0),
    PRIMARY KEY (sample_id, time_step)
);



CREATE TABLE generators (
    generator_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES nodes(node_id),
    p_max DOUBLE PRECISION NOT NULL,
    p_min DOUBLE PRECISION NOT NULL,
    q_max DOUBLE PRECISION NOT NULL,
    q_min DOUBLE PRECISION NOT NULL,
    ramp_up DOUBLE PRECISION NOT NULL CHECK (ramp_up >= 0),
    ramp_down DOUBLE PRECISION NOT NULL CHECK (ramp_down >= 0),
    generation_cost DOUBLE PRECISION NOT NULL CHECK (generation_cost >= 0),
    startup_cost DOUBLE PRECISION NOT NULL CHECK (startup_cost >= 0),
    CHECK (generator_id > 0),
    CHECK (p_min <= p_max),
    CHECK (q_min <= q_max)
);


CREATE TABLE energy_storage (
    storage_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES nodes(node_id),
    power_capacity DOUBLE PRECISION NOT NULL CHECK (power_capacity >= 0),
    energy_capacity DOUBLE PRECISION NOT NULL CHECK (energy_capacity >= 0),
    initial_energy DOUBLE PRECISION NOT NULL CHECK (initial_energy >= 0),
    efficiency DOUBLE PRECISION NOT NULL CHECK (
        efficiency > 0 AND efficiency <= 1
    ),
    CHECK (storage_id > 0),
    CHECK (initial_energy <= energy_capacity)
);


CREATE TABLE electricity_prices (
    time_step INTEGER PRIMARY KEY CHECK (time_step >= 0),
    price DOUBLE PRECISION NOT NULL
);


CREATE TABLE pv_units (
    pv_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES nodes(node_id),
    capacity DOUBLE PRECISION NOT NULL CHECK (capacity >= 0),
    CHECK (pv_id > 0)
);


CREATE TABLE pv_forecasts (
    time_step INTEGER PRIMARY KEY CHECK (time_step >= 0),
    pv_scale DOUBLE PRECISION NOT NULL CHECK (pv_scale >= 0)
);


CREATE TABLE reserve_units (
    reserve_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES nodes(node_id),
    reserve_cost DOUBLE PRECISION NOT NULL CHECK (reserve_cost >= 0),
    CHECK (reserve_id > 0)
);


CREATE TABLE optimization_runs (
    run_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMP WITHOUT TIME ZONE
        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    time_horizon INTEGER NOT NULL CHECK (time_horizon > 0),
    num_samples INTEGER NOT NULL CHECK (num_samples BETWEEN 1 AND 500),
    objective_value DOUBLE PRECISION NOT NULL
);


CREATE TABLE generator_results (
    run_id INTEGER NOT NULL
        REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    generator_id INTEGER NOT NULL
        REFERENCES generators(generator_id),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    p_output DOUBLE PRECISION NOT NULL,
    commitment DOUBLE PRECISION NOT NULL CHECK (
        commitment >= 0 AND commitment <= 1
    ),
    startup DOUBLE PRECISION NOT NULL CHECK (
        startup >= 0 AND startup <= 1
    ),
    PRIMARY KEY (run_id, generator_id, time_step)
);


CREATE TABLE reserve_results (
    run_id INTEGER NOT NULL
        REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    reserve_id INTEGER NOT NULL
        REFERENCES reserve_units(reserve_id),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    reserved_power DOUBLE PRECISION NOT NULL CHECK (reserved_power >= 0),
    PRIMARY KEY (run_id, reserve_id, time_step)
);


CREATE TABLE storage_results (
    run_id INTEGER NOT NULL
        REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    storage_id INTEGER NOT NULL
        REFERENCES energy_storage(storage_id),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    charging_power DOUBLE PRECISION NOT NULL CHECK (charging_power >= 0),
    discharging_power DOUBLE PRECISION NOT NULL CHECK (discharging_power >= 0),
    energy_level DOUBLE PRECISION NOT NULL CHECK (energy_level >= 0),
    charging_status DOUBLE PRECISION NOT NULL CHECK (
        charging_status >= 0 AND charging_status <= 1
    ),
    PRIMARY KEY (run_id, storage_id, time_step)
);


CREATE TABLE node_results (
    run_id INTEGER NOT NULL
        REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    node_id INTEGER NOT NULL
        REFERENCES nodes(node_id),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    net_injection DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, node_id, time_step)
);


CREATE TABLE line_results (
    run_id INTEGER NOT NULL
        REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    line_id INTEGER NOT NULL
        REFERENCES lines(line_id),
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    active_flow DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, line_id, time_step)
);


CREATE TABLE grid_exchange_results (
    run_id INTEGER NOT NULL
        REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    time_step INTEGER NOT NULL CHECK (time_step >= 0),
    power_bought DOUBLE PRECISION NOT NULL CHECK (power_bought >= 0),
    power_sold DOUBLE PRECISION NOT NULL CHECK (power_sold >= 0),
    buy_status DOUBLE PRECISION NOT NULL CHECK (
        buy_status >= 0 AND buy_status <= 1
    ),
    PRIMARY KEY (run_id, time_step)
);

from dataclasses import dataclass

import numpy as np

from stochastic_energy_dispatch import input_repository as repository

PD_BASE_COLUMN = "pd_base"
FROM_COLUMN = "from_node"
TO_COLUMN = "to_node"
R_COLUMN = "resistance"
X_COLUMN = "reactance"

SLACK_NODE_ID = 1
SLACK_BUS_INDEX = 0
IMPEDANCE_SCALING_FACTOR = 0.001


@dataclass
class SystemData:
    A: np.ndarray
    R_prime: np.ndarray
    X_prime: np.ndarray
    P_load: np.ndarray
    PD_base: np.ndarray


def read_network_data(connection, T):
    raw_nodes = repository.fetch_nodes(connection)
    raw_lines = repository.fetch_lines(connection)
    demand_forecasts = repository.fetch_demand_forecasts(connection, T)

    raw_demand = demand_forecasts.pivot(
        index="node_id",
        columns="time_step",
        values="demand_scale",
    )
    raw_demand = raw_demand.sort_index().sort_index(axis=1)

    return raw_nodes, raw_lines, raw_demand


def _validate_network_data(raw_nodes, raw_lines, num_nodes):
    if len(raw_nodes) != num_nodes:
        raise ValueError(f"Expected {num_nodes} nodes, but found {len(raw_nodes)}.")

    if len(raw_lines) != num_nodes - 1:
        raise ValueError(
            "TSSP network model expects a radial network with "
            f"{num_nodes - 1} lines, but found {len(raw_lines)}."
        )

    slack_nodes = raw_nodes.loc[raw_nodes["is_slack"], "node_id"].tolist()

    if slack_nodes != [SLACK_NODE_ID]:
        raise ValueError(
            f"Expected node {SLACK_NODE_ID} to be the only slack node, "
            f"but found slack nodes {slack_nodes}."
        )

    valid_node_ids = set(raw_nodes["node_id"].astype(int))
    line_node_ids = set(raw_lines[FROM_COLUMN].astype(int)) | set(raw_lines[TO_COLUMN].astype(int))

    unknown_node_ids = line_node_ids - valid_node_ids
    if unknown_node_ids:
        raise ValueError(f"Lines reference unknown node IDs: {sorted(unknown_node_ids)}.")


def build_reduced_incidence_matrix(raw_lines, num_nodes):
    line_connections = {
        i + 1: (
            int(raw_lines.loc[i, FROM_COLUMN]),
            int(raw_lines.loc[i, TO_COLUMN]),
        )
        for i in range(len(raw_lines))
    }

    A_full = np.zeros((len(line_connections), num_nodes))

    for branch, (start_node, end_node) in line_connections.items():
        A_full[branch - 1, start_node - 1] = 1
        A_full[branch - 1, end_node - 1] = -1

    return np.delete(A_full, SLACK_BUS_INDEX, axis=1)


def build_voltage_sensitivity_matrices(raw_lines, A_inv):
    R = raw_lines[R_COLUMN].to_numpy() * IMPEDANCE_SCALING_FACTOR
    X = raw_lines[X_COLUMN].to_numpy() * IMPEDANCE_SCALING_FACTOR

    R_diag = np.diag(R)
    X_diag = np.diag(X)

    R_prime = 2 * A_inv @ R_diag @ A_inv.T
    X_prime = 2 * A_inv @ X_diag @ A_inv.T

    return R_prime, X_prime


def build_load_profiles(raw_nodes, raw_demand, T):
    if raw_demand.shape[1] < T:
        raise ValueError(f"Demand profile length {raw_demand.shape[1]} is shorter than T={T}.")

    load_nodes = raw_nodes.loc[~raw_nodes["is_slack"]].copy()
    load_nodes = load_nodes.sort_values("node_id")

    load_node_ids = load_nodes["node_id"].astype(int).tolist()
    demand_node_ids = raw_demand.index.astype(int).tolist()

    if demand_node_ids != load_node_ids:
        raise ValueError(
            "Demand-profile node IDs do not match load-node IDs. "
            f"Demand nodes: {demand_node_ids}; "
            f"load nodes: {load_node_ids}."
        )

    demand_profile_pu = raw_demand.to_numpy()
    PD_base = load_nodes[PD_BASE_COLUMN].to_numpy().reshape(-1, 1)

    P_load = -PD_base * demand_profile_pu

    return P_load, PD_base


def build_system_data(connection, num_nodes, T):
    """Build network data required by the TSSP model."""
    raw_nodes, raw_lines, raw_demand = read_network_data(connection, T)

    _validate_network_data(raw_nodes, raw_lines, num_nodes)

    A = build_reduced_incidence_matrix(raw_lines, num_nodes)

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Reduced incidence matrix is singular; check network connectivity and topology."
        ) from error

    R_prime, X_prime = build_voltage_sensitivity_matrices(raw_lines, A_inv)
    P_load, PD_base = build_load_profiles(raw_nodes, raw_demand, T)

    return SystemData(
        A=A,
        R_prime=R_prime,
        X_prime=X_prime,
        P_load=P_load,
        PD_base=PD_base,
    )

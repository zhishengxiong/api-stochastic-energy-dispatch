from dataclasses import dataclass

import numpy as np
from gurobipy import GRB, Model, quicksum

SLACK_NODE_ID = 1


@dataclass
class ParamsData:
    T: int
    power_factor: float
    penalty_cost: float
    num_non_slack_nodes: int
    scenario_probability: float
    v0: float
    v_up: float
    v_low: float
    P_flow_up: float
    P_flow_low: float


@dataclass
class ModelData:
    system_data: object
    ders_data: object
    params_data: ParamsData


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _validate_unique_nodes(nodes, name: str) -> None:
    if len(nodes) != len(set(nodes)):
        raise ValueError(
            f"{name} contains multiple devices assigned to the same node. "
            "The current TSSP model supports at most one device of each type per node."
        )


def _validate_model_inputs(
    system_data,
    ders_data,
    empirical_samples_info,
    num_nodes,
    T,
) -> None:
    _validate_positive_int(num_nodes, "num_nodes")
    _validate_positive_int(T, "T")

    if num_nodes <= 1:
        raise ValueError("num_nodes must include at least one non-slack node.")

    num_non_slack_nodes = num_nodes - 1
    expected_square_shape = (
        num_non_slack_nodes,
        num_non_slack_nodes,
    )

    if system_data.A.shape != expected_square_shape:
        raise ValueError(
            f"A must have shape {expected_square_shape}, but found {system_data.A.shape}."
        )

    if system_data.R_prime.shape != expected_square_shape:
        raise ValueError(
            f"R_prime must have shape {expected_square_shape}, "
            f"but found {system_data.R_prime.shape}."
        )

    if system_data.X_prime.shape != expected_square_shape:
        raise ValueError(
            f"X_prime must have shape {expected_square_shape}, "
            f"but found {system_data.X_prime.shape}."
        )

    expected_load_shape = (num_non_slack_nodes, T)
    if system_data.P_load.shape != expected_load_shape:
        raise ValueError(
            f"P_load must have shape {expected_load_shape}, but found {system_data.P_load.shape}."
        )

    if system_data.PD_base.shape != (num_non_slack_nodes, 1):
        raise ValueError(
            f"PD_base must have shape ({num_non_slack_nodes}, 1), "
            f"but found {system_data.PD_base.shape}."
        )

    scenarios = empirical_samples_info.get("empirical_samples")
    num_scenarios = empirical_samples_info.get("num_empirical_samples")

    if scenarios is None or num_scenarios is None:
        raise KeyError(
            "empirical_samples_info must contain 'empirical_samples' and 'num_empirical_samples'."
        )

    if num_scenarios <= 0:
        raise ValueError("At least one empirical scenario is required.")

    expected_scenario_shape = (num_scenarios, 2 * T)
    if scenarios.shape != expected_scenario_shape:
        raise ValueError(
            f"Empirical scenarios must have shape {expected_scenario_shape}, "
            f"but found {scenarios.shape}."
        )

    if not np.isfinite(scenarios).all():
        raise ValueError("Empirical scenarios contain NaN or infinite values.")

    valid_device_nodes = set(range(SLACK_NODE_ID + 1, num_nodes + 1))

    device_groups = {
        "generators": ders_data.G_node,
        "PV units": ders_data.PV_node,
        "storage units": ders_data.ESS_node,
        "reserve units": ders_data.R_node,
    }

    for name, nodes in device_groups.items():
        _validate_unique_nodes(nodes, name)

        invalid_nodes = set(nodes) - valid_device_nodes
        if invalid_nodes:
            raise ValueError(f"{name} contain invalid or slack node IDs: {sorted(invalid_nodes)}.")

    occupied_nodes = {}
    for device_type, nodes in device_groups.items():
        for node_id in nodes:
            if node_id in occupied_nodes:
                raise ValueError(
                    f"Node {node_id} contains both {occupied_nodes[node_id]} "
                    f"and {device_type}. The current nodal-injection logic "
                    "uses mutually exclusive device branches."
                )
            occupied_nodes[node_id] = device_type


def _raise_if_not_optimal(model: Model) -> None:
    if model.status == GRB.OPTIMAL:
        return

    status_names = {
        GRB.LOADED: "LOADED",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }

    status_name = status_names.get(model.status, "UNKNOWN")
    raise RuntimeError(
        f"TSSP optimization failed with Gurobi status {status_name} ({model.status})."
    )


def solve_tssp(
    system_data,
    ders_data,
    empirical_samples_info,
    num_nodes,
    T,
):
    """Build and solve the two-stage stochastic scheduling model."""
    # 1. VALIDATE INPUTS
    _validate_model_inputs(
        system_data,
        ders_data,
        empirical_samples_info,
        num_nodes,
        T,
    )

    # 2. PREPARE DATA
    model_data = build_model_data(
        system_data,
        ders_data,
        empirical_samples_info,
        num_nodes,
        T,
    )

    scenarios = empirical_samples_info["empirical_samples"]

    # 3. CONSTRUCT OPTIMIZATION MODEL
    model = Model("tssp_baseline")
    model.setParam("OutputFlag", 0)

    # 4. FIRST-STAGE PROBLEM
    first_stage_vars = add_first_stage_vars(model, model_data)
    add_first_stage_cons(
        model,
        model_data,
        first_stage_vars,
    )
    first_stage_obj = add_first_stage_obj(
        model_data,
        first_stage_vars,
    )

    # 5. SECOND-STAGE RECOURSE PROBLEMS
    expected_recourse_cost = 0

    for scenario_idx, scenario in enumerate(scenarios):
        recourse_cost = build_recourse_problem(
            model,
            model_data,
            first_stage_vars,
            scenario_idx,
            scenario,
        )
        expected_recourse_cost += model_data.params_data.scenario_probability * recourse_cost

    # 6. OBJECTIVE FUNCTION
    total_cost = first_stage_obj + expected_recourse_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # 7. SOLVE MODEL
    model.optimize()
    _raise_if_not_optimal(model)

    # 8. RETRIEVE RESULTS
    x_hat = retrieve_x_hat(first_stage_vars)

    return {
        "cost": model.ObjVal,
        "x_hat": x_hat,
    }


def build_model_data(
    system_data,
    ders_data,
    empirical_samples_info,
    num_nodes,
    T,
):
    p_flow_limit = 3500
    base_voltage_kv = 12.66
    power_factor = 0.9
    penalty_cost = 30

    num_non_slack_nodes = num_nodes - 1
    num_scenarios = empirical_samples_info["num_empirical_samples"]
    scenario_probability = 1 / num_scenarios

    v0 = base_voltage_kv * base_voltage_kv
    v_up = 1.05**2 * v0
    v_low = 0.95**2 * v0

    P_flow_up = p_flow_limit
    P_flow_low = -P_flow_up

    params_data = ParamsData(
        T=T,
        power_factor=power_factor,
        penalty_cost=penalty_cost,
        num_non_slack_nodes=num_non_slack_nodes,
        scenario_probability=scenario_probability,
        v0=v0,
        v_up=v_up,
        v_low=v_low,
        P_flow_up=P_flow_up,
        P_flow_low=P_flow_low,
    )

    return ModelData(
        system_data=system_data,
        ders_data=ders_data,
        params_data=params_data,
    )


def add_first_stage_vars(model, model_data):
    params_data = model_data.params_data
    ders_data = model_data.ders_data

    T = params_data.T
    num_non_slack_nodes = params_data.num_non_slack_nodes

    G_node = ders_data.G_node
    G_pmax = ders_data.G_pmax

    R_node = ders_data.R_node

    ESS_node = ders_data.ESS_node
    ESS_capacity = ders_data.ESS_capacity

    first_stage_vars = {
        "G_p": model.addMVar(
            (len(G_node), T),
            lb=0,
            ub=G_pmax,
            vtype=GRB.CONTINUOUS,
            name="G_p",
        ),
        "G_u": model.addMVar(
            (len(G_node), T),
            vtype=GRB.BINARY,
            name="G_u",
        ),
        "G_s": model.addMVar(
            (len(G_node), T),
            lb=0,
            vtype=GRB.BINARY,
            name="G_s",
        ),
        "R_pmax": model.addMVar(
            (len(R_node), T),
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="R_pmax",
        ),
        "ESS_pch": model.addMVar(
            (len(ESS_node), T),
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="ESS_ch",
        ),
        "ESS_pdis": model.addMVar(
            (len(ESS_node), T),
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="ESS_dis",
        ),
        "ESS_E": model.addMVar(
            (len(ESS_node), T),
            lb=ESS_capacity * 0.2,
            ub=ESS_capacity,
            vtype=GRB.CONTINUOUS,
            name="ESS_E",
        ),
        "ESS_u": model.addMVar(
            (len(ESS_node), T),
            vtype=GRB.BINARY,
            name="ESS_u",
        ),
        "P_flow": model.addMVar(
            (num_non_slack_nodes, T),
            lb=params_data.P_flow_low,
            ub=params_data.P_flow_up,
            name="P_flow",
        ),
        "P_net": model.addMVar(
            (num_non_slack_nodes, T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name="P_net",
        ),
        "flow_buy": model.addMVar(
            T,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="flow_buy",
        ),
        "flow_sell": model.addMVar(
            T,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="flow_sell",
        ),
        "flow_u": model.addMVar(
            T,
            vtype=GRB.BINARY,
            name="flow_u",
        ),
    }

    return first_stage_vars


def add_first_stage_cons(
    model,
    model_data,
    first_stage_vars,
):
    params_data = model_data.params_data
    system_data = model_data.system_data
    ders_data = model_data.ders_data

    T = params_data.T
    num_non_slack_nodes = params_data.num_non_slack_nodes

    A = system_data.A
    P_load = system_data.P_load

    G_node = ders_data.G_node
    G_pmax = ders_data.G_pmax
    G_pmin = ders_data.G_pmin
    G_up_limit = ders_data.G_up_limit
    G_dn_limit = ders_data.G_dn_limit

    PV_node = ders_data.PV_node
    PV = ders_data.PV

    ESS_node = ders_data.ESS_node
    ESS_pmax = ders_data.ESS_pmax
    ESS_Eini = ders_data.ESS_Eini
    ESS_eff = ders_data.ESS_eff

    G_p = first_stage_vars["G_p"]
    G_u = first_stage_vars["G_u"]
    G_s = first_stage_vars["G_s"]

    ESS_pch = first_stage_vars["ESS_pch"]
    ESS_pdis = first_stage_vars["ESS_pdis"]
    ESS_E = first_stage_vars["ESS_E"]
    ESS_u = first_stage_vars["ESS_u"]

    P_flow = first_stage_vars["P_flow"]
    P_net = first_stage_vars["P_net"]

    flow_buy = first_stage_vars["flow_buy"]
    flow_sell = first_stage_vars["flow_sell"]
    flow_u = first_stage_vars["flow_u"]

    # 4.1 Generator commitment, dispatch limits, and ramping
    for t in range(T):
        for g in range(len(G_node)):
            model.addConstr(
                G_p[g, t] <= G_u[g, t] * G_pmax[g, t],
                name=f"G_pmax_rule_{g}_{t}",
            )
            model.addConstr(
                G_p[g, t] >= G_u[g, t] * G_pmin[g, t],
                name=f"G_pmin_rule_{g}_{t}",
            )

    for t in range(T):
        if t == 0:
            for g in range(len(G_node)):
                model.addConstr(
                    G_s[g, t] == G_u[g, t],
                    name=f"G_on_status_{g}_{t}",
                )
        else:
            for g in range(len(G_node)):
                model.addConstr(
                    G_p[g, t] - G_p[g, t - 1] <= G_up_limit[g],
                    name=f"G_p_up_rule_{g}_{t}",
                )
                model.addConstr(
                    G_p[g, t - 1] - G_p[g, t] <= G_dn_limit[g],
                    name=f"G_p_dn_rule_{g}_{t}",
                )
                model.addConstr(
                    G_s[g, t] >= G_u[g, t] - G_u[g, t - 1],
                    name=f"G_on_status_{g}_{t}",
                )

    # 4.2 ESS charging/discharging and energy balance
    for t in range(T):
        for e in range(len(ESS_node)):
            model.addConstr(
                ESS_pch[e, t] <= ESS_u[e, t] * ESS_pmax[e],
                name=f"ESS_ch_limit_{e}_{t}",
            )
            model.addConstr(
                ESS_pdis[e, t] <= (1 - ESS_u[e, t]) * ESS_pmax[e],
                name=f"ESS_dis_limit_{e}_{t}",
            )

    for t in range(T):
        if t == 0:
            for e in range(len(ESS_node)):
                model.addConstr(
                    ESS_E[e, t]
                    == ESS_Eini[e] + ESS_eff[e] * ESS_pch[e, t] - ESS_pdis[e, t] / ESS_eff[e],
                    name=f"ESS_energy_rule_{e}_{t}",
                )
        else:
            for e in range(len(ESS_node)):
                model.addConstr(
                    ESS_E[e, t]
                    == ESS_E[e, t - 1] + ESS_eff[e] * ESS_pch[e, t] - ESS_pdis[e, t] / ESS_eff[e],
                    name=f"ESS_energy_rule_{e}_{t}",
                )

    # 4.3 Day-ahead nodal active-power injection
    for n in range(num_non_slack_nodes):
        node_id = n + 2

        if node_id in G_node:
            g = G_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net[n, t] == P_load[n, t] + G_p[g, t],
                    name=f"P_net_rule_{n}_{t}",
                )

        elif node_id in PV_node:
            pv = PV_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net[n, t] == P_load[n, t] + PV[pv, t],
                    name=f"P_net_rule_{n}_{t}",
                )

        elif node_id in ESS_node:
            e = ESS_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net[n, t] == P_load[n, t] + ESS_pdis[e, t] - ESS_pch[e, t],
                    name=f"P_net_rule_{n}_{t}",
                )

        else:
            for t in range(T):
                model.addConstr(
                    P_net[n, t] == P_load[n, t],
                    name=f"P_net_rule_{n}_{t}",
                )

    # 4.4 Day-ahead active-power balance and substation exchange
    for t in range(T):
        model.addConstr(
            A.T @ P_flow[:, t] == P_net[:, t],
            name=f"P_Ban_{t}",
        )
        model.addConstr(
            flow_buy[t] <= params_data.P_flow_up * flow_u[t],
            name=f"P_buy_{t}",
        )
        model.addConstr(
            flow_sell[t] <= params_data.P_flow_up * (1 - flow_u[t]),
            name=f"P_sell_{t}",
        )
        model.addConstr(
            P_flow[0, t] == flow_buy[t] - flow_sell[t],
            name=f"substation_{t}",
        )


def add_first_stage_obj(
    model_data,
    first_stage_vars,
):
    params_data = model_data.params_data
    ders_data = model_data.ders_data

    T = params_data.T

    G_cost = ders_data.G_cost
    G_Ucost = ders_data.G_Ucost
    R_cost = ders_data.R_cost
    electricity_price = ders_data.electricity_price

    G_p = first_stage_vars["G_p"]
    G_s = first_stage_vars["G_s"]
    R_pmax = first_stage_vars["R_pmax"]
    flow_buy = first_stage_vars["flow_buy"]
    flow_sell = first_stage_vars["flow_sell"]

    return (
        quicksum(G_p[:, t] for t in range(T)) @ G_cost
        + quicksum(G_s[:, t] for t in range(T)) @ G_Ucost
        + quicksum(R_pmax[:, t] for t in range(T)) @ R_cost
        + quicksum(flow_buy[t] * electricity_price[t] for t in range(T))
        - quicksum(flow_sell[t] * electricity_price[t] * 0.8 for t in range(T))
    )


def build_recourse_problem(
    model,
    model_data,
    first_stage_vars,
    scenario_idx,
    scenario,
):
    recourse_vars = add_recourse_vars(
        model,
        model_data,
        scenario_idx,
    )

    add_recourse_cons(
        model,
        model_data,
        first_stage_vars,
        recourse_vars,
        scenario_idx,
        scenario,
    )

    return add_recourse_obj(
        model_data,
        first_stage_vars,
        recourse_vars,
    )


def add_recourse_vars(
    model,
    model_data,
    scenario_idx,
):
    params_data = model_data.params_data
    ders_data = model_data.ders_data

    T = params_data.T
    num_non_slack_nodes = params_data.num_non_slack_nodes

    G_node = ders_data.G_node
    G_pmax = ders_data.G_pmax
    R_node = ders_data.R_node
    PV_node = ders_data.PV_node

    recourse_vars = {
        "G_q": model.addMVar(
            (len(G_node), T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"G_q_{scenario_idx}",
        ),
        "G_p_reg": model.addMVar(
            (len(G_node), T),
            lb=0,
            ub=G_pmax,
            vtype=GRB.CONTINUOUS,
            name=f"G_p_reg_{scenario_idx}",
        ),
        "R_p": model.addMVar(
            (len(R_node), T),
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"R_p_{scenario_idx}",
        ),
        "R_q": model.addMVar(
            (len(R_node), T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"R_q_{scenario_idx}",
        ),
        "P_flow_2S": model.addMVar(
            (num_non_slack_nodes, T),
            lb=params_data.P_flow_low,
            ub=params_data.P_flow_up,
            name=f"P_flow_2S_{scenario_idx}",
        ),
        "Q_flow": model.addMVar(
            (num_non_slack_nodes, T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name=f"Q_flow_{scenario_idx}",
        ),
        "v": model.addMVar(
            (num_non_slack_nodes, T),
            lb=params_data.v_low,
            ub=params_data.v_up,
            name=f"v_{scenario_idx}",
        ),
        "P_net_2S": model.addMVar(
            (num_non_slack_nodes, T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name=f"P_net_2S_{scenario_idx}",
        ),
        "Q_net": model.addMVar(
            (num_non_slack_nodes, T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name=f"Q_net_{scenario_idx}",
        ),
        "P_load_real": model.addMVar(
            (num_non_slack_nodes, T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"P_load_real_{scenario_idx}",
        ),
        "Q_load": model.addMVar(
            (num_non_slack_nodes, T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"Q_load_{scenario_idx}",
        ),
        "PV_real": model.addMVar(
            (len(PV_node), T),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"PV_real_{scenario_idx}",
        ),
        "flow_buy_2S": model.addMVar(
            T,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"flow_buy_2S_{scenario_idx}",
        ),
        "flow_sell_2S": model.addMVar(
            T,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"flow_sell_2S_{scenario_idx}",
        ),
        "load_slack_2S": model.addMVar(
            T,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"load_slack_2S_{scenario_idx}",
        ),
        "load_surplus_2S": model.addMVar(
            T,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"load_surplus_2S_{scenario_idx}",
        ),
        "pv_slack_2S": model.addMVar(
            T,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"pv_slack_2S_{scenario_idx}",
        ),
        "pv_surplus_2S": model.addMVar(
            T,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"pv_surplus_2S_{scenario_idx}",
        ),
    }

    return recourse_vars


def add_recourse_cons(
    model,
    model_data,
    first_stage_vars,
    recourse_vars,
    scenario_idx,
    scenario,
):
    params_data = model_data.params_data
    system_data = model_data.system_data
    ders_data = model_data.ders_data

    T = params_data.T
    num_non_slack_nodes = params_data.num_non_slack_nodes

    A = system_data.A
    R_prime = system_data.R_prime
    X_prime = system_data.X_prime
    PD_base = system_data.PD_base

    G_node = ders_data.G_node
    G_pmax = ders_data.G_pmax
    G_pmin = ders_data.G_pmin
    G_up_limit = ders_data.G_up_limit
    G_dn_limit = ders_data.G_dn_limit
    G_qmax = ders_data.G_qmax
    G_qmin = ders_data.G_qmin

    PV_node = ders_data.PV_node
    PV_cap = ders_data.PV_cap

    ESS_node = ders_data.ESS_node
    R_node = ders_data.R_node

    G_u = first_stage_vars["G_u"]
    R_pmax = first_stage_vars["R_pmax"]
    ESS_pch = first_stage_vars["ESS_pch"]
    ESS_pdis = first_stage_vars["ESS_pdis"]
    flow_buy = first_stage_vars["flow_buy"]
    flow_sell = first_stage_vars["flow_sell"]

    G_q = recourse_vars["G_q"]
    G_p_reg = recourse_vars["G_p_reg"]
    R_p = recourse_vars["R_p"]
    R_q = recourse_vars["R_q"]
    P_flow_2S = recourse_vars["P_flow_2S"]
    Q_flow = recourse_vars["Q_flow"]
    v = recourse_vars["v"]
    P_net_2S = recourse_vars["P_net_2S"]
    Q_net = recourse_vars["Q_net"]
    P_load_real = recourse_vars["P_load_real"]
    Q_load = recourse_vars["Q_load"]
    PV_real = recourse_vars["PV_real"]
    flow_buy_2S = recourse_vars["flow_buy_2S"]
    flow_sell_2S = recourse_vars["flow_sell_2S"]
    load_slack_2S = recourse_vars["load_slack_2S"]
    load_surplus_2S = recourse_vars["load_surplus_2S"]
    pv_slack_2S = recourse_vars["pv_slack_2S"]
    pv_surplus_2S = recourse_vars["pv_surplus_2S"]

    # 5.1 Generator active-power regulation
    for t in range(T):
        for g in range(len(G_node)):
            model.addConstr(
                G_p_reg[g, t] <= G_u[g, t] * G_pmax[g, t],
                name=f"G_reg_pmax_rule_{scenario_idx}_{g}_{t}",
            )
            model.addConstr(
                G_p_reg[g, t] >= G_u[g, t] * G_pmin[g, t],
                name=f"G_reg_pmin_rule_{scenario_idx}_{g}_{t}",
            )

    for t in range(1, T):
        model.addConstr(
            G_p_reg[:, t] - G_p_reg[:, t - 1] <= G_up_limit,
            name=f"G_reg_up_rule_{scenario_idx}_{t}",
        )
        model.addConstr(
            G_p_reg[:, t - 1] - G_p_reg[:, t] <= G_dn_limit,
            name=f"G_reg_dn_rule_{scenario_idx}_{t}",
        )

    # 5.2 Generator reactive-power limits
    for t in range(T):
        for g in range(len(G_node)):
            model.addConstr(
                G_q[g, t] <= G_u[g, t] * G_qmax[g, t],
                name=f"G_qmax_rule_{scenario_idx}_{g}_{t}",
            )
            model.addConstr(
                G_q[g, t] >= G_u[g, t] * G_qmin[g, t],
                name=f"G_qmin_rule_{scenario_idx}_{g}_{t}",
            )

    # 5.3 Reserve active/reactive regulation
    for t in range(T):
        for r in range(len(R_node)):
            model.addConstr(
                R_p[r, t] <= R_pmax[r, t],
                name=f"R_pmax_rule_{scenario_idx}_{r}_{t}",
            )
            model.addConstr(
                R_q[r, t] <= 0.8 * R_pmax[r, t],
                name=f"R_qmax_rule_{scenario_idx}_{r}_{t}",
            )
            model.addConstr(
                R_q[r, t] >= -0.8 * R_pmax[r, t],
                name=f"R_qmin_rule_{scenario_idx}_{r}_{t}",
            )

    for t in range(1, T):
        model.addConstr(
            R_p[:, t] - R_p[:, t - 1] <= 0.2 * R_pmax[:, t],
            name=f"R_up_rule_{scenario_idx}_{t}",
        )
        model.addConstr(
            R_p[:, t - 1] - R_p[:, t] <= 0.2 * R_pmax[:, t],
            name=f"R_dn_rule_{scenario_idx}_{t}",
        )

    # 5.4 Real-time nodal active/reactive-power injection
    for n in range(num_non_slack_nodes):
        node_id = n + 2

        if node_id in G_node:
            g = G_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + G_p_reg[g, t],
                    name=f"P_net_2S_rule_{scenario_idx}_{n}_{t}",
                )
                model.addConstr(
                    Q_net[n, t] == Q_load[n, t] + G_q[g, t],
                    name=f"Q_net_rule_{scenario_idx}_{n}_{t}",
                )

        elif node_id in R_node:
            r = R_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + R_p[r, t],
                    name=f"P_net_2S_rule_{scenario_idx}_{n}_{t}",
                )
                model.addConstr(
                    Q_net[n, t] == Q_load[n, t] + R_q[r, t],
                    name=f"Q_net_rule_{scenario_idx}_{n}_{t}",
                )

        elif node_id in PV_node:
            pv = PV_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + PV_real[pv, t],
                    name=f"P_net_2S_rule_{scenario_idx}_{n}_{t}",
                )
                model.addConstr(
                    Q_net[n, t] == Q_load[n, t],
                    name=f"Q_net_rule_{scenario_idx}_{n}_{t}",
                )

        elif node_id in ESS_node:
            e = ESS_node.index(node_id)
            for t in range(T):
                model.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + ESS_pdis[e, t] - ESS_pch[e, t],
                    name=f"P_net_2S_rule_{scenario_idx}_{n}_{t}",
                )
                model.addConstr(
                    Q_net[n, t] == Q_load[n, t],
                    name=f"Q_net_rule_{scenario_idx}_{n}_{t}",
                )

        else:
            for t in range(T):
                model.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t],
                    name=f"P_net_2S_rule_{scenario_idx}_{n}_{t}",
                )
                model.addConstr(
                    Q_net[n, t] == Q_load[n, t],
                    name=f"Q_net_rule_{scenario_idx}_{n}_{t}",
                )

    # 5.5 Real-time power flow, voltage, and substation exchange
    for t in range(T):
        model.addConstr(
            A.T @ P_flow_2S[:, t] == P_net_2S[:, t],
            name=f"P_Ban_2S_{scenario_idx}_{t}",
        )
        model.addConstr(
            A.T @ Q_flow[:, t] == Q_net[:, t],
            name=f"Q_Ban_{scenario_idx}_{t}",
        )
        model.addConstr(
            v[:, t] == params_data.v0 + R_prime @ P_net_2S[:, t] + X_prime @ Q_net[:, t],
            name=f"v_rule_{scenario_idx}_{t}",
        )

        model.addConstr(
            flow_buy_2S[t] <= 1.2 * flow_buy[t],
            name=f"P_buy_2S_up_limit_{scenario_idx}_{t}",
        )
        model.addConstr(
            flow_buy_2S[t] >= flow_buy[t],
            name=f"P_buy_2S_dn_limit_{scenario_idx}_{t}",
        )
        model.addConstr(
            flow_sell_2S[t] <= 1.2 * flow_sell[t],
            name=f"P_sell_2S_up_limit_{scenario_idx}_{t}",
        )
        model.addConstr(
            flow_sell_2S[t] >= flow_sell[t],
            name=f"P_sell_2S_dn_limit_{scenario_idx}_{t}",
        )
        model.addConstr(
            P_flow_2S[0, t] == flow_buy_2S[t] - flow_sell_2S[t],
            name=f"substation_2S_{scenario_idx}_{t}",
        )

    # 5.6 Uncertainty realization and feasibility adjustment
    pv_scale = scenario[:T]
    load_scale = scenario[T : 2 * T]

    for t in range(T):
        model.addConstr(
            P_load_real[:, t]
            == -PD_base[:, 0] * load_scale[t] + load_slack_2S[t] - load_surplus_2S[t],
            name=f"P_load_real_rule_{scenario_idx}_{t}",
        )
        model.addConstr(
            PV_real[:, t] == PV_cap[:, 0] * pv_scale[t] + pv_slack_2S[t] - pv_surplus_2S[t],
            name=f"PV_real_rule_{scenario_idx}_{t}",
        )
        model.addConstr(
            Q_load[:, t] == P_load_real[:, t] * params_data.power_factor,
            name=f"Q_load_rule_{scenario_idx}_{t}",
        )


def add_recourse_obj(
    model_data,
    first_stage_vars,
    recourse_vars,
):
    params_data = model_data.params_data
    ders_data = model_data.ders_data

    T = params_data.T
    num_non_slack_nodes = params_data.num_non_slack_nodes

    G_cost = ders_data.G_cost
    electricity_price = ders_data.electricity_price
    PV_node = ders_data.PV_node

    G_p = first_stage_vars["G_p"]
    flow_buy = first_stage_vars["flow_buy"]
    flow_sell = first_stage_vars["flow_sell"]

    G_p_reg = recourse_vars["G_p_reg"]
    flow_buy_2S = recourse_vars["flow_buy_2S"]
    flow_sell_2S = recourse_vars["flow_sell_2S"]
    load_slack_2S = recourse_vars["load_slack_2S"]
    load_surplus_2S = recourse_vars["load_surplus_2S"]
    pv_slack_2S = recourse_vars["pv_slack_2S"]
    pv_surplus_2S = recourse_vars["pv_surplus_2S"]

    return (
        quicksum((G_p_reg[:, t] - G_p[:, t]) @ G_cost for t in range(T))
        + quicksum((flow_buy_2S[t] - flow_buy[t]) * electricity_price[t] * 1.4 for t in range(T))
        - quicksum((flow_sell_2S[t] - flow_sell[t]) * electricity_price[t] * 0.6 for t in range(T))
        + quicksum(
            (
                (load_slack_2S[t] + load_surplus_2S[t]) * num_non_slack_nodes
                + (pv_slack_2S[t] + pv_surplus_2S[t]) * len(PV_node)
            )
            for t in range(T)
        )
        * params_data.penalty_cost
    )


def retrieve_x_hat(first_stage_vars):
    return {
        "G_p": first_stage_vars["G_p"].X,
        "G_u": first_stage_vars["G_u"].X,
        "G_s": first_stage_vars["G_s"].X,
        "R_pmax": first_stage_vars["R_pmax"].X,
        "ESS_pch": first_stage_vars["ESS_pch"].X,
        "ESS_pdis": first_stage_vars["ESS_pdis"].X,
        "ESS_E": first_stage_vars["ESS_E"].X,
        "ESS_u": first_stage_vars["ESS_u"].X,
        "P_flow": first_stage_vars["P_flow"].X,
        "P_net": first_stage_vars["P_net"].X,
        "flow_buy": first_stage_vars["flow_buy"].X,
        "flow_sell": first_stage_vars["flow_sell"].X,
        "flow_u": first_stage_vars["flow_u"].X,
    }

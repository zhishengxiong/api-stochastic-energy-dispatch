from dataclasses import dataclass

import numpy as np

from stochastic_energy_dispatch import input_repository as repository

NODE_COLUMN = "node_id"
GENERATOR_ID_COLUMN = "generator_id"
STORAGE_ID_COLUMN = "storage_id"
PV_ID_COLUMN = "pv_id"
RESERVE_ID_COLUMN = "reserve_id"
TIME_COLUMN = "time_step"

PMAX_COLUMN = "p_max"
PMIN_COLUMN = "p_min"
QMAX_COLUMN = "q_max"
QMIN_COLUMN = "q_min"
RU_COLUMN = "ramp_up"
RD_COLUMN = "ramp_down"
COST_COLUMN = "generation_cost"
UCOST_COLUMN = "startup_cost"

POWER_COLUMN = "power_capacity"
ENERGY_COLUMN = "energy_capacity"
EINI_COLUMN = "initial_energy"
EFF_COLUMN = "efficiency"

PRICE_COLUMN = "price"
PV_COLUMN = "pv_scale"
PV_CAP_COLUMN = "capacity"

RESERVE_COST_COLUMN = "reserve_cost"


@dataclass
class DERsData:
    G_id: list
    G_node: list
    G_pmax: np.ndarray
    G_pmin: np.ndarray
    G_up_limit: np.ndarray
    G_dn_limit: np.ndarray
    G_qmax: np.ndarray
    G_qmin: np.ndarray
    G_cost: np.ndarray
    G_Ucost: np.ndarray
    electricity_price: np.ndarray
    PV_id: list
    PV_node: list
    PV: np.ndarray
    PV_cap: np.ndarray
    ESS_id: list
    ESS_node: list
    ESS_pmax: np.ndarray
    ESS_capacity: np.ndarray
    ESS_Eini: np.ndarray
    ESS_eff: np.ndarray
    R_id: list
    R_node: list
    R_cost: np.ndarray


def read_ders_data(connection, T):
    raw_generators = repository.fetch_generators(connection)
    raw_ess = repository.fetch_energy_storage(connection)
    raw_prices = repository.fetch_electricity_prices(connection, T)
    raw_pv_units = repository.fetch_pv_units(connection)
    raw_pv_predictive = repository.fetch_pv_forecasts(connection, T)
    raw_reserve = repository.fetch_reserve_units(connection)

    return (
        raw_generators,
        raw_ess,
        raw_prices,
        raw_pv_units,
        raw_pv_predictive,
        raw_reserve,
    )


def _validate_time_profile(dataframe, T, profile_name):
    expected_time_steps = list(range(T))
    actual_time_steps = dataframe[TIME_COLUMN].astype(int).tolist()

    if actual_time_steps != expected_time_steps:
        raise ValueError(
            f"{profile_name} must contain exactly time steps 0 to {T - 1}. "
            f"Found: {actual_time_steps}."
        )


def expand_to_time_horizon(values, T):
    values = values.to_numpy().reshape(-1, 1)
    return np.tile(values, (1, T))


def build_generator_data(raw_generators, T):
    G_id = raw_generators[GENERATOR_ID_COLUMN].astype(int).tolist()
    G_node = raw_generators[NODE_COLUMN].astype(int).tolist()

    G_pmax = expand_to_time_horizon(raw_generators[PMAX_COLUMN], T)
    G_pmin = expand_to_time_horizon(raw_generators[PMIN_COLUMN], T)
    G_qmax = expand_to_time_horizon(raw_generators[QMAX_COLUMN], T)
    G_qmin = expand_to_time_horizon(raw_generators[QMIN_COLUMN], T)

    G_up_limit = raw_generators[RU_COLUMN].to_numpy()
    G_dn_limit = raw_generators[RD_COLUMN].to_numpy()
    G_cost = raw_generators[COST_COLUMN].to_numpy()
    G_Ucost = raw_generators[UCOST_COLUMN].to_numpy()

    return (
        G_id,
        G_node,
        G_pmax,
        G_pmin,
        G_up_limit,
        G_dn_limit,
        G_qmax,
        G_qmin,
        G_cost,
        G_Ucost,
    )


def build_ess_data(raw_ess):
    ESS_id = raw_ess[STORAGE_ID_COLUMN].astype(int).tolist()
    ESS_node = raw_ess[NODE_COLUMN].astype(int).tolist()

    ESS_pmax = raw_ess[POWER_COLUMN].to_numpy().reshape(-1, 1)
    ESS_capacity = raw_ess[ENERGY_COLUMN].to_numpy().reshape(-1, 1)
    ESS_Eini = raw_ess[EINI_COLUMN].to_numpy().reshape(-1, 1)
    ESS_eff = raw_ess[EFF_COLUMN].to_numpy().reshape(-1, 1)

    return (
        ESS_id,
        ESS_node,
        ESS_pmax,
        ESS_capacity,
        ESS_Eini,
        ESS_eff,
    )


def build_price_profile(raw_prices, T):
    _validate_time_profile(raw_prices, T, "Electricity price profile")
    return np.round(raw_prices[PRICE_COLUMN].to_numpy(), 2)


def build_pv_data(raw_pv_units, raw_pv_predictive, T):
    _validate_time_profile(raw_pv_predictive, T, "PV forecast profile")

    PV_id = raw_pv_units[PV_ID_COLUMN].astype(int).tolist()
    PV_node = raw_pv_units[NODE_COLUMN].astype(int).tolist()

    pv_profile_pu = raw_pv_predictive[PV_COLUMN].to_numpy()
    pv_cap = raw_pv_units[PV_CAP_COLUMN].to_numpy().reshape(-1, 1)
    PV = pv_cap * pv_profile_pu.reshape(1, -1)

    return PV_id, PV_node, PV, pv_cap


def build_reserve_data(raw_reserve):
    R_id = raw_reserve[RESERVE_ID_COLUMN].astype(int).tolist()
    R_node = raw_reserve[NODE_COLUMN].astype(int).tolist()
    R_cost = raw_reserve[RESERVE_COST_COLUMN].to_numpy()

    return R_id, R_node, R_cost


def build_ders_data(connection, T):
    """Build DER data required by the TSSP model."""
    (
        raw_generators,
        raw_ess,
        raw_prices,
        raw_pv_units,
        raw_pv_predictive,
        raw_reserve,
    ) = read_ders_data(connection, T)

    (
        G_id,
        G_node,
        G_pmax,
        G_pmin,
        G_up_limit,
        G_dn_limit,
        G_qmax,
        G_qmin,
        G_cost,
        G_Ucost,
    ) = build_generator_data(raw_generators, T)

    (
        ESS_id,
        ESS_node,
        ESS_pmax,
        ESS_capacity,
        ESS_Eini,
        ESS_eff,
    ) = build_ess_data(raw_ess)

    electricity_price = build_price_profile(raw_prices, T)

    PV_id, PV_node, PV, pv_cap = build_pv_data(
        raw_pv_units,
        raw_pv_predictive,
        T,
    )

    R_id, R_node, R_cost = build_reserve_data(raw_reserve)

    return DERsData(
        G_id=G_id,
        G_node=G_node,
        G_pmax=G_pmax,
        G_pmin=G_pmin,
        G_up_limit=G_up_limit,
        G_dn_limit=G_dn_limit,
        G_qmax=G_qmax,
        G_qmin=G_qmin,
        G_cost=G_cost,
        G_Ucost=G_Ucost,
        electricity_price=electricity_price,
        PV_id=PV_id,
        PV_node=PV_node,
        PV=PV,
        PV_cap=pv_cap,
        ESS_id=ESS_id,
        ESS_node=ESS_node,
        ESS_pmax=ESS_pmax,
        ESS_capacity=ESS_capacity,
        ESS_Eini=ESS_Eini,
        ESS_eff=ESS_eff,
        R_id=R_id,
        R_node=R_node,
        R_cost=R_cost,
    )

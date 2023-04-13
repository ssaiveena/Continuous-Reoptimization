import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
from scipy.optimize import differential_evolution as DE
import time
from numba import njit
import math

# functions to pull numpy arrays from dataframes

def reservoir_training_data(k, v, df, medians, df_demand, init_storage=False):
    dowy = df.dowy.values
    Q = df[k + '_inflow_cfs'].values
    K = v['capacity_taf'] * 1000
    # R_obs = df[k+'_outflow_cfs'].values
    # S_obs = df[k+'_storage_af'].values if K > 0 else np.zeros(dowy.size)
    Q_avg = medians[k + '_inflow_cfs'].values
    R_avg = medians[k + '_outflow_cfs'].values
    S_avg = medians[k + '_storage_af'].values if K > 0 else np.zeros(dowy.size)
    R_max = v['safe_release_cfs']
    if not init_storage:
        S0 = S_avg[-1]
    else:
        S0 = df[k + '_storage_af'].values[-1]

    return (dowy, Q, K, R_avg, S_avg, S0, R_max, df_demand.combined_demand.values)  # deleted R_obs after Q_avg

def reservoir_training_data_updated(k, v, df, medians, df_demand, df_opt, init_storage=False):
    dowy = df.dowy.values
    Q = df[k + '_inflow_cfs'].values
    K = v['capacity_taf'] * 1000
    # R_obs = df[k+'_outflow_cfs'].values
    # S_obs = df[k+'_storage_af'].values if K > 0 else np.zeros(dowy.size)
    Q_avg = medians[k + '_inflow_cfs'].values
    R_avg = medians[k + '_outflow_cfs'].values
    S_avg = medians[k + '_storage_af'].values if K > 0 else np.zeros(dowy.size)
    R_max = v['safe_release_cfs']
    if not init_storage:
        S0 = S_avg[-1]
    else:
        S0 = df_opt[k + '_storage_af'].values[-1]
        
    return (dowy, Q, K, R_avg, S_avg, S0, R_max, df_demand.combined_demand.values)  # deleted R_obs after Q_avg

# pull numpy arrays from dataframes
def get_simulation_data(res_keys, pump_keys, df_hydrology, medians, df_demand=None, init_storage=False):
    dowy = df_hydrology.dowy.values
    Q = df_hydrology[[k + '_inflow_cfs' for k in res_keys]].values
    Q_avg = medians[[k + '_inflow_cfs' for k in res_keys]].values
    R_avg = medians[[k + '_outflow_cfs' for k in res_keys]].values
    S_avg = medians[[k + '_storage_af' for k in res_keys]].values
    Gains_avg = medians['delta_gains_cfs'].values
    Pump_pct_avg = medians[[k + '_pumping_pct' for k in pump_keys]].values
    Pump_cfs_avg = medians[[k + '_pumping_cfs' for k in pump_keys]].values

    if df_demand is None:
        demand_multiplier = np.ones(dowy.size)
    else:
        demand_multiplier = df_demand.combined_demand.values

    if not init_storage:
        S0 = S_avg[-1, :]
    else:
        S0 = df_hydrology[[k + '_storage_af' for k in res_keys]].values[-1, :]

    return (dowy, Q, Q_avg, R_avg, S_avg, Gains_avg, Pump_pct_avg, Pump_cfs_avg, demand_multiplier, S0)

# pull numpy arrays from dataframes
def get_simulation_data_updated(res_keys, pump_keys, df_hydrology, medians, df_opt, df_demand=None, init_storage=False):
    dowy = df_hydrology.dowy.values
    Q = df_hydrology[[k + '_inflow_cfs' for k in res_keys]].values
    Q_avg = medians[[k + '_inflow_cfs' for k in res_keys]].values
    R_avg = medians[[k + '_outflow_cfs' for k in res_keys]].values
    S_avg = medians[[k + '_storage_af' for k in res_keys]].values
    Gains_avg = medians['delta_gains_cfs'].values
    Pump_pct_avg = medians[[k + '_pumping_pct' for k in pump_keys]].values
    Pump_cfs_avg = medians[[k + '_pumping_cfs' for k in pump_keys]].values

    if df_demand is None:
        demand_multiplier = np.ones(dowy.size)
    else:
        demand_multiplier = df_demand.combined_demand.values

    if not init_storage:
        S0 = S_avg[-1, :]
    else:
        S0 = df_opt[[k + '_storage_af' for k in res_keys]].values[-1, :]

    return (dowy, Q, Q_avg, R_avg, S_avg, Gains_avg, Pump_pct_avg, Pump_cfs_avg, demand_multiplier, S0)


@njit
def get_tocs(x, d):
    tp = [0, x[1], x[2], x[3], 366]
    sp = [1, x[4], x[4], 1, 1]
    return np.interp(d, tp, sp)


@njit
def reservoir_step(x, dowy, Q, S, K, R_avg, S_avg, tocs):
    '''
  Advances reservoir storage from one timestep to the next

    Parameters:
      x (np.array): Reservoir rule parameters (5)
      dowy (int): Day of water year
      Q (float): Inflow, cfs
      S (float): Storage, acre-feet
      K (float): Storage capacity, acre-feet
      R_avg (float): Median release for this day of the year, cfs
      S_avg (float): Median storage for this day of the year, acre-feet
      tocs (float): Top of conservation storage, fraction of capacity

    Returns:
      tuple(float, float): the updated release (cfs) and storage (af)

  '''
    R_avg *= cfs_to_afd
    Q *= cfs_to_afd

    R_target = R_avg  # default

    if S < S_avg:
        R_target = R_avg * (S / S_avg) ** x[0]  # exponential hedging

    S_target = max(0, S + Q - R_target)  # assumes 1-day forecast

    if S_target > K * tocs:  # flood pool
        R_target += (S_target - K * tocs) * x[5]
        S_target = max(0, S + Q - R_target)

    if S_target > K:  # spill
        R_target += S_target - K
    elif S_target < K * x[6]:  # below dead pool
        R_target = max(0,R_target - (K * x[6] - S_target))

    return (R_target * afd_to_cfs, np.max(np.array([S + Q - R_target, 0])))


cfs_to_taf = 2.29568411 * 10 ** -5 * 86400 / 1000


@njit
def reservoir_fit(x, dowy, Q, K, R_avg, S_avg, S0, R_max, df_demand):  # removed R_obs after R_avg
    """
    Evaluate reservoir model against historical observations for a set of parameters

      Parameters:
        x (np.array): Reservoir rule parameters (5)
        dowy (np.array(int)): Day of water year over the simulation
        Q (np.array(float)): Inflow, cfs
        S (np.array(float)): Storage, acre-feet
        K (float): Storage capacity, acre-feet
        R_avg (np.array(float)): Median release for each day of the year, cfs
        R_obs (np.array(float)): Observed historical release, cfs
        S_avg (np.array(float)): Median storage for each day of the year, acre-feet
        S_obs (np.array(float)): Observed historical storage, acre-feet.
          Not used currently, but could fit parameters to this instead.
        S0 (float): initial storage, acre-feet

      Returns:
        (float): the negative r**2 value of reservoir releases, to be minimized
  """
    tt = dowy.size
    R, S, res_demand = np.zeros(tt), np.zeros(tt), np.zeros(tt)
    tocs = get_tocs(x, dowy)
    shortage_cost, flood_cost,storage_cost = [np.zeros(tt) for _ in range(3)]
    # D = R_obs #demand is assumed as observed historical release, cfs
    DM = df_demand

    for t in range(0, tt):
        d = dowy[t]
        res_demand[t] = R_avg[d] * DM[t]  # median historical release * demand multiplier
        if t==0:
          inputs_r = (d, Q[t], S0, K, res_demand[t], S_avg[dowy[t - 1]], tocs[t])
        else:
          inputs_r = (d, Q[t], S[t - 1], K, res_demand[t], S_avg[dowy[t - 1]], tocs[t])
        
        R[t], S[t] = reservoir_step(x, *inputs_r)

        # squared deficit.
        # should be able to vectorize this.
        shortage_cost[t] = max(res_demand[t] - R[t], 0) ** 2 

        if R[t] > R_max:
            # flood penalty, high enough to be a constraint
            flood_cost[t] += 10 ** 5 * (R[t] - R_max)

    return shortage_cost.sum() + flood_cost.sum() 

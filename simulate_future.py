import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
from scipy.optimize import differential_evolution as DE
from util import *
import time
from multiprocessing import Pool
from TestCode_Future import get_simulation_data

# functions to pull numpy arrays from dataframes
def water_day_up(d,year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        year_leap=1
    else:
        year_leap=0
    if d>=(274+year_leap):
        yearday = d - (274+year_leap)
    else:
        yearday = d + 91
    return yearday

def func(gridi,gridj):
  nodes = json.load(open('data/nodes.json'))
  rk = [k for k in nodes.keys() if nodes[k]['type'] == 'reservoir' and nodes[k]['fit_policy']]
  Kr = np.array([nodes[k]['capacity_taf'] * 1000 for k in rk])
  pk = [k for k in nodes.keys() if (nodes[k]['type'] == 'pump') and (nodes[k]['fit_policy'])]
  Kp = np.array([nodes[k]['capacity_cfs'] for k in pk])
  safecap = np.array([nodes[k]['safe_release_cfs'] for k in rk])
  
  medians = pd.read_csv('data/historical_medians.csv', index_col=0)
  params = json.load(open('data/params.json'))
  params = tuple(np.array(v) for k, v in params.items())
  
  cmip5_scenarios = pd.read_csv('data/cmip5/scenario_names.csv').name.to_list()
  lulc_scenarios = pd.read_csv('data/lulc/scenario_names.csv').name.to_list()
  sc = cmip5_scenarios[gridi]
  sl = lulc_scenarios[gridj]
  df_Q = pd.read_csv('data/cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)
  df_demand = pd.read_csv('data/lulc/%s.csv.zip' % sl, index_col=0, parse_dates=True)
  df_Q['dowy'] = np.array([water_day_up(d,y) for d,y in zip(df_Q.index.dayofyear,df_Q.index.year)])
  input_data = get_simulation_data(rk, pk, df_Q, medians, df_demand)
  R, S, Delta, shortage_cost, flood_cost = model.simulate(params, Kr, Kp,safecap, *input_data)
  
  df_sim = pd.DataFrame(index=df_Q.index)
  df_sim['dowy'] = df_Q.dowy
  
  for i, r in enumerate(rk):
      df_sim[r + '_outflow_cfs'] = R[:, i]
      df_sim[r + '_storage_af'] = S[:, i]
      df_sim[r + '_flood_cost'] = flood_cost[:,i]
      df_sim[r + '_shortage_cost'] = shortage_cost[:, i]
      
  for i, k in enumerate(['delta_gains_cfs', 'delta_inflow_cfs', 'total_delta_pumping_cfs', 'delta_outflow_cfs']):
      df_sim[k] = Delta[:, i]

  objs = results_to_annual_objectives(df_sim, medians, nodes, rk, df_demand)
  objs.to_csv('output/sim/obj_%s_%s.csv.zip' % (sc, sl), compression='zip')

  # too much to save all the simulation output timeseries
  df_sim.to_csv('output/sim/sim_%s_%s.csv.zip' % (sc,sl), compression='zip')

st = time.time()
exp_a = range(97)#97) #(range(134, 154), 20) #134
exp_b = range(36)#36)           #range(504, 524)) * 20 #20
# auxiliary funciton to make it work
def product_helper(args):
    return func(*args)

def parallel_product(list_a, list_b):
    #spark given number of processes
    p = Pool(10)
    # set each matching item into a tuple
    job_args = [(x, y) for x in list_a for y in list_b]
#[(item_a, list_b[i]) for i, item_a in enumerate(list_a)]
    # map to pool
    p.map(product_helper, job_args)

if __name__ == '__main__':
    parallel_product(exp_a, exp_b)

    et  = time.time()
    print(et-st)
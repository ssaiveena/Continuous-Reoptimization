import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
from scipy.optimize import differential_evolution as DE
from util import *

# functions to pull numpy arrays from dataframes
def reservoir_training_data(k, v, df, medians, init_storage=False):
  dowy = df.dowy.values
  Q = df[k+'_inflow_cfs'].values
  K = v['capacity_taf'] * 1000
  R_obs = df[k+'_outflow_cfs'].values
  S_obs = df[k+'_storage_af'].values if K > 0 else np.zeros(dowy.size)
  Q_avg = medians[k+'_inflow_cfs'].values
  R_avg = medians[k+'_outflow_cfs'].values
  S_avg = medians[k+'_storage_af'].values if K > 0 else np.zeros(dowy.size)

  if not init_storage:
    S0 = S_avg[0]
  else:
    S0 = df[k+'_storage_af'].values[0]

  return (dowy, Q, K, Q_avg, R_avg, R_obs, S_avg, S_obs, S0)

def gains_training_data(df, medians):
  dowy = df.dowy.values
  Q_total = df['total_inflow_cfs'].values
  Q_total_avg = medians['total_inflow_cfs'].values
  S_total_pct = (df['total_storage'].values / 
                 np.array([medians['total_storage'].values[i] for i in df.dowy]))
  Gains_avg = medians['delta_gains_cfs'].values 
  Gains_obs = df['delta_gains_cfs'].values
  return (dowy, Q_total, Q_total_avg, S_total_pct, Gains_avg, Gains_obs)

def pump_training_data(k, v, df, medians):
  dowy = df.dowy.values
  Q_in = df.delta_inflow_cfs.values
  cap = v['capacity_cfs']
  Pump_pct_avg = medians[k+'_pumping_pct'].values
  Pump_cfs_avg = medians[k+'_pumping_cfs'].values

  if k == 'HRO':
    S_total_pct = (df['ORO_storage_af'].values / 
                   np.array([medians['ORO_storage_af'].values[i] for i in df.dowy]))
  else:
    S_total_pct = (df['total_storage'].values / 
                 np.array([medians['total_storage'].values[i] for i in df.dowy]))

  Pump_obs = df[k+'_pumping_cfs'].values
  return (dowy, Q_in, cap, Pump_pct_avg, Pump_cfs_avg, S_total_pct, Pump_obs)


np.random.seed(1337)
variables = json.load(open('data/nodes.json'))
df = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)['10-01-1997':]
medians = pd.read_csv('data/historical_medians.csv', index_col=0)
params = {}

df = df['10-01-2009':] # env rule change

# fit reservoir policy parameters
for k,v in variables.items():
  if v['type'] != 'reservoir' or not v['fit_policy']: continue
  training_data = reservoir_training_data(k, v, df, medians)
  opt = DE(model.reservoir_fit, 
           bounds = [(1,3), (0,100), (100,250), (250,366), (0,1), (0,1), (0,0.2)], 
           args = training_data) 
  params[k] = opt.x.tolist()
  print('%s: R2=%0.2f, NFE=%d' % (k, -opt.fun, opt.nfev))

# fit gains parameters
training_data = gains_training_data(df, medians)
opt = DE(model.gains_fit, bounds = [(0,2), (0.5,3), (0,1)], args = training_data)
params['Gains'] = opt.x.tolist()
print('Gains: R2=%0.2f, NFE=%d' % (-opt.fun, opt.nfev))

# fit pump parameters
for k,v in variables.items():
  if v['type'] != 'pump' or not v['fit_policy']: continue
  training_data = pump_training_data(k, v, df, medians)
  opt = DE(model.pump_fit, bounds = [(0,5), (5000,30000), (0,5), (0,8000), (0,8000), (0,1), (0,1)], args = training_data)
  params[k] = opt.x.tolist()
  print('%s: R2=%0.2f, NFE=%d' % (k, -opt.fun, opt.nfev))

with open('data/params.json', 'w') as f:
    json.dump(params, f, indent=2)

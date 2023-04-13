import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
from scipy.optimize import differential_evolution as DE
from util import *
import time
from numba import njit
import math
from setup_Future import reservoir_training_data
from setup_Future import reservoir_training_data_updated
from setup_Future import reservoir_fit
from setup_Future import get_simulation_data
from setup_Future import get_simulation_data_updated
from multiprocessing import Pool
import os

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

def func(gridi,gridj,f,w,add_num):
    cmip5_scenarios = pd.read_csv('data/cmip5/scenario_names.csv').name.to_list()
    lulc_scenarios = pd.read_csv('data/lulc/scenario_names.csv').name.to_list()
    filedir = 'output/scenarios/Reoptimize/f_' + str(f) + '_w_' + str(w) + '/'

# np.random.seed(1337)
    variables = json.load(open('data/nodes.json'))
    df = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)['10-01-1997':]
    medians = pd.read_csv('data/historical_medians.csv', index_col=0)
    params1 = {}

    rk = [k for k in variables.keys() if (variables[k]['type'] == 'reservoir') and (variables[k]['fit_policy'])]
    Kr = np.array([variables[k]['capacity_taf'] * 1000 for k in rk])
    pk = [k for k in variables.keys() if (variables[k]['type'] == 'pump') and (variables[k]['fit_policy'])]
    Kp = np.array([variables[k]['capacity_cfs'] for k in pk])
    safecap = np.array([variables[k]['safe_release_cfs'] for k in rk])

    sc = cmip5_scenarios[gridi]
    sl = lulc_scenarios[gridj]
    df_Q = pd.read_csv('data/cmip5/%s.csv.zip' % sc, index_col=0, parse_dates=True)
    df_demand = pd.read_csv('data/lulc/%s.csv.zip' % sl, index_col=0, parse_dates=True)
    df_Q['dowy'] = np.array([water_day_up(d,y) for d,y in zip(df_Q.index.dayofyear,df_Q.index.year)])

    splitdata = np.array_split(df_Q, np.intersect1d(np.where(df_Q.index.day == 1), np.where(df_Q.index.month == 10)))
    splitdata_demand = np.array_split(df_demand, np.intersect1d(np.where(df_demand.index.day == 1), np.where(df_demand.index.month == 10)))
    # defining data for frequency "f"
    n_f = int(len(df_Q) / (f * 365))
    num = n_f - w + add_num
    frames_hist_w = [pd.DataFrame() for x in range(num)]       # defining data for historical window "w"
    frames_hist_w_demand = [pd.DataFrame() for x in range(num)]
    frames = [pd.DataFrame() for x in range(num)]
    frames_demand = [pd.DataFrame() for x in range(num)]

    for i in range(1,num):
       for freq in range(w):
           frames_hist_w[i] = pd.concat([frames_hist_w[i],splitdata[freq+(i-1)*f+1]])
           frames_hist_w_demand[i] = pd.concat([frames_hist_w_demand[i],splitdata_demand[freq+(i-1)*f+1]])

       for j in range(f):
           frames[i] = pd.concat([frames[i], splitdata[j + (i-1) * f + 1+w]])
           frames_demand[i] = pd.concat([frames_demand[i], splitdata_demand[j + (i-1) * f + 1+w]])
    df_opt1 = pd.DataFrame()
    objs = pd.DataFrame()
    for nsplit in range(1,num):
        df = frames_hist_w[nsplit]
        for k,v in variables.items():
            if v['type'] != 'reservoir' or not v['fit_policy']: continue
 
            if nsplit>1:
              training_data = reservoir_training_data_updated(k, v, df, medians, frames_hist_w_demand[nsplit],df_opt,init_storage=True)
            else:
              training_data = reservoir_training_data(k, v, df, medians, frames_hist_w_demand[nsplit],init_storage=False)
            
            fun1 = []
            opt1 = {}

            for nopt in range(10):
              opt = DE(reservoir_fit, tol=1, maxiter=100000,
                   bounds=[(1, 3), (0, 100), (100, 250), (250, 366),  (0, 1), (0, 1), (0, 0.2)],
                   args=training_data)
              fun1.append(opt.fun)
              opt1[nopt] = opt.x.tolist()

            params1[k] = opt1[fun1.index(min(fun1))]


        with open(filedir + 'data_%s_%s_%0.0f.json' % (sc, sl, nsplit), 'w') as ff:
            json.dump(params1, ff,indent=2)

        params = json.load(open('data/params.json'))
        for k in params1.keys():
            params[k] = params1[k]
        params = tuple(np.array(v) for k,v in params.items())

        df = frames[nsplit]
        if nsplit>1:
            input_data = get_simulation_data_updated(rk, pk, df, medians, df_opt, df_demand=frames_demand[nsplit], init_storage=True)
        else:        
            input_data = get_simulation_data(rk, pk, df, medians,df_demand=frames_demand[nsplit], init_storage=False)
        RO,SO,DeltaO, shortage_cost, flood_cost = model.simulate(params, Kr, Kp, safecap, *input_data)

        
        df_opt = pd.DataFrame(index=df.index)
        df_opt['dowy'] = df.dowy

        for i,r in enumerate(rk):
            df_opt[r+'_outflow_cfs'] = RO[:,i]
            df_opt[r+'_storage_af'] = SO[:,i]
            df_opt[r + '_flood_cost'] = flood_cost[:,i]
            df_opt[r + '_shortage_cost'] = shortage_cost[:, i]

        delta_keys = ['delta_gains_cfs', 'delta_inflow_cfs', 'HRO_pumping_cfs', 'TRP_pumping_cfs', 'delta_outflow_cfs']
        for i,k in enumerate(delta_keys):
            df_opt[k] = DeltaO[:,i]
        df_opt['total_delta_pumping_cfs'] = df_opt.HRO_pumping_cfs + df_opt.TRP_pumping_cfs

        objs = pd.concat([objs,results_to_annual_objectives(df_opt, medians, variables, rk, df_demand=frames_demand[nsplit])])
        df_opt1 = pd.concat([df_opt1,df_opt])
    filename = 'reopt_opt_historical_%s_%s_%.f.csv.zip' % (sc, sl, nsplit)
    df_opt1.to_csv(filedir + filename)
    objs.to_csv(filedir + 'reopt_obj_historical_opt_%s_%s_%.f.csv.zip' % (sc, sl, nsplit))

def get_add_num(f,w):
    #this function determines the number of optimizations to perform depending on the frequency of operation and historical window
    i = f
    j = w
    if i == 1:
        add_num = 1
    elif i == 2:
        add_num = 3 if j == 5 else (6 if j == 10 else (8 if j == 15 else 11 if j == 20 else(16 if j == 30 else 26)))
    elif i == 5:
        add_num = 5 if j == 5 else (9 if j == 10 else (13 if j == 15 else 17 if j == 20 else(25 if j == 30 else 41)))
    elif i == 10:
        add_num = 6 if j == 5 else (10 if j == 10 else (15 if j == 15 else 19 if j == 20 else(28 if j == 30 else 46)))
    elif i == 15:
        add_num = 6 if j == 5 else (11 if j == 10 else (15 if j == 15 else 20 if j == 20 else(29 if j == 30 else 48)))
    else:
        add_num = 6 if j == 5 else (10 if j == 10 else (15 if j == 15 else 20 if j == 20 else(29 if j == 30 else 48)))
    return add_num

f_comb = [1, 2, 5, 10, 15, 20] #years of frequency of operation
w_comb = [5, 10, 15, 20, 30, 50]#years of historical window of operation

st = time.time()
exp_a = range(97) #number of climate scenarios
exp_b = range(36)#number of land use scenarios

# auxiliary funciton to make it work
def product_helper(args):
    return func(*args)

def parallel_product(list_a, list_b,f,w,add_num):
    #spark given number of processes
    p = Pool(12)
    # set each matching item into a tuple
    job_args = [(x, y,f, w, add_num) for x,y in zip(list_a,list_b)] #iterating through all possible combinations of f and w
    # map to pool
    p.map(product_helper, job_args)

if __name__ == '__main__':
    for f in f_comb:
        for w in w_comb:
            print(f,w)
            parallel_product(exp_a, exp_b,f,w,get_add_num(f,w))

    et  = time.time()
    print(et-st)

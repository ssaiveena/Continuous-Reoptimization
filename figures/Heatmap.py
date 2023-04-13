#Heatmap as Figure 3 in the manuscript
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
from jsonmerge import merge
import os
from util import water_day
import seaborn as sns
from joblib import Parallel, delayed
import math

def get_add_num(f,w):
    i =f
    j=w
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
#########################################################################################################################
#################################Plot objective function as heat map for individual reservoirs###########################
#########################################################################################################################
def calculate_reliability(data, medians, k, df_demand=None):
    name = k + '_outflow_cfs'
    res_target = pd.Series([medians[name][i] for i in data.dowy], index=data.index)
    if df_demand is not None:
        res_target *= df_demand.combined_demand
    rel = data[name].resample('AS-OCT').sum() / res_target.resample('AS-OCT').sum()
    rel[rel > 1] = 1
    return rel

def calculate_reliability_cost(data, k):
    # name = k + '_outflow_cfs'
    name = k + '_flood_cost'
    name1 = k + '_shortage_cost'
    rel = data[name].resample('AS-OCT').sum() + data[name1].resample('AS-OCT').sum() #choose one of these two when calculating shortage cost or flood cost
    return rel


def sqrt_func(i, j, res, cmip5_scenarios, lulc_scenarios,f, w):
    df_demand = pd.read_csv('data/lulc/%s.csv.zip' % lulc_scenarios[j], index_col=0, parse_dates=True)
    dateslist = []
    data = []

    n_f = int(54057 / (f * 365))
    num = n_f - w + get_add_num(f, w)  # math.ceil(n_w/n_f) - 1
    res_key = [k for k in variables.keys()]
    
    data = pd.read_csv('output/scenarios/Reoptimize/f_%d_w_%d/reopt_opt_historical_%s_%s_%d.csv.zip' %(f, w, cmip5_scenarios[i], lulc_scenarios[j], num-1),usecols=[res_key[res] + '_flood_cost',res_key[res] + '_shortage_cost','Unnamed: 0','dowy']) 
    data.index = pd.to_datetime(data['Unnamed: 0'].values)
      
    rel_cal = calculate_reliability_cost(data, res_key[res])
    #rel_cal = calculate_reliability(data, medians, res_key[res], df_demand = df_demand)
    mask = (rel_cal.index.year >= 2030) & (rel_cal.index.year <= 2080) #choosing years from 2030 to 2080 in the future
    data_reopt = rel_cal.loc[mask].mean()
    
    return data_reopt
    
def sim(i, j, res, cmip5_scenarios, lulc_scenarios):
    df_demand = pd.read_csv('data/lulc/%s.csv.zip' % lulc_scenarios[j], index_col=0, parse_dates=True)
    dateslist = []
    data = []

    res_key = [k for k in variables.keys()]
    
    data = pd.read_csv('output/scenarios/sim_%s_%s.csv.zip' %(cmip5_scenarios[i], lulc_scenarios[j]),usecols=[res_key[res] + '_shortage_cost',res_key[res] + '_flood_cost','Unnamed: 0']) 
    data.index = pd.to_datetime(data['Unnamed: 0'].values)
    data['dowy'] = np.array([water_day(d) for d in data.index.dayofyear])
    #rel_cal = calculate_reliability(data, medians, res_key[res], df_demand = df_demand)
    rel_cal = calculate_reliability_cost(data, res_key[res])
    mask = (rel_cal.index.year >= 2030) & (rel_cal.index.year <= 2080)
    data_reopt = rel_cal.loc[mask].mean()

    return data_reopt

def PF(i, j, res, cmip5_scenarios, lulc_scenarios):
    df_demand = pd.read_csv('data/lulc/%s.csv.zip' % lulc_scenarios[j], index_col=0, parse_dates=True)
    dateslist = []
    data = []

    res_key = [k for k in variables.keys()]
    
    data = pd.read_csv('output/scenarios/opt_historical_PerfectForesight_%s_%s_%.f.csv.zip' %(cmip5_scenarios[i], lulc_scenarios[j], 147),usecols=[res_key[res] + '_shortage_cost',res_key[res] + '_flood_cost','Unnamed: 0'])
    data.index = pd.to_datetime(data['Unnamed: 0'].values)
    data['dowy'] = np.array([water_day(d) for d in data.index.dayofyear])
    #rel_cal = calculate_reliability(data, medians, res_key[res], df_demand = df_demand)
    rel_cal = calculate_reliability_cost(data, res_key[res])
    
    mask = (rel_cal.index.year >= 2030) & (rel_cal.index.year <= 2080)
    data_reopt = rel_cal.loc[mask].mean()
     
    return data_reopt

variables = json.load(open('data1.json'))
medians = pd.read_csv('data/historical_medians.csv', index_col=0)

cmip5_scenarios = pd.read_csv('data/cmip5/scenario_names.csv').name.to_list()
lulc_scenarios = pd.read_csv('data/lulc/scenario_names.csv').name.to_list()

#define different colors for combinations of f and w
colors_f = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
res_keys = [k for k in variables.keys()]

data_agg_reopt_heatmap =[ [ 0 for y in range(6) ] for x in range( 6) ]
data_agg_sim_heatmap =[0]
data_agg_PF_heatmap = [0]
data_agg_reopt_heatmap_med =[ [ 0 for y in range(6) ] for x in range( 6) ]
data_agg_sim_heatmap_med =[0]
data_agg_PF_heatmap_med = [0]
fig, ax = plt.subplots()

f_comb = [1,2,5,10,15,20]
w_comb = [5,10,15,20,30,50]
for res in range(len(res_keys)):
    f_ind=0
    for f in f_comb:
        w_ind = 0
        for w in w_comb:
            #data_reopt = [0 for x in range(97*36)]
            
            data_agg_reopt = Parallel(n_jobs=120)(delayed(sqrt_func)(i, j,res=res, cmip5_scenarios=cmip5_scenarios, lulc_scenarios=lulc_scenarios,f =f, w=w) for i in range(97) for j in range(36))

            #mask = (data_agg_reopt.index.year >= 2030) & (data_agg_reopt.index.year < 2080)
            data_agg_reopt_heatmap[f_ind][w_ind] = np.percentile(data_agg_reopt,75) - np.percentile(data_agg_reopt,25)#data_agg_reopt.loc[mask].mean()
            data_agg_reopt_heatmap_med[f_ind][w_ind] = np.percentile(data_agg_reopt,50) 
            w_ind = w_ind+1
        f_ind = f_ind +1
    
   
    data_agg_sim = Parallel(n_jobs=120)(delayed(sim)(i, j,res=res, cmip5_scenarios=cmip5_scenarios, lulc_scenarios=lulc_scenarios) for i in range(97) for j in range(36))
    data_agg_sim_heatmap = np.percentile(data_agg_sim,75) - np.percentile(data_agg_sim,25)
    data_agg_sim_heatmap_med = np.percentile(data_agg_sim,50)

    data_agg_PF = Parallel(n_jobs=120)(delayed(PF)(i, j,res=res, cmip5_scenarios=cmip5_scenarios, lulc_scenarios=lulc_scenarios) for i in range(97) for j in range(36))
    data_agg_PF_heatmap = np.percentile(data_agg_PF,75) - np.percentile(data_agg_PF,25)
    data_agg_PF_heatmap_med = np.percentile(data_agg_PF,50)
    
    x_axis_labels = [5,10,15,20,30,50] # labels for x-axis
    y_axis_labels = [1,2,5,10,15,20] # labels for y-axis
    min_val =np.min(data_agg_reopt_heatmap/data_agg_PF_heatmap)#,np.min([data_agg_sim_heatmap,data_agg_PF_heatmap])])
    max_val = np.max(data_agg_reopt_heatmap/data_agg_PF_heatmap)#,np.max([data_agg_sim_heatmap,data_agg_PF_heatmap])])
    
    s = sns.heatmap(-data_agg_reopt_heatmap/(data_agg_sim_heatmap-data_agg_PF_heatmap) + data_agg_sim_heatmap/((data_agg_sim_heatmap-data_agg_PF_heatmap), annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels,cmap='RdBu', vmin=-1, vmax=1)
    s.set(xlabel='Historical Window', ylabel='Frequency')

    cb = s.collections[0].colorbar
    cb = fig.Figure.colorbar(s)

    plt.savefig('heatmap_total_norm%d.png' %res)
    plt.close()


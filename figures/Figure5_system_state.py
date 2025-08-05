##################################################################################################################
############################Plotting states of the system##############################################################
#########################################################################################################
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


sc = 'fio-esm_rcp45_r1i1p1'
sl = 'FORE-SCE-A1B'
#Plotting data for carryover storage
fig, (ax1,ax2,ax3) = plt.subplots(nrows = 3,ncols = 1, sharex=True)
ax3.locator_params(axis='y', nbins=4)
filedir = 'output/scenarios/'
path = filedir + "sim_%s_%s.csv.zip" % (sc, sl)

data = pd.read_csv(path)
data.index = pd.to_datetime(data['Unnamed: 0'].values)
mask = (data.index.year >= 2081) & (data.index.year <= 2091)
data = data.loc[mask]
plt.subplot(3, 1, 1)
plt.yscale("log")
plt.plot(data.index,data['SHA_storage_af'], color='#404040',alpha=0.5)
plt.subplot(3, 1, 2)
plt.yscale("log")
plt.plot(data.index,data['SHA_outflow_cfs'], color='#404040',alpha=0.5)
plt.subplot(3, 1, 3)
print(min(data['SHA_storage_af']))
plt.plot(data.index,data['SHA_flood_cost'] + data['SHA_shortage_cost'], color='#404040',alpha=0.5)


filedir = 'output/scenarios/'
path = filedir + "opt_historical_PerfectForesight_%s_%s_%.f.csv.zip" % (sc, sl, 147)
ti_m = os.path.getmtime(path)

data = pd.read_csv(path)
data.index = pd.to_datetime(data['Unnamed: 0'].values)
mask = (data.index.year >= 2081) & (data.index.year <= 2091)
data = data.loc[mask]
plt.subplot(3, 1, 1)
plt.yscale("log")
plt.plot(data.index,data['SHA_storage_af'], color='#2c7bb6')
print(min(data['SHA_storage_af']))
plt.subplot(3, 1, 2)
plt.yscale("log")
plt.plot(data.index,data['SHA_outflow_cfs'], color='#2c7bb6')
plt.subplot(3, 1, 3)
plt.yscale("symlog",linthreshy=1)
plt.plot(data.index,data['SHA_flood_cost'] + data['SHA_shortage_cost'], color='#2c7bb6')


f = 15 #choosing optimal combiantion of f and w 
w = 50
filedir = 'output/scenarios/Reoptimize/f_' + str(f) + '_w_' + str(w) + '/'
path = filedir + "reopt_opt_historical_%s_%s_%.f.csv.zip" % (sc, sl, 6)
ti_m = os.path.getmtime(path)


data = pd.read_csv(path)
data.index = pd.to_datetime(data['Unnamed: 0'].values)

mask = (data.index.year >= 2081) & (data.index.year <= 2091) #choosing a decade of data
data = data.loc[mask]

plt.subplot(3, 1, 1)
plt.yscale("log")
plt.plot(data.index,data['SHA_storage_af'], color='#ca0020')
print(min(data['SHA_storage_af']))
plt.ylabel('Storage')
plt.subplot(3, 1, 2)
plt.yscale("log")
plt.plot(data.index,data['SHA_outflow_cfs'], color='#ca0020')
plt.ylabel('Outflow')
plt.subplot(3, 1, 3)
plt.yscale("symlog",linthreshy=1e1)
plt.plot(data.index,data['SHA_flood_cost'] + data['SHA_shortage_cost'], color='#ca0020')
plt.ylabel('Total Cost')

plt.legend(['sim','PF','f=15, w=50'], ncol = 3, loc = 'lower center', bbox_to_anchor=(0.5, -0.5))
plt.tight_layout(w_pad=0.1)
plt.savefig('System_state.pdf') #Figure 5 of manuscript

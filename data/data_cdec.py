import numpy as np
import pandas as pd
import json 
from datetime import date
from util import *

# update historical data from CDEC through today
updating = False # False will overwrite the file
fname = 'data/historical.csv'
variables = json.load(open('data/nodes.json'))
sd = '1994-10-01' # approximate
today = date.today().strftime("%Y-%m-%d")

if updating:
  initial = pd.read_csv(fname, index_col=0, parse_dates=True)
  sd = (initial.index[-1]).strftime('%Y-%m-%d')

df = pd.DataFrame(index=pd.date_range(start=sd, freq='D', end=today))

for cdec_id,v in variables.items():
  for s,n in zip(v['sensor_ids'], v['sensor_names']):
    k = '%s_%s' % (cdec_id, n) if not v["is_substitution"] else n
    print(k)
    df[k] = np.nan

    try:
      df_k = cdec_sensor_data(cdec_id, sensor=s, duration='D', sd=sd, ed=today)
      df_k = df_k.replace('---', np.nan).astype(float) # some values missing
      df[k][df_k.index] = df_k.values
    except:
        print('CDEC Error: %s skipped' % k)

if updating:
  new = initial.append(df)
else:
  new = df

new = new.dropna(how='all') # remove rows that contain all nans
df = new[~new.index.duplicated(keep='last')]

# clean up missing data and add some columns
df = df.fillna(method='ffill')
df[df < 0] = 0.01
df['dowy'] = np.array([water_day(d) for d in df.index.dayofyear])

df.loc[sd:, 'PAR_storage_af'] += df.loc[sd:, 'CMN_storage_af']
df['PAR_outflow_cfs'] = df['CMN_outflow_cfs']

df['total_inflow_cfs'] = df.filter(like='inflow_cfs').sum(axis=1)
df['total_reservoir_outflows_cfs'] = df.filter(like='outflow_cfs').drop('delta_outflow_cfs', axis=1).sum(axis=1)
df['total_storage'] = df.filter(regex='_af$').sum(axis=1)

df.loc[['2012-07-01', '2013-12-31'], 'delta_outflow_cfs'] = np.nan # bad data points
df.delta_outflow_cfs.fillna(method='ffill', inplace=True)
df['delta_inflow_cfs'] = df['delta_outflow_cfs'] + df['HRO_pumping_cfs'] + df['TRP_pumping_cfs']
df['delta_gains_cfs'] = df.delta_inflow_cfs - df.total_reservoir_outflows_cfs
df['total_delta_pumping_cfs'] = df.TRP_pumping_cfs + df.HRO_pumping_cfs
df['HRO_pumping_pct'] = df.HRO_pumping_cfs / df.delta_inflow_cfs
df['TRP_pumping_pct'] = df.TRP_pumping_cfs / df.delta_inflow_cfs
df['total_delta_pumping_pct'] = df.total_delta_pumping_cfs / df.delta_inflow_cfs

# also append the medians for each variable
# delta variables - only use since 2009
medians = pd.DataFrame(index=range(0,366))
for k in df.columns:
  if k == 'dowy': continue
  medians[k] = df.groupby('dowy')[k].median()
  if any(x in k for x in ('delta', 'HRO', 'TRP')):
    medians[k] = df['10-01-2009':].groupby('dowy')[k].median()
medians.loc[365] = medians.loc[364] # leap years

print(df.isnull().sum() * 100 / len(df)) # check pct nans
df.to_csv('data/historical.csv')
medians.to_csv('data/historical_medians.csv')

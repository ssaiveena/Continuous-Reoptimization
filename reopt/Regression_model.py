import numpy as np
import pandas as pd
import json
from jsonmerge import merge
import os
import matplotlib.backends.backend_pdf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numba import njit 
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models
import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization
import graphviz # for plotting decision tree graphs

def fitting(X, y, criterion, splitter, mdepth):

    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Fit the model
    model = tree.DecisionTreeRegressor( random_state=1234, max_depth=4,
                                        splitter=splitter)
                                        
    clf = model.fit(X_train, y_train)
    importance = model.feature_importances_
    print(importance)
    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    text_representation = tree.export_text(model)
    print(text_representation)

    # Tree summary and model evaluation metrics
    print('*************** Tree Summary ***************')
    #print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print('No. of leaves: ', clf.tree_.n_leaves)
    print('No. of features: ', clf.n_features_in_)
    print('--------------------------------------------------------')
    print("")
    #################################################################
    #The accuracy scores below are plotted in the supplementary material
    print('*************** Evaluation on Test Data ***************')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    #print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    #print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')
    
    # Use graphviz to plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X.columns, 
                                filled=True, 
                                rounded=True, 
                                #rotate=True,
                               ) 
    graph = graphviz.Source(dot_data)
 
    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf, graph

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

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""

    colNames = ['Mean Flow', 'Peak Flow', 'Median Flow', 'Coeff Var', 'Skew','Demand']

    # define index as dates of ressmpled data.
    # data is resampled annually with end of year in September

    # create empty dataframe
    WYDataDF = pd.DataFrame(columns=colNames)

    # resample data
    WYData = DataDF#.resample('AS-OCT')

    # add metrics to dataframe
    WYDataDF.at[0,'Mean Flow'] = WYData['Discharge'].mean()
    WYDataDF['Peak Flow'] = WYData['Discharge'].max()
    #WYDataDF['Peak dowy'] = DataDF.loc[WYData['Discharge'].idxmax()]['dowy'].values
    WYDataDF['Median Flow'] = WYData['Discharge'].median()
    WYDataDF['Coeff Var'] = WYData['Discharge'].std() / WYData['Discharge'].mean() * 100
    WYDataDF['Skew'] = WYData['Discharge'].skew()
    WYDataDF['Demand'] = WYData['demand'].mean()
    return (WYDataDF)

def calculate_reliability(data, medians, k, df_demand=None):
    name = k + '_outflow_cfs'
    res_target = pd.Series([medians[name][i] for i in data.dowy], index=data.index)
    if df_demand is not None:
        res_target *= df_demand.combined_demand
    rel = data[name].resample('AS-OCT').sum() / res_target.resample('AS-OCT').sum()
    rel[rel > 1] = 1
    return rel

variables1 = json.load(open('data/params.json'))
medians = pd.read_csv('data/historical_medians.csv', index_col=0)

cmip5_scenarios = pd.read_csv('data/cmip5/scenario_names.csv').name.to_list()
lulc_scenarios = pd.read_csv('data/lulc/scenario_names.csv').name.to_list()
res_keys = [k for k in variables1.keys()]
f_comb = [1,2,5,10,15,20]
w_comb = [5,10,15,20,30,50]


rel_all = pd.DataFrame()

for f in f_comb:
    for w in w_comb:
        print(f,w)
        data_param = pd.DataFrame()
        for res in range(1):#len(res_keys)):
            cmiplulcind = 0
                     
            for cmip5 in cmip5_scenarios:
                for lulc in lulc_scenarios:
                    df_Q = pd.read_csv('data/cmip5/%s.csv.zip' % cmip5, index_col=0, parse_dates=True)
                    df_demand = pd.read_csv('data/lulc/%s.csv.zip' % lulc, index_col=0, parse_dates=True)
                    splitdata = np.array_split(df_Q,
                                               np.intersect1d(np.where(df_Q.index.day == 1), np.where(df_Q.index.month == 10)))
                    splitdata_demand = np.array_split(df_demand, np.intersect1d(np.where(df_demand.index.day == 1), np.where(df_demand.index.month == 10)))
        ##################################################################################################################3333
                    data = []
                    n_f = int(54057 / (f * 365))
                    num = n_f - w + get_add_num(f, w)  # math.ceil(n_w/n_f) - 1
                    frames = pd.DataFrame() 
                    frames_demand = [pd.DataFrame() for x in range(num)]
                    frames_w = [pd.DataFrame() for x in range(num)]
                    frames_w_demand = [pd.DataFrame() for x in range(num)]
                    
                    res_key = [k for k in variables1.keys()]
                    
                    for i in range(1,num):
                      for freq in range(w):
                        frames_w[i] = pd.concat([frames_w[i],splitdata[freq+(i-1)*f+1]])
                        frames_w_demand[i] = pd.concat([frames_w_demand[i],splitdata_demand[freq+(i-1)*f+1]])
   
                    for i in range(1, num):
                        frames = pd.DataFrame()
                        df1 = frames_w[i]
                        df_demand1 = frames_w_demand[i]
                        variables = json.load(open('output/scenarios/Reoptimize/f_%d_w_%d/data_%s_%s_%d.json' % (f, w, cmip5, lulc, i)))
                        frames['Discharge'] = df1[res_key[res] + '_inflow_cfs'].values
                        frames['demand'] = df_demand1['combined_demand'].values
                        data = GetAnnualStatistics(frames)
                        data['var1'] = variables[res_key[res]][0]
                        data_param = pd.concat([data_param, data])

                    data_rel = pd.read_csv('output/scenarios/Reoptimize/f_%d_w_%d/reopt_opt_historical_%s_%s_%d.csv.zip' %(f, w, cmip5, lulc, num-1),usecols=[res_key[res] + '_outflow_cfs','Unnamed: 0','dowy'])
                    data_rel.index = pd.to_datetime(data_rel['Unnamed: 0'].values)
                    rel_cal = calculate_reliability(data_rel, medians, res_key[res], df_demand = df_demand)
                    rel_all = pd.concat([rel_all,rel_cal])
                        
        # Select data for modeling
        X=data_param[['Mean Flow', 'Peak Flow', 'Median Flow', 'Coeff Var', 'Skew','Demand']]
        y=data_param['var1'].values
        
        # Fit the model and display results
        X_train, X_test, y_train, y_test, clf, graph = fitting(X, y ,'gini', 'best',
                                                               mdepth=4)
        
        name, ext = os.path.splitext('fig_up_20_%d_0.pdf' %w)
        #graph = graphviz.Source(dot)
        graph.format = ext[1:]
        graph.view(name, cleanup = True) 
        
        from dtreeviz.trees import dtreeviz # remember to load the package
        
        viz = dtreeviz(clf, X, y,
                        target_name="param",
                        feature_names=X.columns)
        
        viz.save('reg_%d_%d_0.svg' % (f,w))

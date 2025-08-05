##################################################################################################################
############################Plotting Parameter varations ##############################################################
#########################################################################################################
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


def get_add_num(f, w):
	i = f
	j = w
	if i == 1:
		add_num = 1
	elif i == 2:
		add_num = 3 if j == 5 else (6 if j == 10 else (8 if j == 15 else 11 if j == 20 else (16 if j == 30 else 26)))
	elif i == 5:
		add_num = 5 if j == 5 else (9 if j == 10 else (13 if j == 15 else 17 if j == 20 else (25 if j == 30 else 41)))
	elif i == 10:
		add_num = 6 if j == 5 else (10 if j == 10 else (15 if j == 15 else 19 if j == 20 else (28 if j == 30 else 46)))
	elif i == 15:
		add_num = 6 if j == 5 else (11 if j == 10 else (15 if j == 15 else 20 if j == 20 else (29 if j == 30 else 48)))
	else:
		add_num = 6 if j == 5 else (10 if j == 10 else (15 if j == 15 else 20 if j == 20 else (29 if j == 30 else 48)))
	return add_num


pdf = matplotlib.backends.backend_pdf.PdfPages("Param_variation.pdf") #Figure 4 in paper

train_params = json.load(open('data/params.json'))
variables = json.load(open('data1.json'))

# dateall = range(1956, 2099)

cmip5_w = 'cnrm-cm5_rcp45_r1i1p1'  # choosing wet and dry scenario
cmip5_d = 'miroc-esm-chem_rcp60_r1i1p1'
lulc = 'LUCAS-BAU_Med-0010' #choose a random lulc scenario
# define different colors for combinations of f and w
colors_f = ['#4575b4', '#d73027']  # '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb',
lw = [0.8, 1.2]
res_keys = [k for k in variables.keys()]
for res in range(1):  # len(res_keys)):
	df_Q_w = pd.read_csv('data/cmip5/%s.csv.zip' % cmip5_w, index_col=0, parse_dates=True)
	df_Q_d = pd.read_csv('data/cmip5/%s.csv.zip' % cmip5_d, index_col=0, parse_dates=True)
	splitdata_w = np.array_split(df_Q_w, np.intersect1d(np.where(df_Q_w.index.day == 1),
														np.where(df_Q_w.index.month == 10)))
	splitdata_d = np.array_split(df_Q_d, np.intersect1d(np.where(df_Q_d.index.day == 1),
														np.where(df_Q_d.index.month == 10)))
	##################################################################################################################3333
	f_comb = [15] #selected combination of f and w
	w_comb = [50]
	fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True)
	numb = 0
	for f, w in zip(f_comb, w_comb):

		# fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6), (ax7,ax8)) = plt.subplots(4, 2)

		dateslist = []
		data = []
		data_reopt_w = []
		data_reopt_d = []
		data_PF = []

		n_f = int(54057 / (f * 365))
		num = n_f - w + get_add_num(f, w)  
		frames_w = [pd.DataFrame() for x in range(num)]
		frames_d = [pd.DataFrame() for x in range(num)]

		for i in range(1, num):
			for j in range(f):
				frames_w[i] = pd.concat([frames_w[i], splitdata_w[j + (i - 1) * f + 1 + w]])
				frames_d[i] = pd.concat([frames_d[i], splitdata_d[j + (i - 1) * f + 1 + w]])
				dateslist = np.append(dateslist, frames_w[i].index.year.unique().to_list())

		for parfile in range(1, num):
			variables_w = json.load(open('output/scenarios/Reoptimize/f_%d_w_%d/data_%s_%s_%d.json' % (f, w, cmip5_w, lulc, parfile)))
			variables_d = json.load(open('output/scenarios/Reoptimize/f_%d_w_%d/data_%s_%s_%d.json' % (f, w, cmip5_d, lulc, parfile)))
			res_key = [k for k in variables.keys()]
			data_reopt_w = np.append(data_reopt_w, variables_w[res_key[res]])
			data_reopt_d = np.append(data_reopt_d, variables_d[res_key[res]])
		data_reopt_w = data_reopt_w.reshape(num - 1, 7).transpose()
		data_reopt_d = data_reopt_d.reshape(num - 1, 7).transpose()

		train_params_data = train_params[res_key[res]]  # reading the simulated data for plotting

		for i in range(7):  # 7 for 7 parameter of the reservoir policy
			plt.subplot(4, 2, i + 1)
			plt.plot(np.unique(dateslist)[:len(np.unique(dateslist)) - 1], np.repeat(data_reopt_w[i], f),color=colors_f[0], linewidth=1)
			plt.plot(np.unique(dateslist)[:len(np.unique(dateslist)) - 1], np.repeat(data_reopt_d[i], f),color=colors_f[1], linewidth=1)
			plt.plot(np.unique(dateslist), np.repeat(train_params_data[i], len(np.unique(dateslist))),color='#000000')
			plt.ylabel('x$_{%.0f}$' % i, fontsize=12)

			if i < 5:
				plt.xticks([])
			numb = numb + 1
		ax = plt.subplot(4, 2, 8)
		ax.axis('off')

		plt.plot([], [], linewidth=1.0, color='#d73027', marker='None')
		plt.plot([], [], linewidth=1.0, color='#4575b4', marker='None')
		plt.plot([], [], linewidth=1.0, color='#000000', marker='None')
		plt.legend(['Dry scenario', 'Wet scenario', 'sim'], fontsize='small', loc='lower center')
		plt.tight_layout(h_pad=0.1)
		pdf.savefig(fig)
		plt.close()
pdf.close()

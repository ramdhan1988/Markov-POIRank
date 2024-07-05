import os, sys, time, pickle, tempfile
import math, random, itertools
import pandas as pd
import numpy as np
import scipy

import sklearn 
import joblib
import cython
import pulp
from scipy.linalg import kron
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from utils import calc_dist_vec, calc_poi_info, extract_traj, getquery_id_traj_dict

random.seed(1234567890)
np.random.seed(1234567890)

LOG_ZERO = -1000

data_dir = 'data'
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb']
dat_ix = 1 # Osaka

ALPHA_SET = [0.1, 0.3, 0.5, 0.7, 0.9]  # trade-off parameters
BIN_CLUSTER = 5  # discritization parameter

RANKSVM_COST = 10  # RankSVM regularisation constant
N_JOBS = 2         # number of parallel jobs
USE_GUROBI = True # whether to use GUROBI as ILP solver


# Poi
fpoi = os.path.join(data_dir, 'poi-' + dat_suffix[dat_ix] + '.csv')
poi_all = pd.read_csv(fpoi)
poi_all.set_index('poiID', inplace=True)

# Trajectory 
ftraj = os.path.join(data_dir, 'traj-' + dat_suffix[dat_ix] + '.csv')
traj_all = pd.read_csv(ftraj)
trajid_set_all = sorted(traj_all['trajID'].unique().tolist())

# Information POI
poi_info_all = calc_poi_info(trajid_set_all, traj_all, poi_all)

# Trajectory dict # Query
traj_dict, QUERY_ID_DICT = getquery_id_traj_dict(trajid_set_all, traj_all)

# Dataset = Len Trajectoy > 2
WHOLE_SET = traj_all[traj_all['trajLen'] > 2]['trajID'].unique()
WHOLE_SET = np.random.permutation(WHOLE_SET)
splitix = int(len(WHOLE_SET)*0.5)
PART1 = WHOLE_SET[:splitix]
PART2 = WHOLE_SET[splitix:]

print('#poi in total:', len(poi_all))
print('#traj in total:', len(trajid_set_all))
print('#traj (length > 2):', traj_all[traj_all['trajLen'] > 2]['trajID'].unique().shape[0])
print('#query tuple:', len(QUERY_ID_DICT))

# Save File = POL
poi_all.to_csv('data/poi_'+dat_suffix[dat_ix]+'_all.csv', index=True)
poi_info_all.to_csv('data/poi_info_all_'+dat_suffix[dat_ix]+'_all.csv', index=True)
np.array(trajid_set_all).tofile('data/traj_id_set_'+dat_suffix[dat_ix]+'_all.csv', ",")
WHOLE_SET.tofile('data/traj_all_upper_2_'+dat_suffix[dat_ix]+'.csv', ",")
PART1.tofile('data/traj_all_upper_2_part1_'+dat_suffix[dat_ix]+'.csv', ",")
PART2.tofile('data/traj_all_upper_2_part2_'+dat_suffix[dat_ix]+'.csv', ",")


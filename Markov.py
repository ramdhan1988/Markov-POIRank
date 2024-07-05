import json
import os, sys
import pandas as pd
import numpy as np
import csv
from KMean import K_mean
from Transition_matrix import *
from utils import calc_F1, getquery_id_traj_dict

BIN_CLUSTER = 5  # discritization parameter
data_dir = 'data'
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb']
dat_ix = 1 # Osaka

# POI features used to factorise transition matrix of Markov Chain with POI features (vector) as states:
# - Category of POI
# - Popularity of POI (discritize with uniform log-scale bins, #bins <=5 )
# - The number of POI visits (discritize with uniform log-scale bins, #bins <=5 )
# - The average visit duration of POI (discritise with uniform log-scale bins, #bins <= 5)
# - The neighborhood relationship between POIs (clustering POI(lat, lon) using k-means, #clusters <= 5)
                                              
# READ FILE POI - Hasil proses preprocessing
fpoi = os.path.join(data_dir, 'poi-' + dat_suffix[dat_ix] + '.csv')
poi_all = pd.read_csv(fpoi, index_col=0)

# Trajectory 
ftraj = os.path.join(data_dir, 'traj-' + dat_suffix[dat_ix] + '.csv')
traj_all = pd.read_csv(ftraj)

# POI information
fpoi_info_all = os.path.join(data_dir, 'poi_info_all_' + dat_suffix[dat_ix] + '_all.csv')
poi_info_all = pd.read_csv(fpoi_info_all, index_col=0)
poi_train = sorted(poi_info_all.index)

# READ FILE POI - Hasil proses preprocessing
ftrajid_set_all = os.path.join(data_dir, 'traj_id_set_' + dat_suffix[dat_ix] + '_all.csv')
trajid_set_all = open(ftrajid_set_all, "r")
trajid_set_all = list(csv.reader(trajid_set_all, delimiter=","))[0]

# Trajectory per user
traj_dict, QUERY_ID_DICT = getquery_id_traj_dict(trajid_set_all, traj_all)

## 1. Transition Matrix between POI Cateogries
poi_cats = poi_all.loc[poi_train, 'poiCat'].unique().tolist()
poi_cats.sort()
POI_CAT_LIST = poi_cats
gen_transmat_cat(trajid_set_all, traj_dict, poi_info_all, POI_CAT_LIST)

## 2. Transition Matrix between POI Popularity Classes
poi_pops = poi_info_all.loc[poi_train, 'popularity']
expo_pop1 = np.log10(max(1, min(poi_pops)))
expo_pop2 = np.log10(max(poi_pops))

nbins_pop = BIN_CLUSTER
logbins_pop = np.logspace(np.floor(expo_pop1), np.ceil(expo_pop2), nbins_pop+1)
logbins_pop[0] = 0  # deal with underflow
if logbins_pop[-1] < poi_info_all['popularity'].max():
    logbins_pop[-1] = poi_info_all['popularity'].max() + 1
gen_transmat_pop(trajid_set_all, traj_dict, poi_info_all, logbins_pop=logbins_pop)[0]

## 3. Transition Matrix between the Number of POI Visit Classes
poi_visits = poi_info_all.loc[poi_train, 'nVisit']
expo_visit1 = np.log10(max(1, min(poi_visits)))
expo_visit2 = np.log10(max(poi_visits))

nbins_visit = BIN_CLUSTER
logbins_visit = np.logspace(np.floor(expo_visit1), np.ceil(expo_visit2), nbins_visit+1)
logbins_visit[0] = 0  # deal with underflow
if logbins_visit[-1] < poi_info_all['nVisit'].max():
    logbins_visit[-1] = poi_info_all['nVisit'].max() + 1
gen_transmat_visit(trajid_set_all, traj_dict, poi_info_all, logbins_visit)[0]

## 4. Transition Matrix between POI Average Visit Duration Classes
poi_durations = poi_info_all.loc[poi_train, 'avgDuration']
expo_duration1 = np.log10(max(1, min(poi_durations)))
expo_duration2 = np.log10(max(poi_durations))

nbins_duration = BIN_CLUSTER
logbins_duration = np.logspace(np.floor(expo_duration1), np.ceil(expo_duration2), nbins_duration+1)
logbins_duration[0] = 0  # deal with underflow
logbins_duration[-1] = np.power(10, expo_duration2+2)
gen_transmat_duration(trajid_set_all, traj_dict, poi_info_all, logbins_duration)[0]

## 5. Transition Matrix between POI Neighborhood Classes
X = poi_all.loc[poi_train, ['poiLon', 'poiLat']]
nclusters = BIN_CLUSTER
clusters, POI_CLUSTER_LIST, POI_CLUSTERS = K_mean (nclusters, X, poi_train)
gen_transmat_neighbor(trajid_set_all, traj_dict, poi_info_all, POI_CLUSTERS)[0]

## 6. Transition Matrix between POIs user All Transition Matrix
transmat_PoiPoi = gen_poi_logtransmat(trajid_set_all, set(poi_info_all.index), traj_dict, poi_info_all, poi_cats=POI_CAT_LIST, logbins_pop=logbins_pop, logbins_visit=logbins_visit, logbins_duration=logbins_duration, poi_clusters=POI_CLUSTERS, debug=True)

###### Main Recommend trajectories by leveraging POI-POI transition probabilities.
recdict_tran = dict()
cnt = 1
for i in range(len(trajid_set_all)):
    tid = trajid_set_all[i]
    te = traj_dict[tid]

    # trajectory is too short
    if len(te) < 3: continue
        
    trajid_list_train = trajid_set_all[:i] + trajid_set_all[i+1:]
    poi_info = calc_poi_info(trajid_list_train, traj_all, poi_all)
        
    # start/end is not in training set
    if not (te[0] in poi_info.index and te[-1] in poi_info.index): continue
    print(te, '#%d ->' % cnt); cnt += 1; sys.stdout.flush()

    # recommendation leveraging transition probabilities
    poi_logtransmat = gen_poi_logtransmat(trajid_list_train, set(poi_info.index), traj_dict, poi_info, debug=True)
    edges = poi_logtransmat.copy()

    tran_dp = find_viterbi(poi_info.copy(), edges.copy(), te[0], te[-1], len(te))
    tran_ilp = find_ILP(poi_info.copy(), edges.copy(), te[0], te[-1], len(te))

    recdict_tran[tid] = {
        'REAL':te, 
        'REC_DP':tran_dp, # Dynamic Programming - viterbi Algorithm
        'REC_ILP':tran_ilp  # ILP - Grubi
    }
    print(' '*5, 'Tran  DP:', tran_dp)
    print(' '*5, 'Tran ILP:', tran_ilp)
    sys.stdout.flush()

json_object = json.dumps(recdict_tran, indent=4)
with open("result/TranDP_ILP_Prediction_" + dat_suffix[dat_ix] +".json", "w") as outfile:
    outfile.write(json_object)

R1_tran = []; R2_tran = []; P1_tran = []; P2_tran = []; F11_tran = []; F12_tran = []; pF11_tran = []; pF12_tran = []
for tid in sorted(recdict_tran.keys()):
    recall, precision, F1 = calc_F1(recdict_tran[tid]['REAL'], recdict_tran[tid]['REC_DP'])
    R1_tran.append(recall)
    P1_tran.append(precision)
    F11_tran.append(F1)
    pF11_tran.append(calc_pairsF1(recdict_tran[tid]['REAL'], recdict_tran[tid]['REC_DP']))

    recall, precision, F1 = calc_F1(recdict_tran[tid]['REAL'], recdict_tran[tid]['REC_ILP'])
    R2_tran.append(recall)
    P2_tran.append(precision)
    F12_tran.append(F1)
    pF12_tran.append(calc_pairsF1(recdict_tran[tid]['REAL'], recdict_tran[tid]['REC_ILP']))

matric_evaluation = {
    "Dynamic_programming": {
        "Recall": [ float(np.mean(R1_tran)) , float(np.std(R1_tran)) ],
        "Precision": [ float(np.mean(P1_tran)), float(np.std(P1_tran)) ],
        "F1": [ float(np.mean(F11_tran)), float(np.std(F11_tran)) ],
        "pairsF1" : [ float(np.mean(pF11_tran)), float(np.std(pF11_tran)) ] 
    }, 
    "Integer_linear_programming": {
        "Recall": [ float(np.mean(R2_tran)) , float(np.std(R2_tran)) ],
        "Precision": [ float(np.mean(P2_tran)), float(np.std(P2_tran)) ],
        "F1": [ float(np.mean(F12_tran)), float(np.std(F12_tran)) ],
        "pairsF1" : [ float(np.mean(pF12_tran)), float(np.std(pF12_tran)) ] 
    }
};

json_matric_evaluation = json.dumps(matric_evaluation, indent=4)
with open("result/TranDP_ILP_matric_evaluation_" + dat_suffix[dat_ix] +".json", "w") as outfile:
    outfile.write(json_matric_evaluation)

print('Dynamic_programming: Recall (%.3f, %.3f)' % (np.mean(R1_tran), np.std(R1_tran)))
print('Dynamic_programming: Precision (%.3f, %.3f)' % (np.mean(P1_tran), np.std(P1_tran)))
print('Dynamic_programming: F1 (%.3f, %.3f)' % (np.mean(F11_tran), np.std(F11_tran)))
print('Dynamic_programming: pairsF1 (%.3f, %.3f)' %  (np.mean(pF11_tran), np.std(pF11_tran)))

print('Integer_linear_programming: Recall (%.3f, %.3f)' % (np.mean(R2_tran), np.std(R2_tran)))
print('Integer_linear_programming: Precision (%.3f, %.3f)' % (np.mean(P2_tran), np.std(P2_tran)))
print('Integer_linear_programming: F1 (%.3f, %.3f)' % (np.mean(F12_tran), np.std(F12_tran)))
print('Integer_linear_programming: pairsF1 (%.3f, %.3f)' % (np.mean(pF12_tran), np.std(pF12_tran)))
import os, tempfile, sys
import numpy as np
import pandas as pd 
from KMean import K_mean
import csv
import json
from svmRank import Ranksvm
from utils import calc_dist_vec, gen_test_df, gen_train_df, getquery_id_traj_dict, calc_F1, calc_pairsF1, true_F1, true_pairsF1

ranksvm_dir = 'libsvm-ranksvm-3.32'  # directory that contains rankSVM binaries: train, predict, svm-scale
data_dir = 'data'
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb']
dat_ix = 0 # Osaka

RANKSVM_COST = 10  # RankSVM regularisation constant
N_JOBS = 2         # number of parallel jobs
BIN_CLUSTER = 5  # discritization parameter

# POI Features used for ranking, given query (`startPOI`, `endPOI`, `nPOI`):
###
# -`category`: one-hot encoding of POI category, encode `True` as `1` and `False` as `-1`
# -`neighbourhood`: one-hot encoding of POI cluster, encode `True` as `1` and `False` as `-1`
# -`popularity`: log of POI popularity, i.e., the number of distinct users that visited the POI
# -`nVisit`: log of the total number of visit by all users
# -`avgDuration`: log of average POI visit duration
# -`trajLen`: trajectory length, i.e., the number of POIs `nPOI` in trajectory, copy from query
# -`sameCatStart`: 1 if POI category is the same as that of `startPOI`, -1 otherwise
# -`sameCatEnd`: 1 if POI category is the same as that of `endPOI`, -1 otherwise
# -`distStart`: distance (haversine formula) from `startPOI`
# -`distEnd`: distance from `endPOI`
# -`diffPopStart`: difference in POI popularity from `startPOI` (NO LOG as it could be negative)
# -`diffPopEnd`: difference in POI popularity from `endPOI`
# -`diffNVisitStart`: difference in the total number of visit from `startPOI`
# -`diffNVisitEnd`: difference in the total number of visit from `endPOI`
# -`diffDurationStart`: difference in average POI visit duration from the actual duration spent at `startPOI`
# -`diffDurationEnd`: difference in average POI visit duration from the actual duration spent at `endPOI`
# -`sameNeighbourhoodStart`: 1 if POI resides in the same cluster as that of `startPOI`, -1 otherwise
# -`sameNeighbourhoodEnd`: 1 if POI resides in the same cluster as that of `endPOI`, -1 otherwise


# READ FILE POI - Hasil proses preprocessing
fpoi = os.path.join(data_dir, 'poi-' + dat_suffix[dat_ix] + '.csv')
poi_all = pd.read_csv(fpoi, index_col=0)

# Trajectory 
ftraj = os.path.join(data_dir, 'traj-' + dat_suffix[dat_ix] + '.csv')
traj_all = pd.read_csv(ftraj)

# Distance between POI
POI_DISTMAT = pd.DataFrame(data=np.zeros((poi_all.shape[0], poi_all.shape[0]), dtype=float), index=poi_all.index, columns=poi_all.index)
for ix in poi_all.index:
    POI_DISTMAT.loc[ix] = calc_dist_vec(poi_all.loc[ix, 'poiLon'], poi_all.loc[ix, 'poiLat'], poi_all['poiLon'], poi_all['poiLat'])

# READ FILE POI - Hasil proses preprocessing
ftrajid_set_all = os.path.join(data_dir, 'traj_id_set_' + dat_suffix[dat_ix] + '_all.csv')
trajid_set_all = open(ftrajid_set_all, "r")
trajid_set_all = list(csv.reader(trajid_set_all, delimiter=","))[0]

# POI information
fpoi_info_all = os.path.join(data_dir, 'poi_info_all_' + dat_suffix[dat_ix] + '_all.csv')
poi_info = pd.read_csv(fpoi_info_all, index_col=0)
poi_train = sorted(poi_info.index)

# Category POI
poi_cats = poi_all.loc[poi_train, 'poiCat'].unique().tolist()

recdict_rank = dict()
cnt = 1
traj_dict, QUERY_ID_DICT = getquery_id_traj_dict(trajid_set_all, traj_all)

# Get Cluster 
X = poi_all.loc[poi_train, ['poiLon', 'poiLat']]
nclusters = BIN_CLUSTER
clusters, POI_CLUSTER_LIST, POI_CLUSTERS = K_mean (nclusters, X, poi_train)

print(dat_suffix[dat_ix])
print(f'trajid_set_all ', len(trajid_set_all))
print(f'traj_dict ', len(traj_dict))
print(f'traj_all ', len(traj_all))
print(f'poi_all ', len(poi_all))
print(f'POI_CLUSTERS ', len(POI_CLUSTERS))
print(f'POI_CAT_LIST ', len(poi_cats))
print(f'POI_CLUSTER_LIST ', len(POI_CLUSTER_LIST))

for i in range(len(trajid_set_all)): # 
    tid = trajid_set_all[i]
    te = traj_dict[tid]

    # Trajectory is too short
    if len(te) < 3: continue
    trajid_list_train = trajid_set_all[:i] + trajid_set_all[i+1:]

    # Start/end is not in training set
    if not (te[0] in poi_info.index and te[-1] in poi_info.index): continue

    print(te, '#%d ->' % cnt)
    print(f'of ', len(trajid_set_all))
    cnt += 1
    sys.stdout.flush()

    # Library
    ranksvm = Ranksvm(ranksvm_dir, useLinear=False, debug=False, city=dat_suffix[dat_ix])
    # recommendation leveraging ranking
    train_df    = gen_train_df(trajid_list_train, 
                               traj_dict, 
                               poi_info, 
                               poi_clusters=POI_CLUSTERS, 
                               cats=poi_cats, 
                               clusters=POI_CLUSTER_LIST, 
                               n_jobs=N_JOBS, 
                               poi_distmat=POI_DISTMAT, 
                               query_id_dict=QUERY_ID_DICT)
    ranksvm.train(train_df, cost=RANKSVM_COST)
    test_df     = gen_test_df(te[0],
                            te[-1],
                            len(te),
                            poi_info,
                            poi_clusters=POI_CLUSTERS,
                            cats=poi_cats,
                            clusters=POI_CLUSTER_LIST,
                            POI_DISTMAT=POI_DISTMAT, 
                            QUERY_ID_DICT=QUERY_ID_DICT, 
                            poi_all = poi_all)
    rank_df = ranksvm.predict(test_df)

    # POI popularity based ranking
    poi_info.sort_values(by='popularity', ascending=False, inplace=True)
    ranks1 = poi_info.index.tolist()
    rank_pop = [te[0]] + [x for x in ranks1 if x not in {te[0], te[-1]}][:len(te)-2] + [te[-1]]

    # POI feature based ranking
    rank_df.sort_values(by='rank', ascending=False, inplace=True)
    ranks2 = rank_df.index.tolist()
    rank_feature = [te[0]] + [x for x in ranks2 if x not in {te[0], te[-1]}][:len(te)-2] + [te[-1]]

    recdict_rank[tid] = {
        'REAL':te, 
        'REC_POP':rank_pop, 
        'REC_FEATURE':rank_feature
    }
    print(' '*5, 'Rank POP:', rank_pop)
    print(' '*5, 'Rank POI:', rank_feature)
    sys.stdout.flush()

json_object = json.dumps(recdict_rank, indent=4)
with open("result/RankPOP_Prediction_" + dat_suffix[dat_ix] +"_"+str(RANKSVM_COST)+".json", "w") as outfile:
    outfile.write(json_object)

R1_rank = []; R2_rank = [] 
P1_rank = []; P2_rank = []
F11_rank = []; F12_rank = []
pF11_rank = []; pF12_rank = []
true_F11_rank = []; true_F12_rank = []
true_pF11_rank = []; true_pF12_rank = []

for key in sorted(recdict_rank.keys()):
    recall, precision, F1 = calc_F1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_POP'])
    R1_rank.append(recall)
    P1_rank.append(precision)
    F11_rank.append(F1)
    pF11_rank.append(calc_pairsF1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_POP']))
    true_F11_rank.append(true_F1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_POP']))
    true_pF11_rank.append(true_pairsF1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_POP']))

    recall, precision, F1 = calc_F1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_FEATURE'])
    R2_rank.append(recall)
    P2_rank.append(precision)
    F12_rank.append(F1)
    pF12_rank.append(calc_pairsF1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_FEATURE']))
    true_F12_rank.append(true_F1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_FEATURE']))
    true_pF12_rank.append(true_pairsF1(recdict_rank[key]['REAL'], recdict_rank[key]['REC_FEATURE']))

matric_evaluation = {
    "Popularity": {
        "Recall": [ float(np.mean(R1_rank)) , float(np.std(R1_rank)) ],
        "Precision": [ float(np.mean(P1_rank)), float(np.std(P1_rank)) ],
        "F1": [ float(np.mean(F11_rank)), float(np.std(F11_rank)) ],
        "pairsF1" : [ float(np.mean(pF11_rank)), float(np.std(pF11_rank)) ],
        "true_F1" : [ float(np.mean(true_F11_rank)), float(np.std(true_F11_rank)) ], 
        "true_pairF1" : [ float(np.mean(true_pF11_rank)), float(np.std(true_pF11_rank)) ] 
    }, 
    "POIRank": {
        "Recall": [ float(np.mean(R2_rank)) , float(np.std(R2_rank)) ],
        "Precision": [ float(np.mean(P2_rank)), float(np.std(P2_rank)) ],
        "F1": [ float(np.mean(F12_rank)), float(np.std(F12_rank)) ],
        "pairsF1" : [ float(np.mean(pF12_rank)), float(np.std(pF12_rank)) ] ,
        "true_F1" : [ float(np.mean(true_F12_rank)), float(np.std(true_F12_rank)) ], 
        "true_pairF1" : [ float(np.mean(true_pF12_rank)), float(np.std(true_pF12_rank)) ] 
    }
};

json_matric_evaluation = json.dumps(matric_evaluation, indent=4)
with open("result/RankPOP_matric_evaluation_" + dat_suffix[dat_ix] +".json", "w") as outfile:
    outfile.write(json_matric_evaluation)

print('Rank POP: Recall (%.3f, %.3f)' % (np.mean(R1_rank), np.std(R1_rank)))
print('Rank POP: Precision (%.3f, %.3f)' % (np.mean(P1_rank), np.std(P1_rank)))
print('Rank POP: F1 (%.3f, %.3f)' % (np.mean(F11_rank), np.std(F11_rank)))
print('Rank POP: pairsF1 (%.3f, %.3f)' %  (np.mean(pF11_rank), np.std(pF11_rank)))
print('Rank POP: true F1 (%.3f, %.3f)' %  (np.mean(true_F11_rank), np.std(true_F11_rank)))
print('Rank POP: true pairsF1 (%.3f, %.3f)' %  (np.mean(true_pF11_rank), np.std(true_pF11_rank)))

print('Rank POI: Recall (%.3f, %.3f)' % (np.mean(R2_rank), np.std(R2_rank)))
print('Rank POI: Precision (%.3f, %.3f)' % (np.mean(P2_rank), np.std(P2_rank)))
print('Rank POI: F1 (%.3f, %.3f)' % (np.mean(F12_rank), np.std(F12_rank)))
print('Rank POI: pairsF1 (%.3f, %.3f)' % (np.mean(pF12_rank), np.std(pF12_rank)))
print('Rank POI: true F1 (%.3f, %.3f)' % (np.mean(true_F12_rank), np.std(true_F12_rank)))
print('Rank POI: true pairsF1 (%.3f, %.3f)' % (np.mean(true_pF12_rank), np.std(true_pF12_rank)))
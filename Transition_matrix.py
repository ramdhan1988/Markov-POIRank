
import csv
import json
import os
import sys
import pandas as pd
import numpy as np
import pulp
import math, random, itertools
from scipy.linalg import kron

from KMean import K_mean
from svmRank import Ranksvm
from utils import calc_F1, calc_dist_vec, calc_pairsF1, calc_poi_info, gen_test_df, gen_train_df, getquery_id_traj_dict, print_progress, true_F1, true_pairsF1


ranksvm_dir = 'libsvm-ranksvm-3.32'  # directory that contains rankSVM binaries: train, predict, svm-scale
data_dir = 'data'
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro']
dat_ix = 3 # Osaka

RANKSVM_COST = 10  # RankSVM regularisation constant
N_JOBS = 2         # number of parallel jobs
BIN_CLUSTER = 5  # discritization parameter
USE_GUROBI = True # whether to use GUROBI as ILP solver
LOG_ZERO = -1000


def normalise_transmat(transmat_cnt):
    transmat = transmat_cnt.copy()
    assert(isinstance(transmat, pd.DataFrame))
    for row in range(transmat.index.shape[0]):
        rowsum = np.sum(transmat.iloc[row] + 1)
        assert(rowsum > 0)
        transmat.iloc[row] = (transmat.iloc[row] + 1) / rowsum
    return transmat

def gen_transmat_cat(trajid_list, traj_dict, poi_info, poi_cats):
    transmat_cat_cnt = pd.DataFrame(data=np.zeros((len(poi_cats), len(poi_cats)), 
                                                  dtype=float),
                                    columns=poi_cats, 
                                    index=poi_cats)
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t)-1):
                p1 = t[pi]
                p2 = t[pi+1]
                assert(p1 in poi_info.index and p2 in poi_info.index)
                cat1 = poi_info.loc[p1, 'poiCat']
                cat2 = poi_info.loc[p2, 'poiCat']
                transmat_cat_cnt.loc[cat1, cat2] += 1
    return normalise_transmat(transmat_cat_cnt)

def gen_transmat_pop(trajid_list, traj_dict, poi_info, logbins_pop):
    nbins = len(logbins_pop) - 1
    transmat_pop_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=float), \
                                    columns=np.arange(1, nbins+1), index=np.arange(1, nbins+1))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t)-1):
                p1 = t[pi]
                p2 = t[pi+1]
                assert(p1 in poi_info.index and p2 in poi_info.index)
                pop1 = poi_info.loc[p1, 'popularity']
                pop2 = poi_info.loc[p2, 'popularity']
                pc1, pc2 = np.digitize([pop1, pop2], logbins_pop)
                transmat_pop_cnt.loc[pc1, pc2] += 1
    return normalise_transmat(transmat_pop_cnt), logbins_pop

def gen_transmat_visit(trajid_list, traj_dict, poi_info, logbins_visit):
    nbins = len(logbins_visit) - 1
    transmat_visit_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=float), \
                                      columns=np.arange(1, nbins+1), index=np.arange(1, nbins+1))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t)-1):
                p1 = t[pi]
                p2 = t[pi+1]
                assert(p1 in poi_info.index and p2 in poi_info.index)
                visit1 = poi_info.loc[p1, 'nVisit']
                visit2 = poi_info.loc[p2, 'nVisit']
                vc1, vc2 = np.digitize([visit1, visit2], logbins_visit)
                transmat_visit_cnt.loc[vc1, vc2] += 1
    return normalise_transmat(transmat_visit_cnt), logbins_visit

def gen_transmat_duration(trajid_list, traj_dict, poi_info, logbins_duration):
    nbins = len(logbins_duration) - 1
    transmat_duration_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=float), \
                                         columns=np.arange(1, nbins+1), index=np.arange(1, nbins+1))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t)-1):
                p1 = t[pi]
                p2 = t[pi+1]
                assert(p1 in poi_info.index and p2 in poi_info.index)
                d1 = poi_info.loc[p1, 'avgDuration']
                d2 = poi_info.loc[p2, 'avgDuration']
                dc1, dc2 = np.digitize([d1, d2], logbins_duration)
                transmat_duration_cnt.loc[dc1, dc2] += 1
    return normalise_transmat(transmat_duration_cnt), logbins_duration

def gen_transmat_neighbor(trajid_list, traj_dict, poi_info, poi_clusters):
    nclusters = len(poi_clusters['clusterID'].unique())
    transmat_neighbor_cnt = pd.DataFrame(data=np.zeros((nclusters, nclusters), dtype=float), \
                                         columns=np.arange(nclusters), index=np.arange(nclusters))
    for tid in trajid_list:
        t = traj_dict[tid]
        if len(t) > 1:
            for pi in range(len(t)-1):
                p1 = t[pi]
                p2 = t[pi+1]
                assert(p1 in poi_info.index and p2 in poi_info.index)
                c1 = poi_clusters.loc[p1, 'clusterID']
                c2 = poi_clusters.loc[p2, 'clusterID']
                transmat_neighbor_cnt.loc[c1, c2] += 1
    return normalise_transmat(transmat_neighbor_cnt), poi_clusters

def gen_poi_logtransmat(trajid_list, poi_set, traj_dict, poi_info, poi_cats, logbins_pop, logbins_visit, logbins_duration, poi_clusters, debug=False):
    print('gen_transmat_cat...')
    transmat_cat                        = gen_transmat_cat(trajid_list, traj_dict, poi_info, poi_cats)
    print('gen_transmat_pop...')
    transmat_pop,      logbins_pop      = gen_transmat_pop(trajid_list, traj_dict, poi_info, logbins_pop)
    print('gen_transmat_visit...')
    transmat_visit,    logbins_visit    = gen_transmat_visit(trajid_list, traj_dict, poi_info, logbins_visit)
    print('gen_transmat_duration...')
    transmat_duration, logbins_duration = gen_transmat_duration(trajid_list, traj_dict, poi_info, logbins_duration)
    print('gen_transmat_neighbor...')
    transmat_neighbor, poi_clusters     = gen_transmat_neighbor(trajid_list, traj_dict, poi_info, poi_clusters)

    # Kronecker product
    print("Kronecker product...")
    transmat_ix = list(itertools.product(transmat_cat.index, 
                                         transmat_pop.index, 
                                         transmat_visit.index,
                                         transmat_duration.index, 
                                         transmat_neighbor.index))
    transmat_value = transmat_cat.values
    for transmat in [transmat_pop, transmat_visit, transmat_duration, transmat_neighbor]:
        transmat_value = kron(transmat_value, transmat.values)
    transmat_feature = pd.DataFrame(data=transmat_value, index=transmat_ix, columns=transmat_ix)
    
    poi_train = sorted(poi_set)
    feature_names = ['poiCat', 'popularity', 'nVisit', 'avgDuration', 'clusterID']
    poi_features = pd.DataFrame(data=np.zeros((len(poi_train), len(feature_names))), \
                                columns=feature_names, index=poi_train)
    poi_features.index.name = 'poiID'
    poi_features['poiCat'] = poi_info.loc[poi_train, 'poiCat']
    poi_features['popularity'] = np.digitize(poi_info.loc[poi_train, 'popularity'], logbins_pop)
    poi_features['nVisit'] = np.digitize(poi_info.loc[poi_train, 'nVisit'], logbins_visit)
    poi_features['avgDuration'] = np.digitize(poi_info.loc[poi_train, 'avgDuration'], logbins_duration)
    poi_features['clusterID'] = poi_clusters.loc[poi_train, 'clusterID']
    
    # shrink the result of Kronecker product and deal with POIs with the same features
    poi_logtransmat = pd.DataFrame(data=np.zeros((len(poi_train), len(poi_train)), dtype=float), \
                                   columns=poi_train, index=poi_train)
    for p1 in poi_logtransmat.index:
        rix = tuple(poi_features.loc[p1])
        for p2 in poi_logtransmat.columns:
            cix = tuple(poi_features.loc[p2])
            value_ = transmat_feature.loc[(rix,), (cix,)]
            poi_logtransmat.loc[p1, p2] = value_.values[0, 0]
    
    # group POIs with the same features
    features_dup = dict()
    for poi in poi_features.index:
        key = tuple(poi_features.loc[poi])
        if key in features_dup:
            features_dup[key].append(poi)
        else:
            features_dup[key] = [poi]
    if debug == True:
        for key in sorted(features_dup.keys()):
            print(key, '->', features_dup[key])
            
    # deal with POIs with the same features
    for feature in sorted(features_dup.keys()):
        n = len(features_dup[feature])
        if n > 1:
            group = features_dup[feature]
            v1 = poi_logtransmat.loc[group[0], group[0]]  # transition value of self-loop of POI group
            
            # divide incoming transition value (i.e. unnormalised transition probability) uniformly among group members
            for poi in group:
                poi_logtransmat[poi] /= n
                
            # outgoing transition value has already been duplicated (value copied above)
            
            # duplicate & divide transition value of self-loop of POI group uniformly among all outgoing transitions,
            # from a POI to all other POIs in the same group (excluding POI self-loop)
            v2 = v1 / (n - 1)
            for pair in itertools.permutations(group, 2):
                poi_logtransmat.loc[pair[0], pair[1]] = v2
                            
    # normalise each row
    print("Normalise each row...")
    for p1 in poi_logtransmat.index:
        poi_logtransmat.loc[p1, p1] = 0
        rowsum = poi_logtransmat.loc[p1].sum()
        assert(rowsum > 0)
        logrowsum = np.log10(rowsum)
        for p2 in poi_logtransmat.columns:
            if p1 == p2:
                poi_logtransmat.loc[p1, p2] = LOG_ZERO  # deal with log(0) explicitly
            else:
                poi_logtransmat.loc[p1, p2] = np.log10(poi_logtransmat.loc[p1, p2]) - logrowsum
    
    return poi_logtransmat

def find_viterbi(V, E, ps, pe, L, withNodeWeight=False, alpha=0.5, withStartEndIntermediate=False):
    assert(isinstance(V, pd.DataFrame))
    assert(isinstance(E, pd.DataFrame))
    assert(ps in V.index)
    assert(pe in V.index)
    assert(2 < L <= V.index.shape[0])  
    if withNodeWeight == True:
        assert(0 < alpha < 1)
        beta = 1 - alpha
    else:
        alpha = 0
        beta = 1
        weightkey = 'weight'
        if weightkey not in V.columns:
            V['weight'] = 1  # dummy weights, will not be used as alpha=0
    if withStartEndIntermediate == True:
        excludes = [ps]
    else:
        excludes = [ps, pe]
    
    A = pd.DataFrame(data=np.zeros((L-1, V.shape[0]), dtype=float), columns=V.index, index=np.arange(2, L+1))
    B = pd.DataFrame(data=np.zeros((L-1, V.shape[0]), dtype=int),   columns=V.index, index=np.arange(2, L+1))

    A += np.inf
    for v in V.index:            
        if v not in excludes:
            A.loc[2, v] = alpha * (V.loc[ps, 'weight'] + V.loc[v, 'weight']) + beta * E.loc[ps, v]  # ps--v
            B.loc[2, v] = ps
    
    for l in range(3, L+1):
        for v in V.index:
            if withStartEndIntermediate == True: # ps-~-v1---v 
                values = [A.loc[l-1, v1] + alpha * V.loc[v, 'weight'] + beta * E.loc[v1, v] for v1 in V.index]
            else: # ps-~-v1---v 
                values = [A.loc[l-1, v1] + alpha * V.loc[v, 'weight'] + beta * E.loc[v1, v] \
                          if v1 not in [ps, pe] else -np.inf for v1 in V.index] # exclude ps and pe
            
            maxix = np.argmax(values)
            A.loc[l, v] = values[maxix]
            B.loc[l, v] = V.index[maxix]
          
    path = [pe]
    v = path[-1]
    l = L
    while l >= 2:
        path.append( int( B.loc[l, v] ) )
        v = path[-1]
        l -= 1
    path.reverse()
    return path

def find_ILP(V, E, ps, pe, L, withNodeWeight=False, alpha=0.5):
    assert(isinstance(V, pd.DataFrame))
    assert(isinstance(E, pd.DataFrame))
    assert(ps in V.index)
    assert(pe in V.index)
    assert(2 < L <= V.index.shape[0])
    if withNodeWeight == True:
        assert(0 < alpha < 1)
    beta = 1 - alpha
    p0 = str(ps); pN = str(pe); N = V.index.shape[0]

    # REF: pythonhosted.org/PuLP/index.html
    pois = [str(p) for p in V.index] # create a string list for each POI
    pb = pulp.LpProblem('MostLikelyTraj', pulp.LpMaximize) # create problem
    # visit_i_j = 1 means POI i and j are visited in sequence
    visit_vars = pulp.LpVariable.dicts('visit', (pois, pois), 0, 1, pulp.LpInteger) 
    # a dictionary contains all dummy variables
    dummy_vars = pulp.LpVariable.dicts('u', [x for x in pois if x != p0], 2, N, pulp.LpInteger)
    
    # add objective
    objlist = []
    if withNodeWeight == True:
        objlist.append(alpha * V.loc[int(p0), 'weight'])
    for pi in [x for x in pois if x != pN]:     # from
        for pj in [y for y in pois if y != p0]: # to
            if withNodeWeight == True:
                objlist.append(visit_vars[pi][pj] * (alpha * V.loc[int(pj), 'weight'] + beta * E.loc[int(pi), int(pj)]))
            else:
                objlist.append(visit_vars[pi][pj] * E.loc[int(pi), int(pj)])
    pb += pulp.lpSum(objlist), 'Objective'
    
    # add constraints, each constraint should be in ONE line
    pb += pulp.lpSum([visit_vars[p0][pj] for pj in pois if pj != p0]) == 1, 'StartAt_p0'
    pb += pulp.lpSum([visit_vars[pi][pN] for pi in pois if pi != pN]) == 1, 'EndAt_pN'
    if p0 != pN:
        pb += pulp.lpSum([visit_vars[pi][p0] for pi in pois]) == 0, 'NoIncoming_p0'
        pb += pulp.lpSum([visit_vars[pN][pj] for pj in pois]) == 0, 'NoOutgoing_pN'
    pb += pulp.lpSum([visit_vars[pi][pj] for pi in pois if pi != pN for pj in pois if pj != p0]) == L-1, 'Length'
    for pk in [x for x in pois if x not in {p0, pN}]:
        pb += pulp.lpSum([visit_vars[pi][pk] for pi in pois if pi != pN]) == \
              pulp.lpSum([visit_vars[pk][pj] for pj in pois if pj != p0]), 'ConnectedAt_' + pk
        pb += pulp.lpSum([visit_vars[pi][pk] for pi in pois if pi != pN]) <= 1, 'Enter_' + pk + '_AtMostOnce'
        pb += pulp.lpSum([visit_vars[pk][pj] for pj in pois if pj != p0]) <= 1, 'Leave_' + pk + '_AtMostOnce'
    for pi in [x for x in pois if x != p0]:
        for pj in [y for y in pois if y != p0]:
            pb += dummy_vars[pi] - dummy_vars[pj] + 1 <= (N - 1) * (1 - visit_vars[pi][pj]), \
                    'SubTourElimination_' + pi + '_' + pj
    #pb.writeLP("traj_tmp.lp")
    # solve problem: solver should be available in PATH
    if USE_GUROBI == True:
        gurobi_options = [('TimeLimit', '7200'), ('Threads', str(N_JOBS)), ('NodefileStart', '0.2'), ('Cuts', '2')]
        pb.solve(pulp.GUROBI_CMD(path='gurobi_cl', options=gurobi_options)) # GUROBI
    else:
#         pb.solve(pulp.COIN_CMD(path='cbc', options=['-threads', str(N_JOBS), '-strategy', '1', '-maxIt', '2000000']))#CBC
#         pb.solve(pulp.getSolver('PULP_CBC_CMD', options=['-threads', str(N_JOBS), '-strategy', '1', '-maxIt', '2000000']))#CBC
        pb.solve(pulp.getSolver('PULP_CBC_CMD'))#CBC
    visit_mat = pd.DataFrame(data=np.zeros((len(pois), len(pois)), dtype=float), index=pois, columns=pois)
    for pi in pois:
        for pj in pois: visit_mat.loc[pi, pj] = visit_vars[pi][pj].varValue

    # build the recommended trajectory
    recseq = [p0]
    while True:
        pi = recseq[-1]
        pj = visit_mat.loc[pi].idxmax()
        assert(round(visit_mat.loc[pi, pj]) == 1)
        recseq.append(pj)
        if pj == pN: return [int(x) for x in recseq]

def cv_choose_alpha(alpha_set, validation_set, short_traj_set, traj_dict, traj_all, poi_all, poi_cats, poi_clusters, poi_clusters_list):
    assert(len(set(validation_set) & set(short_traj_set)) == 0)  # NO intersection
    best_score = 0
    best_alpha = 0
    cnt = 1; 
    total = len(validation_set) * len(alpha_set)
    for alpha_i in alpha_set:
        print(alpha_i)
        
        scores = []
        for i in range(len(validation_set)):
            tid = validation_set[i]
            te = traj_dict[tid]
            assert(len(te) > 2)
            
            trajid_list_train = list(short_traj_set) + list(validation_set[:i]) + list(validation_set[i+1:])
            poi_info = calc_poi_info(trajid_list_train, traj_all, poi_all)
            
            # start/end is not in training set
            if not (te[0] in poi_info.index and te[-1] in poi_info.index): 
                print('Failed cross-validation instance:', te)
                continue
        
            ranksvm = Ranksvm(ranksvm_dir, useLinear=False)
            train_df = gen_train_df(trajid_list_train, \
                                    traj_dict, poi_info, \
                                    poi_clusters=poi_clusters, \
                                    cats=poi_cats, \
                                    clusters=poi_clusters_list, \
                                    n_jobs=N_JOBS)
            ranksvm.train(train_df, cost=RANKSVM_COST)
            test_df = gen_test_df(te[0], \
                                    te[-1], \
                                    len(te), \
                                    poi_info, \
                                    poi_clusters=poi_clusters, \
                                    cats=poi_cats, \
                                    clusters=poi_clusters_list)
            rank_df = ranksvm.predict(test_df)

            poi_logtransmat = gen_poi_logtransmat(trajid_list_train, set(poi_info.index), traj_dict, poi_info)
            edges = poi_logtransmat.copy()
            
            nodes = rank_df.copy()
            nodes['weight'] = np.log10(nodes['probability'])
            nodes.drop('probability', axis=1, inplace=True)
            comb = find_viterbi(nodes, edges, te[0], te[-1], len(te), withNodeWeight=True, alpha=alpha_i)
            
            scores.append(calc_pairsF1(te, comb))
            
            print_progress(cnt, total); cnt += 1
            
        mean_score = np.mean(scores)
        print('alpha:', alpha_i, ' mean pairs-F1:', mean_score)
        if best_score > mean_score: continue
        best_score = mean_score
        best_alpha = alpha_i
            
    return best_alpha

def get_logbins_pop(poi_info_all, poi_train):
    poi_pops = poi_info_all.loc[poi_train, 'popularity']
    expo_pop1 = np.log10(max(1, min(poi_pops)))
    expo_pop2 = np.log10(max(poi_pops))
    nbins_pop = BIN_CLUSTER
    logbins_pop = np.logspace(np.floor(expo_pop1), np.ceil(expo_pop2), nbins_pop+1)
    logbins_pop[0] = 0  # deal with underflow
    if logbins_pop[-1] < poi_info_all['popularity'].max():
        logbins_pop[-1] = poi_info_all['popularity'].max() + 1
    return logbins_pop

def get_logbins_visit(poi_info_all, poi_train):
    poi_visits = poi_info_all.loc[poi_train, 'nVisit']
    expo_visit1 = np.log10(max(1, min(poi_visits)))
    expo_visit2 = np.log10(max(poi_visits))
    nbins_visit = BIN_CLUSTER
    logbins_visit = np.logspace(np.floor(expo_visit1), np.ceil(expo_visit2), nbins_visit+1)
    logbins_visit[0] = 0  # deal with underflow
    if logbins_visit[-1] < poi_info_all['nVisit'].max():
        logbins_visit[-1] = poi_info_all['nVisit'].max() + 1
    return logbins_visit

def get_logbins_duration(poi_info_all, poi_train):
    poi_durations = poi_info_all.loc[poi_train, 'avgDuration']
    expo_duration1 = np.log10(max(1, min(poi_durations)))
    expo_duration2 = np.log10(max(poi_durations))
    nbins_duration = BIN_CLUSTER
    logbins_duration = np.logspace(np.floor(expo_duration1), np.ceil(expo_duration2), nbins_duration+1)
    logbins_duration[0] = 0  # deal with underflow
    logbins_duration[-1] = np.power(10, expo_duration2+2)
    return logbins_duration

### READ DATA ###

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

recdict_tran = dict()
cnt = 1
traj_dict, QUERY_ID_DICT = getquery_id_traj_dict(trajid_set_all, traj_all)
logbins_pop = get_logbins_pop(poi_info_all=poi_info, poi_train=poi_train)
logbins_visit =  get_logbins_visit(poi_info_all=poi_info, poi_train=poi_train)
logbins_duration = get_logbins_duration(poi_info_all=poi_info, poi_train=poi_train)

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

for i in range(len(trajid_set_all)):
    tid = trajid_set_all[i]
    te = traj_dict[tid]
    
    # trajectory is too short
    if len(te) < 3: continue
        
    trajid_list_train = trajid_set_all[:i] + trajid_set_all[i+1:]

    print('calc_poi_info...')
    poi_info = calc_poi_info(trajid_list_train, traj_all, poi_all)

    # start/end is not in training set
    if not (te[0] in poi_info.index and te[-1] in poi_info.index): continue
    
    print(te, '#%d ->' % cnt)
    cnt += 1
    sys.stdout.flush()
    
    # recommendation leveraging transition probabilities
    poi_logtransmat = gen_poi_logtransmat(trajid_list_train, 
                                          set(poi_info.index), 
                                          traj_dict, 
                                          poi_info,  
                                          poi_cats, 
                                          logbins_pop, 
                                          logbins_visit, 
                                          logbins_duration, 
                                          poi_clusters=POI_CLUSTERS, 
                                          debug=False)
    edges = poi_logtransmat.copy()

    tran_dp = find_viterbi(poi_info.copy(), 
                           edges.copy(), 
                           te[0], 
                           te[-1], 
                           len(te))
    
    tran_ilp = find_ILP(poi_info.copy(),
                        edges.copy(),
                        te[0],
                        te[-1],
                        len(te))
    
    recdict_tran[tid] = {
        'REAL':te, 
        'REC_Markov_DP':tran_dp, 
        'REC_MarkovPath_ILP':tran_ilp
    }
    
    print(' '*5, 'Tran  DP (Markov):', tran_dp)
    print(' '*5, 'Tran ILP (Markov Path):', tran_ilp)
    sys.stdout.flush()

json_object = json.dumps(recdict_tran, indent=4)
with open("result/Markov_Prediction_" + dat_suffix[dat_ix] +".json", "w") as outfile:
    outfile.write(json_object)

R1_trans = []; R2_trans = [] 
P1_trans = []; P2_trans = []
F11_trans = []; F12_trans = []
pF11_trans = []; pF12_trans = []
true_F11_trans = []; true_F12_trans = []
true_pF11_trans = []; true_pF12_trans = []

for key in sorted(recdict_tran.keys()):
    recall, precision, F1 = calc_F1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_Markov_DP'])
    R1_trans.append(recall)
    P1_trans.append(precision)
    F11_trans.append(F1)
    pF11_trans.append(calc_pairsF1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_Markov_DP']))
    true_F11_trans.append(true_F1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_Markov_DP']))
    true_pF11_trans.append(true_pairsF1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_Markov_DP']))

    recall, precision, F1 = calc_F1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_MarkovPath_ILP'])
    R2_trans.append(recall)
    P2_trans.append(precision)
    F12_trans.append(F1)
    pF12_trans.append(calc_pairsF1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_MarkovPath_ILP']))
    true_F12_trans.append(true_F1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_MarkovPath_ILP']))
    true_pF12_trans.append(true_pairsF1(recdict_tran[key]['REAL'], recdict_tran[key]['REC_MarkovPath_ILP']))

matric_evaluation = {
    "Markov_DP": {
        "Recall": [ float(np.mean(R1_trans)) , float(np.std(R1_trans)) ],
        "Precision": [ float(np.mean(P1_trans)), float(np.std(P1_trans)) ],
        "F1": [ float(np.mean(F11_trans)), float(np.std(F11_trans)) ],
        "pairsF1" : [ float(np.mean(pF11_trans)), float(np.std(pF11_trans)) ],
        "true_F1" : [ float(np.mean(true_F11_trans)), float(np.std(true_F11_trans)) ], 
        "true_pairF1" : [ float(np.mean(true_pF11_trans)), float(np.std(true_pF11_trans)) ] 
    }, 
    "MarkovPath_ILP": {
        "Recall": [ float(np.mean(R2_trans)) , float(np.std(R2_trans)) ],
        "Precision": [ float(np.mean(P2_trans)), float(np.std(P2_trans)) ],
        "F1": [ float(np.mean(F12_trans)), float(np.std(F12_trans)) ],
        "pairsF1" : [ float(np.mean(pF12_trans)), float(np.std(pF12_trans)) ] ,
        "true_F1" : [ float(np.mean(true_F12_trans)), float(np.std(true_F12_trans)) ], 
        "true_pairF1" : [ float(np.mean(true_pF12_trans)), float(np.std(true_pF12_trans)) ] 
    }
};

json_matric_evaluation = json.dumps(matric_evaluation, indent=4)
with open("result/Markov_matric_evaluation_" + dat_suffix[dat_ix] +".json", "w") as outfile:
    outfile.write(json_matric_evaluation)

print('Rank Markov: Recall (%.3f, %.3f)' % (np.mean(R1_trans), np.std(R1_trans)))
print('Rank Markov: Precision (%.3f, %.3f)' % (np.mean(P1_trans), np.std(P1_trans)))
print('Rank Markov: F1 (%.3f, %.3f)' % (np.mean(F11_trans), np.std(F11_trans)))
print('Rank Markov: pairsF1 (%.3f, %.3f)' %  (np.mean(pF11_trans), np.std(pF11_trans)))
print('Rank Markov: true F1 (%.3f, %.3f)' %  (np.mean(true_F11_trans), np.std(true_F11_trans)))
print('Rank Markov: true pairsF1 (%.3f, %.3f)' %  (np.mean(true_pF11_trans), np.std(true_pF11_trans)))

print('Rank Markov Path: Recall (%.3f, %.3f)' % (np.mean(R2_trans), np.std(R2_trans)))
print('Rank Markov Path: Precision (%.3f, %.3f)' % (np.mean(P2_trans), np.std(P2_trans)))
print('Rank Markov Path: F1 (%.3f, %.3f)' % (np.mean(F12_trans), np.std(F12_trans)))
print('Rank Markov Path: pairsF1 (%.3f, %.3f)' % (np.mean(pF12_trans), np.std(pF12_trans)))
print('Rank Markov Path: true F1 (%.3f, %.3f)' % (np.mean(true_F12_trans), np.std(true_F12_trans)))
print('Rank Markov Path: true pairsF1 (%.3f, %.3f)' % (np.mean(true_pF12_trans), np.std(true_pF12_trans)))

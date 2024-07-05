import os, sys, time, pickle, tempfile
import numpy as np
import pandas as pd 
from joblib import Parallel, delayed

LOG_SMALL = -10
RANKSVM_COST = 10  # RankSVM regularisation constant
N_JOBS = 2         # number of parallel jobs
USE_GUROBI = True # whether to use GUROBI as ILP solver
BIN_CLUSTER = 5  # discritization parameter

result = 'result'
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb']
dat_ix = 1 # Osaka

# Feature
DF_COLUMNS = ['poiID', 'label', 'queryID', 'category', 'neighbourhood', 'popularity', 'nVisit', 'avgDuration', 'trajLen', 'sameCatStart', 'sameCatEnd', 'distStart', 'distEnd', 'diffPopStart', 'diffPopEnd', 'diffNVisitStart', 'diffNVisitEnd', 'diffDurationStart', 'diffDurationEnd', 'sameNeighbourhoodStart', 'sameNeighbourhoodEnd']

def gen_train_subdf(poi_id, query_id_set, poi_info, poi_clusters, cats, clusters, query_id_rdict, POI_DISTMAT):
    assert(isinstance(cats, list))
    assert(isinstance(clusters, list))
    
    columns = DF_COLUMNS
    poi_distmat = POI_DISTMAT
    df_ = pd.DataFrame(index=np.arange(len(query_id_set)), columns=columns)
    
    pop, nvisit = poi_info.loc[poi_id, 'popularity'], poi_info.loc[poi_id, 'nVisit']
    cat, cluster = poi_info.loc[poi_id, 'poiCat'], poi_clusters.loc[poi_id, 'clusterID'] 
    duration = poi_info.loc[poi_id, 'avgDuration']
    
    for j in range(len(query_id_set)):
        qid = query_id_set[j]
        assert(qid in query_id_rdict) # qid --> (start, end, length)
        (p0, pN, trajLen) = query_id_rdict[qid]
        idx = df_.index[j]
        df_.loc[idx, 'poiID'] = poi_id
        df_.loc[idx, 'queryID'] = qid
        df_.at[idx, 'category'] = tuple((cat == np.array(cats)).astype(int) * 2 - 1)
        df_.at[idx, 'neighbourhood'] = tuple((cluster == np.array(clusters)).astype(int) * 2 - 1)
        df_.loc[idx, 'popularity'] = LOG_SMALL if pop < 1 else np.log10(pop)
        df_.loc[idx, 'nVisit'] = LOG_SMALL if nvisit < 1 else np.log10(nvisit)
        df_.loc[idx, 'avgDuration'] = LOG_SMALL if duration < 1 else np.log10(duration)
        df_.loc[idx, 'trajLen'] = trajLen
        df_.loc[idx, 'sameCatStart'] = 1 if cat == poi_info.loc[p0, 'poiCat'] else -1
        df_.loc[idx, 'sameCatEnd']   = 1 if cat == poi_info.loc[pN, 'poiCat'] else -1
        df_.loc[idx, 'distStart'] = poi_distmat.loc[poi_id, p0]
        df_.loc[idx, 'distEnd']   = poi_distmat.loc[poi_id, pN]
        df_.loc[idx, 'diffPopStart'] = pop - poi_info.loc[p0, 'popularity']
        df_.loc[idx, 'diffPopEnd']   = pop - poi_info.loc[pN, 'popularity']
        df_.loc[idx, 'diffNVisitStart'] = nvisit - poi_info.loc[p0, 'nVisit']
        df_.loc[idx, 'diffNVisitEnd']   = nvisit - poi_info.loc[pN, 'nVisit']
        df_.loc[idx, 'diffDurationStart'] = duration - poi_info.loc[p0, 'avgDuration']
        df_.loc[idx, 'diffDurationEnd']   = duration - poi_info.loc[pN, 'avgDuration']
        df_.loc[idx, 'sameNeighbourhoodStart'] = 1 if cluster == poi_clusters.loc[p0, 'clusterID'] else -1
        df_.loc[idx, 'sameNeighbourhoodEnd']   = 1 if cluster == poi_clusters.loc[pN, 'clusterID'] else -1
        
    return df_

def gen_train_df(trajid_list, traj_dict, poi_info, poi_clusters, cats, clusters, n_jobs, poi_distmat, query_id_dict):   
    train_trajs = [traj_dict[x] for x in trajid_list if len(traj_dict[x]) > 2]
    
    qid_set = sorted(set([query_id_dict[(t[0], t[-1], len(t))] for t in train_trajs]))
    poi_set = set()
    for tr in train_trajs:
        poi_set = poi_set | set(tr)
    
    query_id_rdict = dict()
    for k, v in query_id_dict.items(): 
        query_id_rdict[v] = k  # qid --> (start, end, length)
    
    train_df_list = Parallel(n_jobs=n_jobs)\
                            (delayed(gen_train_subdf)(poi, qid_set, poi_info, poi_clusters, cats, clusters, query_id_rdict, poi_distmat) 
                             for poi in poi_set)
                        
    assert(len(train_df_list) > 0)
    df_ = train_df_list[0]
    for j in range(1, len(train_df_list)):
        df_ = df_._append(train_df_list[j], ignore_index=True)            
        
    # set label
    df_.set_index(['queryID', 'poiID'], inplace=True)
    df_['label'] = 0
    for t in train_trajs:
        qid = query_id_dict[(t[0], t[-1], len(t))]
        for poi in t[1:-1]:  # do NOT count if the POI is startPOI/endPOI
            df_.loc[(qid, poi), 'label'] += 1

    df_.reset_index(inplace=True)
    return df_

def gen_test_df(startPOI, endPOI, nPOI, poi_info, poi_clusters, cats, clusters, POI_DISTMAT, QUERY_ID_DICT, poi_all):
    assert(isinstance(cats, list))
    assert(isinstance(clusters, list))
    
    columns = DF_COLUMNS
    poi_distmat = POI_DISTMAT
    query_id_dict = QUERY_ID_DICT
    key = (p0, pN, trajLen) = (startPOI, endPOI, nPOI)
    assert(key in query_id_dict)
    assert(p0 in poi_info.index)
    assert(pN in poi_info.index)
    
    df_ = pd.DataFrame(index=np.arange(poi_info.shape[0]), columns=columns)
    poi_list = sorted(poi_info.index)
    
    qid = query_id_dict[key]
    df_['queryID'] = qid
    df_['label'] = np.random.rand(df_.shape[0]) # label for test data is arbitrary according to libsvm FAQ

    for i in range(df_.index.shape[0]):
        poi = poi_list[i]
        lon, lat = poi_info.loc[poi, 'poiLon'], poi_info.loc[poi, 'poiLat']
        pop, nvisit = poi_info.loc[poi, 'popularity'], poi_info.loc[poi, 'nVisit']
        cat, cluster = poi_info.loc[poi, 'poiCat'], poi_clusters.loc[poi, 'clusterID']
        duration = poi_info.loc[poi, 'avgDuration']
        idx = df_.index[i]
        df_.loc[idx, 'poiID'] = poi
        df_.at [idx, 'category'] = tuple((cat == np.array(cats)).astype(int) * 2 - 1)
        df_.at [idx, 'neighbourhood'] = tuple((cluster == np.array(clusters)).astype(int) * 2 - 1)
        df_.loc[idx, 'popularity'] = LOG_SMALL if pop < 1 else np.log10(pop)
        df_.loc[idx, 'nVisit'] = LOG_SMALL if nvisit < 1 else np.log10(nvisit)
        df_.loc[idx, 'avgDuration'] = LOG_SMALL if duration < 1 else np.log10(duration)
        df_.loc[idx, 'trajLen'] = trajLen
        df_.loc[idx, 'sameCatStart'] = 1 if cat == poi_all.loc[p0, 'poiCat'] else -1
        df_.loc[idx, 'sameCatEnd']   = 1 if cat == poi_all.loc[pN, 'poiCat'] else -1
        df_.loc[idx, 'distStart'] = poi_distmat.loc[poi, p0]
        df_.loc[idx, 'distEnd']   = poi_distmat.loc[poi, pN]
        df_.loc[idx, 'diffPopStart'] = pop - poi_info.loc[p0, 'popularity']
        df_.loc[idx, 'diffPopEnd']   = pop - poi_info.loc[pN, 'popularity']
        df_.loc[idx, 'diffNVisitStart'] = nvisit - poi_info.loc[p0, 'nVisit']
        df_.loc[idx, 'diffNVisitEnd']   = nvisit - poi_info.loc[pN, 'nVisit']
        df_.loc[idx, 'diffDurationStart'] = duration - poi_info.loc[p0, 'avgDuration']
        df_.loc[idx, 'diffDurationEnd']   = duration - poi_info.loc[pN, 'avgDuration']
        df_.loc[idx, 'sameNeighbourhoodStart'] = 1 if cluster == poi_clusters.loc[p0, 'clusterID'] else -1
        df_.loc[idx, 'sameNeighbourhoodEnd']   = 1 if cluster == poi_clusters.loc[pN, 'clusterID'] else -1
        
    return df_

def gen_data_str(df_, df_columns):
    for col in df_columns:
        assert(col in df_.columns)
    lines = []
    for idx in df_.index:
        slist = [str(df_.loc[idx, 'label'])]
        slist.append(' qid:')
        slist.append(str(int(df_.loc[idx, 'queryID'])))
        fid = 1
        for j in range(3, len(df_columns)):
            values_ = df_.at[idx, df_columns[j]]
            values_ = values_ if isinstance(values_, tuple) else [values_]
            for v in values_:
                slist.append(' ')
                slist.append(str(fid)); fid += 1
                slist.append(':')
                slist.append(str(v))
        slist.append('\n')
        lines.append(''.join(slist))
    return ''.join(lines)

def softmax(x):
    x1 = x.copy()
    x1 -= np.max(x1)  # numerically more stable, REF: http://cs231n.github.io/linear-classify/#softmax
    expx = np.exp(x1)
    return expx / np.sum(expx, axis=0) # column-wise sum

def gen_fname(dat_ix):
    assert(0 <= dat_ix < len(dat_suffix))
    suffix = dat_suffix[dat_ix] + '.pkl'
    frank = os.path.join(result, 'poirank-' + suffix)
    ftran = os.path.join(result, 'markov-tran-' + suffix)
    fcomb = os.path.join(result, 'markov-rank-comb-' + suffix)
    frand = os.path.join(result, 'rand-' + suffix)
    return frank, ftran, fcomb, frand

def print_progress(cnt, total):
    """Display a progress bar"""
    assert(cnt > 0 and total > 0 and cnt <= total)
    length = 80
    ratio = cnt / total
    n = int(length * ratio)
    sys.stdout.write('\r[%-80s] %d%% \n' % ('-'*n, int(ratio*100)))
    sys.stdout.flush()

def extract_traj(tid, traj_all):    
    traj = traj_all[traj_all['trajID'] == int(tid)].copy()
    traj.sort_values(by=['startTime'], ascending=True, inplace=True)
    return traj['poiID'].tolist()

def calc_poi_info(trajid_list, traj_all, poi_all):
    assert(len(trajid_list) > 0)
    poi_info = traj_all[ traj_all['trajID'] == int(trajid_list[0])] [['poiID', 'poiDuration']].copy() # trajid_list[0]
    
    for i in range(1, len(trajid_list)):
        traj = traj_all[traj_all['trajID'] == int(trajid_list[i])][['poiID', 'poiDuration']]
        poi_info = poi_info._append(traj, ignore_index=True)
    
    poi_info = poi_info.groupby('poiID').agg(['mean', 'size'])
    poi_info.columns = poi_info.columns.droplevel()
    poi_info.reset_index(inplace=True)
    poi_info.rename(columns={'mean':'avgDuration', 'size':'nVisit'}, inplace=True)
    poi_info.set_index('poiID', inplace=True) 
    poi_info['poiCat'] = poi_all.loc[poi_info.index, 'poiCat']
    poi_info['poiLon'] = poi_all.loc[poi_info.index, 'poiLon']
    poi_info['poiLat'] = poi_all.loc[poi_info.index, 'poiLat']

    # POI popularity: the number of distinct users that visited the POI
    trajid_list = list(map(int, trajid_list)) # change to int value list
    pop_df = traj_all[traj_all['trajID'].isin(trajid_list)][['poiID', 'userID']].copy()
    pop_df = pop_df.groupby('poiID').agg(pd.Series.nunique)
    pop_df.rename(columns={'userID':'nunique'}, inplace=True)
    poi_info['popularity'] = pop_df.loc[poi_info.index, 'nunique']
    
    return poi_info.copy()

def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088 # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius
    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist =  2 * radius * np.arcsin( np.sqrt( (np.sin(0.5*dlat))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5*dlng))**2 ))
    return dist

def calc_F1(traj_act, traj_rec, noloop=False):
    '''Compute recall, precision and F1 for recommended trajectories'''
    assert(isinstance(noloop, bool))
    assert(len(traj_act) > 0)
    assert(len(traj_rec) > 0)
    
    if noloop == True:
        intersize = len(set(traj_act) & set(traj_rec))
    else:
        match_tags = np.zeros(len(traj_act), dtype=bool)
        for poi in traj_rec:
            for j in range(len(traj_act)):
                if match_tags[j] == False and poi == traj_act[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]
        
    recall = intersize / len(traj_act)
    precision = intersize / len(traj_rec)
    F1 = 2 * precision * recall / (precision + recall)
    return recall, precision, F1

def calc_pairsF1(y, y_hat):
    assert(len(y) > 0)
    assert(len(y) == len(set(y))) # no loops in y
    n = len(y)
    nr = len(y_hat)
    n0 = n*(n-1)//2
    n0r = nr*(nr-1)//2
    
    # y determines the correct visiting order
    order_dict = dict()
    for i in range( int(n) ):
        order_dict[y[i]] = i

    nc = 0
    for i in range( int(nr) ):
        poi1 = y_hat[i]
        for j in range(i+1, int(nr)):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]: nc += 1

    precision =0.0
    recall=0.0 
    F1=0.0
    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        F1 = 0
    else:
        F1 = 2. * precision * recall / (precision + recall)
    return F1

def true_F1(expected, predict, noloop=False):
    '''Compute recall, precision and F1 for recommended trajectories'''
    assert (isinstance(noloop, bool))
    assert (len(expected) > 0)
    assert (len(predict) > 0)
    expected = expected[1:-1]
    predict = predict[1:-1]
    predict_size = len(expected)
    if noloop == True:
        intersize = len(set(expected) & set(predict))
    else:
        match_tags = np.zeros(predict_size, dtype=np.bool_)
        for poi in predict:
            for j in range(len(expected)):
                if match_tags[j] == False and poi == expected[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]
    
    recall = intersize * 1.0 / len(expected)
    if ( len(predict)):
        precision = intersize * 1.0 / len(predict)
    else:
        precision = 0

    denominator = recall + precision
    if denominator == 0:
        denominator = 1
    score = 2 * precision * recall * 1.0 / denominator
    return score

def true_pairsF1(y, y_hat):
    #['296142', '3976', '14458', '3976'] ['296142', '3976', '3976', '3976'] 0.75 0.5 0.5 0.0 0.24672975470447223
    assert (len(y) > 2)
    y = y[1:-1]
    y_hat = y_hat[1:-1]
    #assert (len(y) == len(set(y)))  # no loops in y
    # cdef int n, nr, nc, poi1, poi2, i, j
    # cdef double n0, n0r
    n = len(y)
    nr = len(y_hat)

    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2
    # y determines the correct visiting order
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        #print(f'i:{i}, poi1:{poi1}, nr:{nr}')
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            #print(f'j:{j}, poi2:{poi2}, nc:{nc}')
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1
    if n0r == 0:
        n0 = 1
        n0r = 1
        if(len(y_hat)):
            if y[0] == y_hat[0]:
                nc = 1
        
    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0r)
    if nc == 0:
        f1 = 0
    else:
        f1 = 2. * precision * recall / (precision + recall)
    """
    if f1 == 0:
        print(order_dict)
        print(y)
        print(y_hat)
        print()
    """
    return f1

def getquery_id_traj_dict (trajid_set_all, traj_all):

    traj_dict = dict()
    for trajid in trajid_set_all:
        traj = extract_traj(trajid, traj_all)
        assert(trajid not in traj_dict)
        traj_dict[trajid] = traj
    QUERY_ID_DICT = dict()  # (start, end, length) --> qid
    keys = [(traj_dict[x][0], traj_dict[x][-1], len(traj_dict[x])) \
            for x in sorted(traj_dict.keys()) if len(traj_dict[x]) > 2] # Len Trajectoy > 2
    cnt = 0
    for key in keys:
        if key not in QUERY_ID_DICT:   # (start, end, length) --> qid
            QUERY_ID_DICT[key] = cnt
            cnt += 1
    return traj_dict, QUERY_ID_DICT

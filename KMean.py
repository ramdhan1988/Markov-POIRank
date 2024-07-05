import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


BIN_CLUSTER = 5  # discritization parameter

def K_mean (nclusters, X, poi_train):
    kmeans = KMeans(n_clusters=nclusters, random_state=987654321)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    POI_CLUSTER_LIST = sorted(np.unique(clusters))
    POI_CLUSTERS = pd.DataFrame(data=clusters, index=poi_train)
    POI_CLUSTERS.index.name = 'poiID'
    POI_CLUSTERS.rename(columns={0:'clusterID'}, inplace=True)
    POI_CLUSTERS['clusterID'] = POI_CLUSTERS['clusterID'].astype(int)
    return clusters, POI_CLUSTER_LIST, POI_CLUSTERS
import sklearn.cluster as skc
from sklearn.preprocessing import normalize
import numpy as np
import community
import networkx as nx
import logging

class ClusteringMethod:

    def __init__(self,affinity_data, clusatering_type, params=None):

        if not (affinity_data.transpose() == affinity_data).all():
            logging.info('Adjacency Matrix not symmetric, converted to Symmetric')
            affinity_data = (affinity_data + affinity_data.T)/2.

        if params is None:
            params = {}

        if 'Abs_Tol' not in params:
            params['Abs_Tol'] = 1e-12

        if 'n_cluster' not in params:
            params['n_cluster'] = int(affinity_data.shape[0] / 5)

        affinity_data[affinity_data < params['Abs_Tol']] = 0

        self._adjacency = normalize(affinity_data, axis=1, norm='l1')
        clusatering_type = clusatering_type.lower()
        if clusatering_type == 'spectral':
            self._Graph = skc.SpectralClustering(n_clusters=params['n_cluster'],
                                            affinity='precomputed')
        elif clusatering_type == 'louvain':
            self._Graph = nx.from_numpy_matrix(self._adjacency)
        else:
            logging.error('Have not been developed yet. Come back later :)')
        self.method = clusatering_type

    def compute_cluster(self):

        if self.method == 'spectral':
            self._Graph.fit(self._adjacency, self._Graph.n_clusters)
        elif self.method == 'louvain':
            self.partition = community.best_partition(self._Graph)

    def compute_cluster_matrix_skc(self):

        print('to be developed')


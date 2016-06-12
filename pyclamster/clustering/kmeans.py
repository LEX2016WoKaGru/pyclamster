# -*- coding: utf-8 -*-
"""
Created on 10.06.16

Created for pyclamster

    Copyright (C) {2016}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules

# External modules
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.metrics import silhouette_score

# Internal modules
from .labels import Labels

__version__ = ""


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    Class for the KMeans algorithm based on sklearn MiniBatchKMeans algorithm.
    in addition to the KMeans of sklearn this algorithm returns for the
    transform method a Labels instance and this class could also determine the
    best number of clusters with the method bestK.
    """
    def __init__(self, n_cluster=None):
        """
        Args:
            n_cluster (optional[int]): Number of clusters. If it is none then
                the number of clusters will be determined with an automatic
                method within the fit method.
        """
        self.base_algorithm = MiniBatchKMeans
        self.algorithm = None
        self.n_cluster = n_cluster

    @property
    def labels(self):
        assert self.algorithm is not None, "The algorithm isn't trained yet"
        return Labels(self.algorithm.labels_)

    def fit(self, X):
        if self.n_cluster is None:
            self.n_cluster, _ = self.bestK(X)
        self.algorithm = self.base_algorithm(self.n_cluster)
        self.algorithm.fit(X)
        return self

    def predict(self, X):
        return Labels(self.algorithm.predict(X))

    def transform(self, X):
        return self.algorithm.transform(X)

    def bestK(self, X, range_k=(2, 20)):
        """
        Based on the silhouette score at the moment.
        Args:
            X (numpy array):

        Returns:
            n_cluster, scores
        """
        scores = {}
        for k in range(range_k[0], range_k[1]+1):
            clusterer = self.base_algorithm(k).fit(X)
            labels = clusterer.labels_
            scores[k] = silhouette_score(X, labels, sample_size=1000,
                                         random_state=42)
        n_cluster = max(scores, key=scores.get)
        return n_cluster, scores

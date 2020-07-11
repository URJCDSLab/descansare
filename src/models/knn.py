import numpy as np


class Knn:
    def __init__(self, k=3, weights=None, dist_matrix=None):
        self.weights = weights
        self.k = k
        self.dist_matrix = dist_matrix
        self.df = None
        self.neighbours = None
        self.preds = None
        self.range_vars = None

    def fit(self, X, target, ref):

        if self.weights is None:
            self.weights = np.ones(X.shape[1])*(1/X.shape[1])

        max_vars = X.max(axis=0)
        min_vars = X.min(axis=0)
        self.X = X
        self.range_vars = max_vars-min_vars
        self.dist_matrix = self.dist()
        self.knn(target, ref)

    def knn(self, target, ref):
        tot_neighbours = []
        preds = []
        for vec in self.dist_matrix:
            neighbours = vec.argsort()[1:self.k + 1]
            max_ref_neighbour = neighbours[ref[neighbours].argmax()]
            preds.append(target[max_ref_neighbour])
            tot_neighbours.append(neighbours)
        self.neighbours = tot_neighbours
        self.preds = preds

    def dist(self):
        dist = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i, instance in enumerate(self.X):
            for instance_next in self.X[1:]:
                try:
                    dist[i, i + 1] = np.sum((np.abs(instance - instance_next)/self.range_vars) * self.weights)
                    dist[i + 1, i] = dist[i, i + 1]
                except:
                    pass

    def predict(self, X_pred):
        pass

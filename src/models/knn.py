import numpy as np
from itertools import compress


class Knn:
    def __init__(self, k=3, weights=None, dist_matrix=None):
        self.weights = weights
        self.k = k
        self.dist_matrix = dist_matrix
        self.data_train = None
        self.neighbours = None
        self.preds = None
        self.range_vars = None
        self.target = None
        self.ref = None

    def fit(self, data_train, target, ref):

        if self.weights is None:
            self.weights = np.ones(data_train.shape[1]) * (1 / data_train.shape[1])

        max_vars = data_train.max(axis=0)
        min_vars = data_train.min(axis=0)
        self.data_train = data_train
        self.target = target
        self.ref = ref
        self.range_vars = max_vars - min_vars
        self.dist_matrix = self.__dist(self.data_train)
        self.neighbours, self.preds = self.__knn(self.dist_matrix)

    def __knn(self, dist_matrix):

        neighbours = dist_matrix.argsort()[:, 1:self.k + 1]
        max_ref_neighbours = self.ref[neighbours].argmax(axis=1)
        # se tiene que poder hacer mejor
        preds_index = neighbours.reshape(-1, )[
            [i * neighbours.shape[1] for i in range(neighbours.shape[0])] + max_ref_neighbours]
        preds = np.hstack((self.target[preds_index].reshape(-1, 1), self.ref[preds_index].reshape(-1, 1)))
        return neighbours, preds

    def __dist(self, data_points):

        # dist = np.zeros((data_points.shape[0], self.data_train.shape[0]))

        def f(vec):
            return np.sum((np.abs(vec - self.data_train) / self.range_vars) * self.weights, axis=1)


        # vectorizar este for
        # for i, vec in enumerate(data_points):
        #     dists = np.sum((np.abs(vec - self.data_train) / self.range_vars) * self.weights, axis=1)
        dist = np.apply_along_axis(f, 1, data_points)
            # dist[i, :] = dists

        return dist

    def predict(self, data_pred, neighbours_index=False, dist_matrix=False):

        # One-dimensional vector reshape
        if data_pred.ndim == 1:
            data_pred = np.reshape(data_pred, (1, data_pred.shape[0]))

        dist_preds = self.__dist(data_pred)
        neighbours, preds = self.__knn(dist_preds)
        mask = [True, neighbours_index, dist_matrix]
        predicted = [preds, neighbours, dist_preds]
        try:
            predictions = list(*compress(predicted, mask))
        except:
            predictions = list(compress(predicted, mask))
        return predictions

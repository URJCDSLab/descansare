import numpy as np
from itertools import compress


class Knn:
    def __init__(self, k=11, weights=None, dist_matrix=None):
        """K nearest neighbours for pressure settings that maximize the SQI value

        Args:
            k (int): Number of neighbours. Defaults to 11.
            weights (list, optional): Weigths for input variables. Defaults to None.
            dist_matrix (ndarray, optional): Pre defined matrix distance between training samples. Defaults to None.
        """
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
        """Fit the model using data_train as training data and target(SQI) as target values for ref setting pressure.

        Args:
            data_train (ndarray): [description]
            target (1d-array): [description]
            ref (1d-array): [description]
        """
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
        """Internal method that returns neighbours and predictions given a distance matrix

        Args:
            dist_matrix (ndarray): Distance matrix 

        Returns:
            tuple: neighbours, preds for each sample
        """
        neighbours = dist_matrix.argsort()[:, 1:self.k + 1]
        max_ref_neighbours = self.ref[neighbours].argmax(axis=1)
        # se tiene que poder hacer mejor
        preds_index = neighbours.reshape(-1, )[
            [i * neighbours.shape[1] for i in range(neighbours.shape[0])] + max_ref_neighbours]
        preds = np.hstack((self.target[preds_index].reshape(-1, 1), self.ref[preds_index].reshape(-1, 1)))
        return neighbours, preds

    def __dist(self, data_points):
        """Internal method that compute the distance matrix

        Args:
            data_points (ndarray): Samples to calculate the distance between them and the training samples.

        Returns:
                ndarray: dist matrix
        """
        def f(vec):
            """Auxiliary function to vectorize distance calculus

            Args:
                vec (array): Vector of samples
            
            Returns:
                function: funciton to compute the calculus of distance matrix
            """
            return np.sum((np.abs(vec - self.data_train) / self.range_vars) * self.weights, axis=1)

        dist = np.apply_along_axis(f, 1, data_points)

        return dist

    def predict(self, data_pred, neighbours_index=False, dist_matrix=False):
        """	 Predict the class labels for the provided data.

        Args:
            data_pred (ndarray): Samples for prediccion
            neighbours_index (bool, optional): The indexes of the neighbours are provided if neighbours_index=True. Defaults to False.
            dist_matrix (bool, optional): The distance matrix are provided if dist_matrix=True. Defaults to False.

        Returns:
            list: list of predictions
        """
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

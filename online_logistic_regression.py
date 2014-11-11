__author__ = 'mbarnes1'
import numpy as np
from itertools import izip


class OnlineLogisticRegression(object):
    """
    An implementation of multiclass online logistic regression
    """
    def __init__(self, alpha=0.1, number_classes=2, number_features=9):
        """
        :param alpha: the learning rate
        :param number_classes:
        :param number_features:
        """
        self._alpha = alpha
        self.weights = list()
        while len(self.weights) < number_classes:
            self.weights.append(np.random.rand(number_features)*2-1)  # randomly distributed over unit hypersphere
        self.predictions = list()  # predictions made thus far
        self.loss = list()  # loss for each class, at each timestep

    def update(self, features, label):
        """
        Make prediction, get loss, update weights
        :param features: 1D array of number_features features
        :param label: The true label, zero indexed
        """
        (prediction, delta_loss) = self._delta_loss(features, label)
        self.predictions.append(prediction)
        new_weights = list()
        for w, d_loss in izip(self.weights, delta_loss):
            new_weights.append(project(w + self._alpha*d_loss))
        self.weights = new_weights

    def predict(self, x):
        """
        Predicts the class of feature vector x
        :param x:
        :return cls: The class label
        """
        probabilities = self._probability(x)
        return np.argmax(probabilities)

    def _probability(self, x):
        """
        Returns the probability for each class
        :param x: Features vector
        :return probabilities: List of probabilities
        """
        normalizer = 0
        for w in self.weights:
            normalizer += np.exp(np.dot(w, x))
        probabilities = list()
        for w in self.weights:
            probabilities.append(np.exp(np.dot(w, x))/normalizer)
        return probabilities

    def _delta_loss(self, x, true_label):
        """
        The derivative of the loss
        :param x: 1D feature vector
        :param true_label: The true label
        :return prediction: The predicted class
        :return delta_loss: A list of loss vectors for each weight vector
        """
        probabilities = self._probability(x)
        prediction = np.argmax(probabilities)
        indicators = np.zeros(len(probabilities))
        indicators[true_label] = 1
        delta_loss = list()
        for probability, indicator in izip(probabilities, indicators):
            delta_loss.append(-(probability - indicator)*x)
        return prediction, delta_loss


def project(w):
    """
    Projects weight vector to hypersphere w/ radius 1
    :param w: 1D weight vector
    :return w_proj: The projected weight vector, such that the L2 norm <= 1
    """
    if np.linalg.norm(w) > 1:
        w_proj = w/np.linalg.norm(w)
    else:
        w_proj = w
    return w_proj
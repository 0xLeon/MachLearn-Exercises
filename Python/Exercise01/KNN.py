import collections
import enum
import typing

import numpy as np

class DiscreteKNN(object):
    def __init__(self, k: int, classes: enum.Enum):
        self.trainData = np.array([], dtype=np.float)
        self.trainLabels = np.array([])
        self.k = k
        self.classes = classes

    def train(self, trainData, trainLabels):
        self.trainData = trainData
        self.trainLabels = trainLabels

    def predict(self, predicate: typing.List[typing.List[float]]):
        distances = np.linalg.norm(self.trainData - np.array(predicate), axis=1)
        kSmallest = np.argpartition(distances, self.k)[0:self.k]

        return self.classes(collections.Counter(self.trainLabels[kSmallest]).most_common(1)[0][0])

    def computeEmpiricalLoss(self, testLabels, predictedLabels):
        return (1.0 / len(predictedLabels)) * np.sum([int(a != b) for (a, b) in zip(testLabels, predictedLabels)])

import pandas as pd
import numpy as np
from collections import OrderedDict


class Leaf:
    def __init__(self, df):
        """
        Inputs:
        df - a pd.DataFrame
        :param df:
        """

        # a dictionary of counts of target classes in the Leaf's branch
        self.predictions = df.iloc[:, -1].value_counts().to_dict()
        self.predictions = OrderedDict(sorted(self.predictions.items()))
        self.__sum = np.asarray(list(self.predictions.values())).astype(float).sum()

        # normalize the counts to return a dictionary of confidences
        self.predict = {key: str(value / self.__sum) for (key, value) in self.predictions.items()}
        self.predict = OrderedDict(sorted(self.predict.items()))


class Node:
    def __init__(self, col, value, left, right):
        """
        Inputs:
        col - a pd.DataFrame column index
        value - a value in the column
        left - a pd.DataFrame
        right - a pd.DataFrame
        """

        self.threshold = (col, value)
        self.left = left
        self.right = right


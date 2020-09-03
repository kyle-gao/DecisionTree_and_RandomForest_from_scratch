import pandas as pd
import pandas as pd
from helper_functions import *
from Nodes import *


class DecisionTree:

    def __init__(self, df, max_depth=None):
        """df - a pd.DataFrame, last column must be the Target with integer values (as opposed to String)"""
        self.df = df
        self.max_depth = max_depth
        self.tree = None

    def build_tree(self):
        """Builds the train on the DataFrame.
            Call before other methods"""
        self.tree = make_tree(self.df, max_depth=self.max_depth)

    def predict_row(self, row):
        """
        :param row: a pd.DataFrame row
        :return: a dictionary of confidences for each label
        and the index corresponding to the most confident prediction
        """

        confidences_dict = classify(row, self.tree)
        confidences = np.asarray(list(confidences_dict.values()))
        prediction = np.argmax(confidences)
        return confidences_dict, prediction

    def predictions(self, test_df):
        """
        Outputs an array of predictions from the rows of Dataframe test_df
        """

        predictions = []
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            _, prediction = self.predict_row(row)
            predictions.append(prediction)
        return predictions

    def evaluate(self, test):
        """
        Outputs the prediction accuracy of a DataFrame test - test.iloc[:,-1] must be the target column
        """
        length = float(len(test))
        predictions = self.predictions(test)

        return (predictions == test.iloc[:, -1]).sum() / length


class RandomForest():
    def __init__(self, k, m, df, max_depth=None):
        """
        Inputs:
        k - number of features to subsample
        m - number of trees
        df - df - a pd.DataFrame, last column must be the Target with integer values (as opposed to String)
        """
        self.k = k
        self.m = m
        self.df = df
        self.num_features = len(df.columns) - 1
        self.trees = None
        self.max_depth = None

    def train(self, max_depth=None):
        """Populates the random forest, call before other methods
        """
        df_samples = [self.df.sample(frac=1, replace=True) for i in range(self.m)]
        self.trees = [make_tree(df, k=self.k, feature_num=self.num_features, max_depth=self.max_depth) for df in
                      df_samples]

    def predict(self, row):
        """Predicts an integer output of a single row. The prediction is returned as an integer corresponding to the
        index of the OrderedDict. i.e. The classes are sorted, and the prediction returns the index.
        """

        predictions = [classify(row, tree) for tree in self.trees]
        average_pred = np.asarray([list(dic.values()) for dic in predictions]).astype(float)
        average_pred = np.mean(average_pred, axis=0)

        return np.argmax(average_pred)

    def predictions(self, df):
        """
        Outputs an array of predictions from the rows of Dataframe df
        """
        predictions = []
        for i in range(len(df)):
            row = df.iloc[i]
            predictions.append(self.predict(row))
        return predictions

    def evaluate(self, test):
        """
        Outputs the prediction accuracy of a Dataframe test - test.iloc[:,-1] must be the target column
        """
        length = float(len(test))
        predictions = self.predictions(test)

        return (predictions == test.iloc[:, -1]).sum() / length

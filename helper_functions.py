import pandas as pd
import numpy as np
from helper_functions import *
from Nodes import *


def partition(df, column, value):
    """
    :param df:
    :param column:
    :param value:
    :return:
    """
    if df[column].dtype.name in ["category", "object", "bool"]:
        return df.loc[df[column] == value], df.loc[df[column] != value]
    else:
        return df.loc[df[column] >= value], df.loc[df[column] < value]


def gini_impurity(df):
    """
    :param df: a pd.Dataframe
    :return: the gini impurity
    """
    counts = df.iloc[:, -1].value_counts()
    impurity = 1
    for label in counts.index:
        if df.iloc[:, -1].dtype.name == 'category' and isinstance(label, float):
            label = int(label)
            prob_label = counts.iloc[label] / counts.sum()
        else:
            prob_label = counts[label] / counts.sum()
        impurity = impurity - prob_label ** 2
    return impurity


def information_gain(left,right,current):
    """
    :param left: a pd.Dataframe
    :param right:  a pd.Dataframe
    :param current: current gini impurity
    :return: the information gained by the left/right split.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini_impurity(left) - (1 - p) * gini_impurity(right)


def best_split(df):
    """
    Finds the partition with the lowest gini impurity
    :param df: a pd.Dataframe
    :return: best_gain, saved_col, saved_value
    best_gain - the information_gain of the best_split
    saved_col - the feature label of the split
    saved_value - the threshold value of the split
    """
    current = gini_impurity(df)
    best_gain = 0
    saved_col = None
    saved_value = None

    for column in df.columns[:-1]:
        values = df[column]
        for value in values:
            # split the data
            left, right = partition(df, column, value)
            # skip the split if one of the splits is empty
            if len(left) == 0 or len(right) == 0:
                continue
            info_gain = information_gain(left, right, current)
            if info_gain > best_gain:
                best_gain = info_gain
                saved_col = column
                saved_value = value
    return best_gain, saved_col, saved_value


def make_tree(df, depth = 0, max_depth = None, k = None, feature_num = None):
    """
    :param df: a pd.DataFrame
    :param depth: int, recursion depth counter
    :param max_depth: int > 0, max depth
    :param k: int > 0, number of features to subsample when splitting
    :param feature_num: int>0, total number of features
    :return: a recursive tree of Leaf and Node
    """
    if isinstance(k, int) and isinstance(feature_num, int):
        # Randomly choose k feature columns
        columns_randindex = np.random.choice(np.arange(feature_num), k, replace=False)
        columns_randindex = np.sort(columns_randindex)
        columns = df.columns[columns_randindex]
        reduced_df = df[columns]
    else:
        reduced_df = df

    # we calculate the split based on the reduced dataframe
    gain, col, val = best_split(reduced_df)
    if gain == 0:
        return Leaf(df)
    if isinstance(max_depth, int) and depth >= max_depth:
        return Leaf(df)

    left, right = partition(df, col, val)

    # recursive calls
    left_branch = make_tree(left, k = k, feature_num = feature_num, depth = depth + 1, max_depth = max_depth)
    right_branch = make_tree(right, k = k, feature_num = feature_num, depth = depth + 1, max_depth = max_depth)
    return Node(col, val, left_branch, right_branch)


def print_tree(node, df, spacing=""):
    """Recursively prints the tree from df """

    # base case: node is Leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predict)
        return

    (col, val) = node.threshold

    if df[col].dtype.name in ["category", "object", "bool"]:
        print(df[col].dtype.name)
        print(spacing + str(col) + "==" + str(val) + "?")
    else:
        print(spacing + str(col) + ">=" + str(val) + "?")

        # recursive calls
    print(spacing + '--> True:')
    print_tree(node.left, df, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.right, df, spacing + "  ")


def classify(row, node):
    """
    :param row: A pd.DataFrame row, i.e. df.iloc[row]
    :param node: A tree of nodes or a Node
    :return: a dictionary with prediction confidences with keys corresponding to target classes
    """
    if isinstance(node, Leaf):
        return node.predict
    col, val = node.threshold

    if row[col].dtype.name == "float64":
      if row[col]>=val:
        return classify(row, node.left)
      else:
        return classify(row, node.right)
    else:
      if row[col]==val:
        return classify(row, node.left)
      else:
        return classify(row, node.right)


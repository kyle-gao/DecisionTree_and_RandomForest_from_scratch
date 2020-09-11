"""
Copyright 2020 Yi Lin(Kyle) Gao
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."""


import numpy as np
import pandas as pd
from sklearn import datasets
from collections import OrderedDict
from helper_functions import *
from Tree import *


def main():
    """We will test the models using the iris dataset"""
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target[:,np.newaxis]
    columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', "Target"]
    df = pd.DataFrame(np.concatenate((x,y),axis=1), columns =  columns)
    df.Target = df.Target.astype('category')
    #len(df) = 150
    df = df.sample(frac=1)
    df_train = df[:125]
    df_test = df[125:]

    tree = DecisionTree(df_train)
    tree.build_tree()
    print(tree.evaluate(df_test))

    rand_forest = RandomForest(k=3,m=9, df=df_train)
    rand_forest.train()
    print(rand_forest.evaluate(df_test))


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def get_data():
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    return train_df, test_df

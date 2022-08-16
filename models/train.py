import time

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from models.preprocessing import preprocess
from utils import DenseTransformer, get_data


def train(model, X_train, y_train, X_test, y_test, model_name):
    t1 = time.time()

    model = make_pipeline(
        TfidfVectorizer(binary=True, ngram_range=(1, 1)),
        DenseTransformer(),
        model
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MSE = mse(y_test, y_pred)

    t2 = time.time()
    training_time = t2 - t1

    print("--- Model:", model_name, "---")
    print("MSE: ", MSE)
    print("Training time:", training_time)
    return MSE, training_time, model


def choose_model(X_train, y_train, X_test, y_test):
    models = [Ridge(fit_intercept=True), LinearRegression(), GradientBoostingRegressor(n_estimators=10)]
    names = ['RidgeRegression', 'LinearRegression', 'GradientBoostingRegressor']

    training_results = []
    for model, name in zip(models, names):
        MSE, training_time, trained_model = train(
            model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name=name)
        joblib.dump(trained_model, f'../models/trained/{name}.joblib')
        training_results.append({'name': name, 'MSE': MSE, 'training_time': training_time})
    return training_results


if __name__ == '__main__':
    train_df, test_df = get_data()
    train_df['prep_x'] = preprocess(train_df)
    X = train_df['prep_x']
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_results = choose_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

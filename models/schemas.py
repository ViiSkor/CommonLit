from pydantic import BaseModel
from typing import List, Any
from joblib import load


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Regressor(metaclass=SingletonMeta):
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = load('C:/Users/VikaS/PycharmProjects/CommonLit/models/trained/RidgeRegression.joblib')

    def get_model(self):
        return self.model


class CommonLit(BaseModel):
    data: List[str]


class CommonLitPredictionResponse(BaseModel):
    prediction: List[float]

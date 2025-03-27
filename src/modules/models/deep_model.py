from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

class DeepModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        n_inputs=10, 
        couches=[
            (3, "sigmoid"),
            (3, "sigmoid"),
            (1, "sigmoid")
        ]
    ):
        """
        _summary_

        Parameters
        ----------
        couches : List(tuples)
            _description_
            [
                (n_neurones, activation),
                ...
            ]
        """
        self.n_inputs = n_inputs
        self.couches = couches

    def fit(self, X, y):
        modele = Sequential()
        modele.add(Dense(self.couches[0][1], input_dim=self.n_inputs, activation=self.couches[0][2]))
        for couche in self.couches[1:]:
            modele.add(Dense(couche[], activation="relu"))
            
        
        
        return self

    def predict(self, X):
        """ Prédit en utilisant le seuil personnalisé. """
        probas = self.base_model.predict_proba(X)[:, 1]  # Probabilité d'appartenance à la classe positive
        return (probas >= self.threshold).astype(int)  # Applique le seuil
    
    def predict_proba(self, X):
        """ Retourne les probabilités, utile pour certaines métriques. """
        return self.base_model.predict_proba(X)

    def set_params(self, **params):
        """ Permet de modifier les paramètres du modèle et du seuil. """
        for param, value in params.items():
            if param == "threshold":
                self.threshold = value
            else:
                self.base_model.set_params(**{param: value})
        return self

    def get_params(self, deep=True):
        """ Retourne les paramètres pour GridSearch ou RandomizedSearch. """
        params = self.base_model.get_params(deep)
        params["threshold"] = self.threshold
        return params

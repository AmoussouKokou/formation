from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import F1Score, AUC

class DeepModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        # n_inputs=10, 
        couches=[
            (3, "sigmoid"),
            (3, "sigmoid"),
            (2, "softmax")
        ],
        loss='categorical_crossentropy', # mean_squared_error
        optimizer = 'adam',
        epochs = 10,
        batch_size = 1,
        verbose = 0,
        metric='accuracy'
    ):
        """
        Modèle de deep learning compatible avec sklearn.

        Parameters
        ----------
        n_inputs : int
            Nombre de caractéristiques en entrée.
        couches : list of tuples
            Liste des couches sous forme de tuples (n_neurones, activation).
        loss : str
            Fonction de perte.
        optimizer : str
            Optimiseur.
        epochs : int
            Nombre d'époques.
        batch_size : int
            Taille des lots.
        verbose : int
            Niveau de verbosité.
        metric : str
            Métrique à utiliser ('accuracy' ou 'f1').
        """
        self.n_inputs = None
        self.couches = couches
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.metric = metric
        self.modele = None  # Initialisation du modèle
        
        
    def fit(self, X, y):
        self.n_inputs = X.shape[1]
        self.modele = self.__build_modele()
        self.modele.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose = self.verbose)
        
        return self
    
    def __build_modele(self):
        modele = Sequential()
        modele.add(Dense(self.couches[0][0], input_dim=self.n_inputs, activation=self.couches[0][1]))
        for couche in self.couches[1:]:
            modele.add(Dense(couche[0], activation=couche[1]))
        
        modele.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.__metrics())
        
        return modele
        
    def __metrics(self):
        # Choix de la métrique
        if self.metric == 'f1':
            return [F1Score(average=None)]
        elif self.metric == "auc_pr":
            return [AUC(curve="PR")]
        elif self.metric in ["auc_roc", "auc"]:
            return [AUC(curve="ROC")]
        else:
            return ['accuracy']
    
    def predict(self, X, seuil = 0.5):
        """ Prédit les classes en utilisant argmax. """
        probas = self.modele.predict(X)
        if self.couches[-1][0]<=1: # si c'est que la proba d etre 1
            return (probas > seuil).astype(int)
        else: # si c'est la proba de chaque classe qui est retournée
            return np.argmax(probas, axis=1)
    
    def predict_proba(self, X, all_classes = False):
        """Retourne les probabilités de classification. 
        all_classes: obtenir la proba de toutes les classes ? ou seulement de la classe dominante
        """
        if self.couches[-1][0]<=1:
            return self.modele.predict(X)
        elif all_classes:
            self.modele.predict(X)
        else:
            return np.argmax(self.modele.predict(X))

    def set_params(self, **params):
        """ Permet de modifier les hyperparamètres. """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        """ Retourne les paramètres pour GridSearch ou RandomizedSearch. """
        return {
            "n_inputs": self.n_inputs,
            "couches": self.couches,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose
        }

if __name__=="__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    # Génération de données fictives
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 exemples, 10 features
    y = np.random.choice([0, 1], size=(100, 1))  # Classes binaires 0 et 1

    # Encodage one-hot pour la classification softmax
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # Division en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Entraînement du modèle avec le F1-score comme métrique
    model = DeepModel(n_inputs=10, metric='f1', epochs=20, batch_size=5, verbose=1)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul du F1-score avec sklearn
    y_test_labels = np.argmax(y_test, axis=1)  # Convertir one-hot en labels
    f1 = f1_score(y_test_labels, y_pred, average='macro')

    print(f"F1-score sur l'ensemble de test : {f1:.4f}")

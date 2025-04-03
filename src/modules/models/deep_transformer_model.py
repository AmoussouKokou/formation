from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, LayerNormalization, Dropout, Flatten, Add
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import AUC

class DeepTransformerModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        n_heads=4,  # Nombre de têtes d'attention
        key_dim=64,  # Dimension de chaque clé
        ff_dim=128,  # Dimension du Feed Forward
        num_blocks=2,  # Nombre de blocs Transformer
        dropout_rate=0.1,
        loss='binary_crossentropy',  # Classification binaire
        optimizer='adam',
        epochs=10,
        batch_size=1,
        verbose=0,
        metric='accuracy'
    ):
        self.n_inputs = None
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
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
        self.modele.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self
    
    def __build_modele(self):
        inputs = Input(shape=(self.n_inputs, 1))  # On ajoute une dimension pour la compatibilité avec l'attention

        x = inputs
        for _ in range(self.num_blocks):
            x = self.__transformer_block(x)
        
        x = Flatten()(x)  # Transformation pour la sortie dense finale
        outputs = Dense(1, activation="sigmoid")(x)  # Sigmoid pour classification binaire

        modele = Model(inputs, outputs)
        modele.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.__metrics())

        return modele

    def __transformer_block(self, inputs):
        """ Bloc Transformer Encoder """
        attention_output = MultiHeadAttention(num_heads=self.n_heads, key_dim=self.key_dim)(inputs, inputs)
        attention_output = Dropout(self.dropout_rate)(attention_output)
        attention_output = Add()([inputs, attention_output])  # Skip connection
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

        # Feed Forward Network
        ff_output = Dense(self.ff_dim, activation="relu")(attention_output)
        ff_output = Dense(inputs.shape[-1])(ff_output)  # Projection
        ff_output = Dropout(self.dropout_rate)(ff_output)
        ff_output = Add()([attention_output, ff_output])  # Skip connection
        return LayerNormalization(epsilon=1e-6)(ff_output)

    def __metrics(self):
        # Choix de la métrique
        if self.metric == "auc_pr":
            return [AUC(curve="PR")]
        elif self.metric in ["auc_roc", "auc"]:
            return [AUC(curve="ROC")]
        else:
            return ['accuracy']
    
    def predict(self, X, seuil=0.5):
        """ Prédit les classes en utilisant un seuil. """
        probas = self.modele.predict(X)
        return (probas >= seuil).astype(int).flatten()
    
    def predict_proba(self, X):
        """Retourne les probabilités de classification. """
        return self.modele.predict(X)

    def set_params(self, **params):
        """ Permet de modifier les hyperparamètres. """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        """ Retourne les paramètres pour GridSearch ou RandomizedSearch. """
        return {
            "n_heads": self.n_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "num_blocks": self.num_blocks,
            "dropout_rate": self.dropout_rate,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose
        }

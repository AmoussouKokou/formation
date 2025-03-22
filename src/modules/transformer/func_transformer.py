import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, OneToOneFeatureMixin

class FuncTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Transformateur compatible scikit-learn qui applique une fonction élément par élément 
    aux colonnes spécifiées d'un DataFrame ou d'un tableau NumPy.
    """

    def __init__(self, func=lambda x: x, inverse_func=None):
        """
        Initialise le transformateur.

        Paramètres :
        ------------
        - func : fonction à appliquer aux valeurs des colonnes spécifiées.
        - inverse_func : fonction inverse pour `inverse_transform` (optionnelle).
        """
        self.func = func
        self.inverse_func = inverse_func

    def fit(self, X, y=None):
        """
        Scikit-learn nécessite `fit`, mais aucune opération n'est requise ici.
        """
        return self

    def transform(self, X):
        """
        Applique la transformation aux données.

        - X : `DataFrame` ou `ndarray`.

        Retourne :
        ----------
        - X_transformed : données transformées.
        """
        # print(X)
        if isinstance(X, pd.DataFrame):
            return X.map(self.func)
        elif isinstance(X, np.ndarray):
            return np.vectorize(self.func)(X)
        else:
            raise TypeError("X doit être un DataFrame ou un ndarray.")

    def inverse_transform(self, X):
        """
        Applique la transformation inverse si `inverse_func` est défini.

        - X : `DataFrame` ou `ndarray`.

        Retourne :
        ----------
        - X_inverse : données restaurées.
        """
        if self.inverse_func is None:
            raise NotImplementedError("Aucune fonction inverse définie.")

        if isinstance(X, pd.DataFrame):
            return X.applymap(self.inverse_func)
        elif isinstance(X, np.ndarray):
            return np.vectorize(self.inverse_func)(X)
        else:
            raise TypeError("X doit être un DataFrame ou un ndarray.")

    def get_feature_names_out(self, input_features=None):
        """Renvoie les noms des colonnes transformées, en conservant celles d'entrée."""
        if input_features is None:
            raise ValueError("Les noms des colonnes d'entrée doivent être fournis.")
        return np.array(input_features, dtype=object)


if __name__=="__main__":
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    # Exemple de dataset
    data = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'salary': [3000, 4000, 5000, 6000],
        'category': ['A', 'B', 'A', 'B']
    })

    # Colonnes numériques et catégorielles
    num_features = ['age', 'salary']
    cat_features = ['category']

    # Transformateur d'identité pour les catégories (ne fait rien)
    IdEncoder = FuncTransformer()

    # Pipeline de transformation des colonnes
    scaler = ColumnTransformer([
        ("num", StandardScaler(), num_features),  # Standardisation des nombres
        ("cat", IdEncoder, cat_features)          # Encodage d'identité pour les catégories
    ])

    # Transformation des données
    data_transformed = scaler.fit_transform(data)

    print(data_transformed)

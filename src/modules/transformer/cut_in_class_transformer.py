import numpy as np
import pandas as pd
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator

class CutInClassTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, bins=4):
        if isinstance(bins, (int, float)):
            self.n_quantiles = bins
            self.bins = bins
        elif isinstance(bins, (list, np.ndarray)):
            self.n_quantiles = None
            self.bins = bins # np.array(bins)
        else:
            raise ValueError("bins doit être un entier (nombre de classes) ou une liste de coupures.")
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un DataFrame Pandas.")
        
        self.quantile_bins = {}
        for col in X.columns:
            if not isinstance(self.bins, (int, float)):# self.bins is not None:
                self.quantile_bins[col] = np.array(self.bins)
            else:
                self.quantile_bins[col] = np.quantile(X[col], q=np.linspace(0, 1, self.n_quantiles + 1))
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col in X.columns:
            bins = self.quantile_bins[col]
            digitized = np.digitize(X[col], bins, right=False) - 1  # Ajuster les indices
            digitized = np.clip(digitized, 0, len(bins) - 2)  # Éviter les indices hors limite
            categories = [f"[{bins[i]}, {bins[i+1]}[" for i in range(len(bins) - 1)]
            X_transformed[col] = pd.Categorical.from_codes(digitized, categories, ordered=True)
        
        return X_transformed
    
    def inverse_transform(self, X):
        raise NotImplementedError("L'inverse transformation n'est pas définie pour une variable catégorielle.")
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("Les noms des colonnes d'entrée doivent être fournis.")
        return np.array(input_features, dtype=object)

if __name__ == "__main__":
    df = pd.DataFrame({'A': np.random.randn(100) * 10 + 50})
    print(df)
    transformer = CutInClassTransformer(bins=[30, 42, 47, 53, 57, 75])
    transformer.fit(df)
    df_transformed = transformer.transform(df)
    print("Données transformées :\n", df_transformed)
    print(df_transformed.A.unique())
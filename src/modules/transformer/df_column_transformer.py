import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

class DfColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer
        self.feature_names = None

    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        self.feature_names = self._clean_feature_names(self.column_transformer.get_feature_names_out())
        return self

    def transform(self, X):
        X_transformed = self.column_transformer.transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def _clean_feature_names(self, feature_names):
        """Retire le préfixe 'step__' ajouté par ColumnTransformer."""
        return [name.split("__")[-1] for name in feature_names]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    roc_curve, auc, classification_report, precision_recall_curve
)
from sklearn.utils.validation import check_is_fitted

class Evaluator(BaseEstimator, TransformerMixin):
    def __init__(self, modele, scoring="f1-score"):
        """
        Classe pour évaluer un modele scikit-learn.

        Parameters:
        -----------
        modele : sklearn.modele.modele
            modele contenant le modèle à évaluer.
        scoring : str, default="accuracy"
            Métrique utilisée pour le scoring en cross-validation.
        """
        self.modele = modele
        self.scoring = scoring

    def fit(self, X=None, y=None):
        """Entraîne le modele.
        #todo: probleme avec l'etat de fit du modele
        """
        # try:
        #     check_is_fitted(self.modele)
        # except:
        #     self.modele.fit(X, y)
        return self  # Respecte la convention sklearn

    def transform(self, X=None):
        """Applique la transformation du modele."""
        return X # self.modele.transform(X)

    def predict(self, X, y_true):
        """
        Prédit les valeurs et retourne les métriques sous forme de dictionnaire.

        Parameters:
        -----------
        X : array-like
            Données à prédire.
        y_true : array-like
            Vraies valeurs.

        Returns:
        --------
        dict
            Dictionnaire contenant accuracy, precision, recall, f1-score et classification report.
        """
        y_pred = self.modele.predict(X)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary"),
            "recall": recall_score(y_true, y_pred, average="binary"),
            "f1_score": f1_score(y_true, y_pred, average="binary"),
            "classification_report": classification_report(y_true, y_pred, output_dict=True)
        }

        # Affichage des résultats
        print(f"🔹 Accuracy  : {metrics['accuracy']:.4f}")
        print(f"🔹 Precision : {metrics['precision']:.4f}")
        print(f"🔹 Recall    : {metrics['recall']:.4f}")
        print(f"🔹 F1-score  : {metrics['f1_score']:.4f}\n")

        # Matrice de confusion
        self.plot_confusion_matrix(y_true, y_pred)

        return metrics

    # def score(self, X, y):
    #     """
    #     Retourne un score basé sur la métrique choisie (utilisé en cross-validation).
    #     #todo : A revoir
    #     Parameters:
    #     -----------
    #     X : array-like
    #         Données.
    #     y : array-like
    #         Vraies étiquettes.

    #     Returns:
    #     --------
    #     float
    #         Score selon la métrique choisie.
    #     """
    #     y_pred = self.modele.predict(X)

    #     metrics = {
    #         "accuracy": accuracy_score(y, y_pred),
    #         "precision": precision_score(y, y_pred, average="binary"),
    #         "recall": recall_score(y, y_pred, average="binary"),
    #         "f1_score": f1_score(y, y_pred, average="binary")
    #     }

    #     return metrics.get(self.scoring, accuracy_score(y, y_pred))  # Par défaut : accuracy

    def plot_confusion_matrix(self, y_true, y_pred):
        """Affiche la matrice de confusion."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"])
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies classes")
        plt.title("Matrice de confusion")
        plt.show()

    def plot_roc_curve(self, X, y_true):
        """Trace la courbe ROC et affiche l'AUC."""
        y_proba = self.modele.predict_proba(X)[:, 1]  # Probabilité de la classe positive
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        sns.set(style="whitegrid", palette="muted")
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="dodgerblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
        plt.xlabel("Taux de faux positifs (FPR)", fontsize=12)
        plt.ylabel("Taux de vrais positifs (TPR)", fontsize=12)
        plt.title("Courbe ROC", fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()
        return roc_auc
    
    def plot_precision_recall_curve(self, X_test, y_test):
        """
        Calcule et affiche la courbe Precision-Recall avec Seaborn.

        Parameters:
        -----------
        X_test : array-like
            Données de test.
        y_test : array-like
            Labels réels de test.
        """
        # Prédictions des probabilités pour la classe positive
        y_scores = self.modele.predict_proba(X_test)[:, 1]  # Pour un problème binaire

        # Calcul de la courbe PR
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)

        # Création du DataFrame pour Seaborn
        import pandas as pd
        pr_df = pd.DataFrame({'Recall': recall, 'Precision': precision})

        # Affichage avec Seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=pr_df, x="Recall", y="Precision", label=f'PR Curve (AUC={pr_auc:.2f})')

        # Personnalisation du graphique
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        return pr_auc

        # plt.figure(figsize=(6, 5))
        # plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        # plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        # plt.xlabel("Taux de faux positifs (FPR)")
        # plt.ylabel("Taux de vrais positifs (TPR)")
        # plt.title("Courbe ROC")
        # plt.legend(loc="lower right")
        # plt.show()
        
        
if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # ====== 1. Génération des données ======
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Division en train (70%), validation (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # ====== 2. Création du pipeline ======
    # pipeline = Pipeline([
    #     ("scaler", StandardScaler()),  # Normalisation
    #     ("clf", LogisticRegression())  # Modèle
    # ])
    modele = LogisticRegression()
    modele.fit(X_train, y_train)

    # ====== 3. Évaluation avec PipelineEvaluator ======
    evaluator = Evaluator(modele, scoring="f1_score")

    # Entraînement sur train
    # evaluator.fit(X_train, y_train)

    # Prédictions et indicateurs sur validation
    metrics = evaluator.predict(X_val, y_val)

    # Courbe ROC sur test
    evaluator.plot_roc_curve(X_test, y_test)
    
    # Courbe PR
    evaluator.plot_precision_recall_curve(X_test, y_test)

    # ====== 4. Cross-validation ======
    # cv_scores = cross_val_score(evaluator, X_train, y_train, cv=5, scoring="accuracy")

    # print(f"\n🔹 Scores de cross-validation : {cv_scores}")
    # print(f"🔹 Score moyen : {cv_scores.mean():.4f}")

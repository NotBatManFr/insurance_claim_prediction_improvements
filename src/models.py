from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .interfaces import IModel

class SklearnModelAdapter(IModel):
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred, model_name: str):
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        
        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred),
            "F1_Score": f1_score(y_true, y_pred),
            "ROC_AUC": roc_auc_score(y_true, y_pred),
            "FNR": FN / (FN + TP) if (FN + TP) > 0 else 0
        }

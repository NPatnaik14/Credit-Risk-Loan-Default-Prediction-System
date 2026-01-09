from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_models(trained_models, X_test, y_test):
    """Evaluates all trained models and returns metrics."""
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Model': name,
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        results.append(metrics)
        
        logger.info(f"--- {name} Evaluation ---")
        logger.info(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
    return pd.DataFrame(results)

def get_confusion_matrix(model, X_test, y_test):
    """Returns confusion matrix for a specific model."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)

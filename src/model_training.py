from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train):
        """Trains all defined models."""
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            logger.info(f"{name} training complete.")
        return self.trained_models

    def save_models(self, preprocessor):
        """Saves trained models and the preprocessor."""
        for name, model in self.trained_models.items():
            path = os.path.join(self.models_dir, f"{name}.joblib")
            joblib.dump(model, path)
            logger.info(f"Saved {name} to {path}")
        
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")

    def load_best_model(self, name='xgboost'):
        """Loads a specific trained model."""
        path = os.path.join(self.models_dir, f"{name}.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise FileNotFoundError(f"Model {name} not found at {path}")

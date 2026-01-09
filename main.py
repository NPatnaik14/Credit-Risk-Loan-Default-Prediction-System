import os
import logging
from src.data_loader import load_data, split_data
from src.preprocessor import CreditRiskPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import evaluate_models
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    # 1. Load Data
    data_path = os.path.join('data', 'credit_risk_dataset.csv')
    df = load_data(data_path)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 3. Preprocess Data
    preprocessor = CreditRiskPreprocessor()
    X_train_processed, y_train_resampled = preprocessor.fit_transform(X_train, y_train, use_smote=True)
    X_test_processed = preprocessor.transform(X_test)
    
    # 4. Train Models
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X_train_processed, y_train_resampled)
    
    # 5. Evaluate Models
    metrics_df = evaluate_models(trained_models, X_test_processed, y_test)
    print("\nModel Metrics Comparison:")
    print(metrics_df)
    
    # 6. Save Best Model and Preprocessor
    # We'll save all but consider XGBoost as the "production" one for the app
    trainer.save_models(preprocessor)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)

class CreditRiskPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def fit_transform(self, X, y=None, use_smote=True):
        """Fits and transforms the data. Applies SMOTE to training data if enabled."""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features_transformed = cat_encoder.get_feature_names_out(categorical_features).tolist()
        self.feature_names = numeric_features + cat_features_transformed

        if use_smote and y is not None:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_transformed, y)
            return X_resampled, y_resampled
        
        return X_transformed

    def transform(self, X):
        """Transforms data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X)

    def get_feature_names(self):
        return self.feature_names

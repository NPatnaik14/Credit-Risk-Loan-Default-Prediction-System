import shap
import matplotlib.pyplot as plt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        # Use TreeExplainer for XGBoost/RF, Kernel or Linear for others if needed
        # For simplicity in this engine, we'll focus on Tree models but handle appropriately
        if hasattr(model, "predict_proba"):
            self.explainer = shap.Explainer(model)
        else:
            self.explainer = shap.Explainer(model)

    def get_global_explanation(self, X_sample):
        """Generates SHAP summary plot."""
        shap_values = self.explainer(X_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        return plt.gcf()

    def get_local_explanation(self, instance):
        """Generates SHAP force/waterfall plot for a single instance."""
        # Ensure instance is 2D
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer(instance)
        plt.figure(figsize=(10, 4))
        # Waterfall plot is great for single instance
        shap.plots.waterfall(shap_values[0], show=False)
        return plt.gcf()

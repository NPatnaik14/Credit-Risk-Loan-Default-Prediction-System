# Credit Risk Prediction Engine

A production-grade Loan Default Prediction system using XGBoost, SMOTE, and SHAP.

## Project Structure
- `data/`: Contains the raw loan dataset.
- `src/`: Core Python modules for the ML pipeline.
    - `data_loader.py`: Ingestion logic.
    - `preprocessor.py`: Cleaning, Encoding, Scaling, and SMOTE.
    - `model_training.py`: Training Logistic Regression, RF, and XGBoost.
    - `evaluation.py`: Performance metrics and assessment.
    - `explanation.py`: Model interpretability using SHAP.
- `models/`: Saved joblib files for the model and preprocessor.
- `main.py`: Full end-to-end pipeline orchestrator.
- `app.py`: Streamlit dashboard for real-time predictions.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Models:**
   ```bash
   python main.py
   ```
   This will load the data, train the models, and save them to the `models/` directory.

3. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## Key Features
- **Class Imbalance Handling:** Uses SMOTE to handle the high default-to-non-default ratio.
- **Explainability:** Global and local explanations via SHAP.
- **Risk Scoring:** Outputs a 0-100% risk score for each applicant.
- **Production Standards:** Modular code, logging, and clear separation of concerns.

## Dataset
The project uses the Credit Risk Dataset containing customer demographic and loan information.
Target variable: `loan_status` (0: Non-default, 1: Default).

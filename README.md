# Mini-Project--Bankruptcy-Prediction-on-Corporate-Financial-Indicators


Corporate Bankruptcy Prediction using Financial Indicators​
End‑to‑end ML model training and evaluation to predict BankruptcyFlag from 95 company financial ratios using Logistic Regression, Random Forest, and XGBoost. Preprocessing is executed via explicit steps in the notebook rather than a single wrapped Pipeline object.​

Repository structure
notebooks/Corporate-Bankruptcy-Prediction-Using-Financial-Indicators.ipynb — Main notebook with data loading, preprocessing steps, model training, and evaluation.​

data/Bankruptcy.csv — Primary dataset (place here or update the path in the notebook).​

README.md — This file.​

Dataset
File: Bankruptcy.csv with 6,819 rows and 96 columns; target is BankruptcyFlag (1 = bankrupt, 0 = solvent). All predictors are numeric ratios.​

Environment and installation
Python 3.9+ recommended.​

Create and activate a virtual environment, then install dependencies:

python -m venv .venv

source .venv/bin/activate (Windows: .venv\Scripts\activate)

pip install -r requirements.txt​

Example requirements.txt

pandas>=2.0.0

numpy>=1.24.0

scikit-learn>=1.3.0

seaborn>=0.12.0

matplotlib>=3.7.0

xgboost>=2.0.0

imbalanced-learn>=0.11.0 (optional for SMOTE)​

How to run
Place Bankruptcy.csv under data/ or update the path variable in the first loading cell.​

Open notebooks/Corporate-Bankruptcy-Prediction-Using-Financial-Indicators.ipynb in Jupyter or VS Code.​

Run all cells sequentially:

Setup and imports.​

Data loading and preview (confirms shape 6819 × 96).​

Preprocessing steps executed in separate cells:

Missing‑value checks and handling (documented inline).

Scaling where required (e.g., for Logistic Regression).

Optional outlier treatment notes.

Optional imbalance handling (class_weight or SMOTE).​

Model training: Logistic Regression, Random Forest, XGBoost.​

Evaluation: metrics table with Accuracy, ROC‑AUC, F1; optional confusion matrix, ROC, and PR curves if cells are executed.​

Reproducibility
Use a stratified train/test split and fixed random_state values as indicated in the notebook cells.​

Because this is a model‑centric setup (not a single packaged Pipeline), keep the cell order and code blocks unchanged to avoid leakage and ensure consistent results.​

Results (from the notebook)
Best performer: XGBoost with Accuracy ≈ 0.962, ROC‑AUC ≈ 0.946, F1 ≈ 0.454; Random Forest is close; Logistic Regression serves as a baseline. Values may vary slightly due to seeds and splits.​

Notes on “model vs pipeline”
This project demonstrates an ML model workflow with explicit preprocessing and training steps. If a single deployable object is needed later, wrap preprocessing and the estimator in a scikit‑learn/imblearn Pipeline and re‑fit; optional code comments in the notebook indicate where this can be added.

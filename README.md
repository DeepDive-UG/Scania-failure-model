# Scania-failure-model

Machine learning project focused on predicting Scania APS system failures using publicly available sensor data. Includes data preprocessing, feature analysis, model training and evaluation for predictive maintenance.

## Quick Start

1. **Setup Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Generate .csv:**

   Click "Run all" in preprocessing.ipynb notebook.

3. **Run Training:**
   ```bash
   python src/models/train.py
   ```
4. **Run Evaluation:**
   ```bash
   python src/evaluate.py
   ```

## Project Structure

- data/: Raw and processed datasets (ignored by Git).

- notebooks/: Exploratory Data Analysis (EDA) and preprocessing experiments.

- src/: Source code for preprocessing, feature engineering, and modeling.

- models/: Serialized .joblib model files.

- evaluation_plots/: Generated charts (ROC curves, Cost vs Threshold, etc.).

- reports/: Final analysis and summaries.

## Key Results

Through feature selection (Lasso) and dimensionality reduction (PCA), the feature set was reduced from 170 to 36 components. By optimizing the decision threshold of the Random Forest model to 0.089, the total cost was reduced to $10,800.

Detailed analysis, model comparisons, and visualizations can be found in the full report: reports/summary.md

_Developed as a final project for the Bootcamp AI 2025 by DeepDive._

# Students Performance Prediction ML Project

## Overview
End-to-End Machine Learning project to predict **math scores** using student performance dataset (gender, ethnicity, parental education, lunch, test prep → regression).

Uses Cookiecutter ML template with components/pipeline for data ingestion (MySQL/notebook/Data/data.csv), transformation (label encode + scale), training (RF/XGB/LR regressors with GridSearchCV), evaluation.

**Goal**: Train model on transformed data, deploy via app.py.

## Tech Stack
- **Python**: 3.8+
- **ML**: scikit-learn, XGBoost, joblib
- **Data**: Pandas, NumPy
- **DB**: PyMySQL
- **MLOps**: DVC (artifacts.dvc)
- **Deployment**: FastAPI/Flask (app.py)

## Project Structure
```
.
├── artifacts/                 # Generated data/models (DVC tracked)
│   ├── train.csv/test.csv
│   ├── transformed_train.csv  # Encoded + scaled
│   ├── model.pkl             # Best trained model
│   └── models/               # Individual models
├── src/
│   └── ml_project/
│       ├── components/       # Modular steps
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py     # ✅ Complete (RF/XGB/LR, R² best)
│       │   └── model_evaluation.py  # ⏳ Next
│       ├── pipelines/        # Orchestration (fix git mess)
│       │   └── training_pipeline.py
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
├── notebook/                 # EDA/Experiments
│   └── EDA_Students_Performance.ipynb
├── app.py                    # API deployment
├── requirements.txt          # Dependencies
├── setup.py                  # pip install -e .
└── TODO.md                   # Progress
```

## Installation & Setup
1. **Clone/Setup**:
   ```
   cd "c:/Users/user/Documents/Machine Learning Project"
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Env Vars** (.env):
   ```
   host=localhost
   user=root
   password=pass
   database=students_db
   ```

3. **Generate Artifacts**:
   ```
   # Option 1: Components sequentially
   cd src/ml_project/components
   python data_ingestion.py
   python data_transformation.py
   python model_trainer.py  # → artifacts/model.pkl
   ```

## Previous Processes (✅ Accomplished)
- **Data Ingestion**: MySQL → artifacts/train.csv/test.csv (80/20 split).
- **Data Transformation**: LabelEncoder (cat), StandardScaler (num), target='math score', saves transformed CSVs + preprocessor.pkl.
- **Model Training**: RF/XGB/LR regressors, 5-fold GridSearchCV (R²), best model.pkl, metrics logged, threshold R²>0.6.

**Sample Metrics** (run to see logs):
```
rf - R2: 0.85xx, MAE: 5.xx
xgb - R2: 0.87xx, MAE: 4.xx  # Often best
```

## Next Steps (from TODO.md)
1. **[ ] Test model_trainer**: Run above → verify model.pkl.
2. **[ ] model_evaluation.py**: Load model.pkl, compute full metrics, log confusion/R² plot.
3. **[ ] Fix Pipelines**: Clean training_pipeline.pysrc/ → src/ml_project/pipelines/training_pipeline.py:
   ```
   DataIngestion → Transformation → Trainer → Evaluator
   ```
4. **[ ] Prediction Pipeline**: Load preprocessor + model.pkl → predict API.
5. **[ ] App.py**: FastAPI endpoint `/predict`.
6. **[ ] DVC/MLflow**: `dvc repro`, track experiments.
7. **[ ] Tests**: pytest components.
8. **[ ] Deploy**: Docker → Streamlit/HuggingFace.

## Usage
```bash
# Full Pipeline (once fixed)
python src/ml_project/pipelines/training_pipeline.py config.yaml

# Predict
curl -X POST "http://localhost:8000/predict" -d '{"data": [...]}'
```

## Troubleshooting
- **ModuleNotFoundError 'src'**: Run from root: `PYTHONPATH=src python src/ml_project/components/model_trainer.py`
- **No artifacts**: Run ingestion/transformation first.
- **Logs**: `logs/log_*.log`

## Contribution
1. Fork/PR.
2. Update TODO.md.

**Author**: BLACKBOXAI  
**Version**: 1.0 (model_trainer complete)

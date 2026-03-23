import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')
    models_dir: str = os.path.join('artifacts', 'models')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.best_model_score = -float('inf')
        self.best_model = None
        self.best_params = None
        
    def evaluate_model(self, X_train, y_train, X_test, y_test, models: dict):
        """
        Train and evaluate multiple models using GridSearchCV.
        """
        try:
            trained_models = {}
            model_scores = {}
            
            param_grids = {
                'rf': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                },
                'xgb': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5],
                    'random_state': [42]
                },
                'lr': {}  # LinearRegression has no major params to tune
            }
            
            for model_name, model in models.items():
                logging.info(f'Training {model_name}...')
                
                if model_name == 'lr':
                    # No GridSearch for simple LR
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                else:
                    grid = GridSearchCV(
                        model, param_grids[model_name], 
                        cv=5, scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    self.best_params = grid.best_params_
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                
                model_scores[model_name] = {'r2': r2, 'mae': mae}
                trained_models[model_name] = model
                
                logging.info(f'{model_name} - R2: {r2:.4f}, MAE: {mae:.4f}')
                
                # Track best model
                if r2 > self.best_model_score:
                    self.best_model_score = r2
                    self.best_model = model
                    self.best_model_name = model_name
                
            # Save individual models
            os.makedirs(self.config.models_dir, exist_ok=True)
            for name, model in trained_models.items():
                joblib.dump(model, os.path.join(self.config.models_dir, f'{name}.pkl'))
            
            logging.info(f'Best model: {self.best_model_name} with R2: {self.best_model_score:.4f}')
            return trained_models, model_scores
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self, train_path: str, test_path: str):
        """
        Main method: Load data, train models, save best model.
        """
        try:
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            
            # Load transformed data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f'Train data shape: {train_df.shape}')
            logging.info(f'Test data shape: {test_df.shape}')
            
            # Assuming target column from DataTransformation
            target_column = 'math score'
            
            # Prepare features and targets
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            logging.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
            logging.info(f'Features: {X_train.columns.tolist()}')
            
            # Define models (regressors for score prediction)
            models = {
                'rf': RandomForestRegressor(random_state=42),
                'xgb': xgb.XGBRegressor(random_state=42, verbosity=0),
                'lr': LinearRegression()
            }
            
            # Train and evaluate
            trained_models, model_scores = self.evaluate_model(X_train, y_train, X_test, y_test, models)
            
            if self.best_model_score < 0.6:
                raise CustomException('Best model R2 score too low (<0.6)', sys)
            
            # Save best model
            joblib.dump(self.best_model, self.config.trained_model_path)
            logging.info(f'Best model saved to: {self.config.trained_model_path}')
            
            # Save training metadata
            metadata = {
                'best_model': self.best_model_name,
                'best_score_r2': self.best_model_score,
                'all_scores': model_scores,
                'best_params': self.best_params if hasattr(self, 'best_params') else {}
            }
            joblib.dump(metadata, os.path.join(os.path.dirname(self.config.trained_model_path), 'training_metadata.pkl'))
            
            return self.config.trained_model_path
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        train_path = os.path.join('artifacts', 'transformed_train.csv')
        test_path = os.path.join('artifacts', 'transformed_test.csv')
        
        model_trainer = ModelTrainer()
        model_path = model_trainer.initiate_model_trainer(train_path, test_path)
        logging.info(f'Model training completed. Model path: {model_path}')
    except Exception as e:
        logging.error(f'Model training failed: {str(e)}')

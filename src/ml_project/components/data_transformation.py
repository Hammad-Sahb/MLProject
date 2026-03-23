import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import pickle
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging

@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join('artifacts', 'transformed_train.csv')
    transformed_test_path: str = os.path.join('artifacts', 'transformed_test.csv')
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    target_column: str = 'math score'

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.label_encoders = {}
        
    def get_data_types(self, dataframe: pd.DataFrame) -> tuple:
        """
        Separate categorical and numerical columns.
        """
        categorical_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = dataframe.select_dtypes(exclude=['object']).columns.tolist()
        
        # Remove target from numerical
        numerical_columns = [col for col in numerical_columns if col != self.transformation_config.target_column]
        
        logging.info(f'Categorical columns: {categorical_columns}')
        logging.info(f'Numerical columns: {numerical_columns}')
        
        return categorical_columns, numerical_columns
    
    def separate_target_feature(self, data: pd.DataFrame) -> tuple:
        """
        Separate features (X) and target (y).
        """
        try:
            X = data.drop(columns=[self.transformation_config.target_column], axis=1)
            y = data[self.transformation_config.target_column]
            logging.info(f'Separated X shape: {X.shape}, y shape: {y.shape}')
            return X, y
        except Exception as e:
            raise CustomException(e, sys)
    
    def apply_label_encoding(self, X: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        """
        Apply LabelEncoder to categorical columns.
        """
        try:
            X_encoded = X.copy()
            for col in categorical_columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.label_encoders[col] = le
                logging.info(f'Label encoded column: {col}')
            return X_encoded
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_preprocessor_pipeline(self, categorical_columns: list, numerical_columns: list):
        """
        Create ColumnTransformer for preprocessing.
        """
        # Ordinal mapping for parental level of education
        parental_map = {
            'some high school': 1, 'high school': 2, 'some college': 3, 
            "associate's degree": 4, "bachelor's degree": 5, 'master\'s degree': 6
        }
        
        cat_pipeline = Pipeline([
            ('label_encoder', 'passthrough')  # We'll apply label manually for all
        ])
        
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        logging.info(f'Numerical columns for scaling: {numerical_columns}')
        
        preprocessor = ColumnTransformer([
            ('cat_pipeline', cat_pipeline, categorical_columns),
            ('num_pipeline', num_pipeline, numerical_columns)
        ], remainder='passthrough')
        
        return preprocessor
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Main method: Load data, preprocess, save artifacts.
        """
        try:
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_path), exist_ok=True)
            
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f'Train data loaded: shape {train_df.shape}')
            logging.info(f'Test data loaded: shape {test_df.shape}')
            
            # Get column types from train
            categorical_columns, numerical_columns = self.get_data_types(train_df)
            
            # Separate target
            train_x, train_y = self.separate_target_feature(train_df)
            test_x, test_y = self.separate_target_feature(test_df)
            
            # Apply label encoding to categoricals (fit on train)
            train_x_encoded = self.apply_label_encoding(train_x, categorical_columns)
            test_x_encoded = test_x.copy()
            for col in categorical_columns:
                test_x_encoded[col] = self.label_encoders[col].transform(test_x_encoded[col].astype(str))
            
            # Combine numerical + encoded cat (now all numerical)
            train_arr = np.column_stack((train_x_encoded, train_x[numerical_columns].values))
            test_arr = np.column_stack((test_x_encoded, test_x[numerical_columns].values))
            
            # Note: numerical_columns may be empty if no num features besides target
            
            # Scale numerical features
            if numerical_columns:
                scaler = StandardScaler()
                train_arr[:, -len(numerical_columns):] = scaler.fit_transform(train_arr[:, -len(numerical_columns):])
                test_arr[:, -len(numerical_columns):] = scaler.transform(test_arr[:, -len(numerical_columns):])
                logging.info('Applied StandardScaler to numerical features')
            else:
                logging.info('No numerical features to scale')
            
            # Create final datasets
            train_transformed = pd.DataFrame(train_arr, columns=[f'{col}_encoded' if col in categorical_columns else col for col in train_x.columns] + numerical_columns)
            train_transformed[self.transformation_config.target_column] = train_y
            
            test_transformed = pd.DataFrame(test_arr, columns=[f'{col}_encoded' if col in categorical_columns else col for col in test_x.columns] + numerical_columns)
            test_transformed[self.transformation_config.target_column] = test_y
            
            # Save transformed data
            train_transformed.to_csv(self.transformation_config.transformed_train_path, index=False)
            test_transformed.to_csv(self.transformation_config.transformed_test_path, index=False)
            
            # Save encoders/scaler as preprocessor
            preprocessor_obj = {
                'label_encoders': self.label_encoders,
                'scaler': scaler if numerical_columns else None,
                'categorical_columns': categorical_columns,
                'numerical_columns': numerical_columns,
                'target_column': self.transformation_config.target_column
            }
            
            with open(self.transformation_config.preprocessor_obj_path, 'wb') as f:
                pickle.dump(preprocessor_obj, f)
            
            logging.info(f'Data transformation completed successfully.')
            logging.info(f'Transformed train saved to: {self.transformation_config.transformed_train_path}')
            logging.info(f'Transformed test saved to: {self.transformation_config.transformed_test_path}')
            logging.info(f'Preprocessor saved to: {self.transformation_config.preprocessor_obj_path}')
            
            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.transformation_config.preprocessor_obj_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


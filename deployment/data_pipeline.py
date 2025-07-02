# data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_wine
import mlflow
import mlflow.sklearn

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and prepare wine dataset"""
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        return df
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess the data for training"""
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def log_data_info(self, X_train, X_test, y_train, y_test):
        """Log data information to MLflow"""
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))
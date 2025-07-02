# model_training.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class ModelTrainer:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        
    def train_random_forest(self, X_train, X_test, y_train, y_test, 
                           n_estimators=100, max_depth=10, random_state=42):
        """Train Random Forest model with MLflow tracking"""
        
        with mlflow.start_run(run_name="RandomForest_Wine_Classification"):
            # Set tags
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("dataset", "wine")
            mlflow.set_tag("framework", "sklearn")
            
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", random_state)
            
            # Train model
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = rf_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Create and log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Random Forest')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig("confusion_matrix_rf.png")
            mlflow.log_artifact("confusion_matrix_rf.png")
            plt.close()
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            with open("classification_report_rf.txt", "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact("classification_report_rf.txt")
            
            # Log model
            mlflow.sklearn.log_model(
                rf_model,
                "random_forest_model",
                registered_model_name="wine_classification_rf",
                signature=mlflow.models.infer_signature(X_train, y_pred)
            )
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance - Random Forest')
            plt.tight_layout()
            plt.savefig("feature_importance_rf.png")
            mlflow.log_artifact("feature_importance_rf.png")
            plt.close()
            
            return rf_model, mlflow.active_run().info.run_id
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test,
                                 C=1.0, max_iter=1000, random_state=42):
        """Train Logistic Regression model with MLflow tracking"""
        
        with mlflow.start_run(run_name="LogisticRegression_Wine_Classification"):
            # Set tags
            mlflow.set_tag("model_type", "LogisticRegression")
            mlflow.set_tag("dataset", "wine")
            mlflow.set_tag("framework", "sklearn")
            
            # Log parameters
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("random_state", random_state)
            
            # Train model
            lr_model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state
            )
            lr_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = lr_model.predict(X_test)
            
            # Calculate and log metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(
                lr_model,
                "logistic_regression_model",
                registered_model_name="wine_classification_lr",
                signature=mlflow.models.infer_signature(X_train, y_pred)
            )
            
            return lr_model, mlflow.active_run().info.run_id

    def hyperparameter_tuning(self, X_train, X_test, y_train, y_test):
        """Perform hyperparameter tuning with MLflow tracking"""
        from sklearn.model_selection import GridSearchCV
        
        with mlflow.start_run(run_name="Hyperparameter_Tuning_RF"):
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            
            # Perform grid search
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Log best parameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Log best score
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Train final model with best parameters
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # Log final metrics
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", accuracy)
            
            # Log best model
            mlflow.sklearn.log_model(
                best_model,
                "best_random_forest_model",
                registered_model_name="wine_classification_best_rf"
            )
            
            return best_model, mlflow.active_run().info.run_id
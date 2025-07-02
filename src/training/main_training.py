# main_training.py
import mlflow
from deployment.data_pipeline import DataPipeline
from training.model_training import ModelTrainer

def main():
    """Main training pipeline"""
    # Set experiment
    mlflow.set_experiment("Wine_Classification_Experiment")
    
    # Initialize components
    data_pipeline = DataPipeline()
    trainer = ModelTrainer()
    
    # Load and preprocess data
    df = data_pipeline.load_data()
    X_train, X_test, y_train, y_test = data_pipeline.preprocess_data(df)
    
    print("Starting model training...")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model, rf_run_id = trainer.train_random_forest(
        X_train, X_test, y_train, y_test,
        n_estimators=100, max_depth=10
    )
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model, lr_run_id = trainer.train_logistic_regression(
        X_train, X_test, y_train, y_test,
        C=1.0, max_iter=1000
    )
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    best_model, best_run_id = trainer.hyperparameter_tuning(
        X_train, X_test, y_train, y_test
    )
    
    print(f"Random Forest Run ID: {rf_run_id}")
    print(f"Logistic Regression Run ID: {lr_run_id}")
    print(f"Best Model Run ID: {best_run_id}")
    
    return best_run_id

if __name__ == "__main__":
    best_run_id = main()
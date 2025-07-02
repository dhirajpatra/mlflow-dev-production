# model_registry.py
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def register_model(self, run_id, model_path, model_name):
        """Register a model to the model registry"""
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        return model_version
    
    def transition_model_stage(self, model_name, version, stage):
        """Transition model to different stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print(f"Model {model_name} version {version} transitioned to {stage}")
    
    def get_model_version(self, model_name, stage="Production"):
        """Get specific model version"""
        model_version = self.client.get_latest_versions(
            model_name, stages=[stage]
        )
        
        if model_version:
            return model_version[0]
        return None
    
    def load_production_model(self, model_name):
        """Load production model"""
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    
    def compare_models(self, model_name, versions):
        """Compare different model versions"""
        comparison_data = []
        
        for version in versions:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            comparison_data.append({
                'version': version,
                'stage': model_version.current_stage,
                'accuracy': run.data.metrics.get('accuracy', 0),
                'f1_score': run.data.metrics.get('f1_score', 0),
                'run_id': model_version.run_id
            })
        
        return comparison_data

# Usage example
def manage_model_lifecycle():
    registry = ModelRegistry()
    
    # Register best model
    model_version = registry.register_model(
        run_id="your_best_run_id",
        model_path="best_random_forest_model",
        model_name="wine_classifier"
    )
    
    # Transition to staging
    registry.transition_model_stage(
        model_name="wine_classifier",
        version=model_version.version,
        stage="Staging"
    )
    
    # After validation, transition to production
    registry.transition_model_stage(
        model_name="wine_classifier",
        version=model_version.version,
        stage="Production"
    )
    
    return model_version
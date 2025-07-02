# azure_deployment.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
import mlflow.azureml

class AzureMLDeployment:
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
    
    def create_model_asset(self, model_name, model_path, run_id):
        """Create model asset in Azure ML"""
        model_uri = f"runs:/{run_id}/{model_path}"
        
        model = Model(
            name=model_name,
            path=model_uri,
            description="Wine classification model trained with MLflow",
            type="mlflow_model"
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        return registered_model
    
    def create_endpoint(self, endpoint_name, description="Wine Classification Endpoint"):
        """Create managed online endpoint"""
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description,
            auth_mode="key"
        )
        
        endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(
            endpoint
        ).result()
        
        return endpoint_result
    
    def create_deployment(self, endpoint_name, deployment_name, model_name, 
                         model_version, instance_type="Standard_F2s_v2"):
        """Create deployment for the endpoint"""
        
        # Create deployment configuration
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"azureml:{model_name}:{model_version}",
            instance_type=instance_type,
            instance_count=1,
            environment_variables={
                "MLFLOW_MODEL_DIRECTORY": "/var/azureml-app/azureml-models"
            }
        )
        
        deployment_result = self.ml_client.online_deployments.begin_create_or_update(
            deployment
        ).result()
        
        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        return deployment_result
    
    def test_deployment(self, endpoint_name, test_data):
        """Test the deployed model"""
        import json
        
        # Prepare test data
        test_json = json.dumps({
            "inputs": test_data.tolist()
        })
        
        # Make prediction
        response = self.ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=test_json
        )
        
        return response

# Scoring script for Azure ML
scoring_script = """
import os
import logging
import json
import mlflow
import mlflow.sklearn
import pandas as pd

def init():
    global model
    
    # Load model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.sklearn.load_model(model_path)
    
    logging.info("Model loaded successfully")

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        inputs = data['inputs']
        
        # Convert to DataFrame
        df = pd.DataFrame(inputs)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Prepare response
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logging.error(f"Error in scoring: {str(e)}")
        return json.dumps({'error': str(e)})
"""

# Save scoring script
with open("score.py", "w") as f:
    f.write(scoring_script)
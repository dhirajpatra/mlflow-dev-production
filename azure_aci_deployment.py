# azure_aci_deployment.py
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model

class AzureACIDeployment:
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.workspace = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )
    
    def deploy_to_aci(self, model_name, service_name, model_uri):
        """Deploy MLflow model to Azure Container Instance"""
        
        # Configuration for ACI
        aci_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            description="Wine classification model deployment"
        )
        
        # Deploy model
        service = mlflow.azureml.deploy(
            model_uri=model_uri,
            workspace=self.workspace,
            deployment_config=aci_config,
            service_name=service_name
        )
        
        # Wait for deployment to complete
        service.wait_for_deployment(show_output=True)
        
        return service
    
    def test_aci_service(self, service, test_data):
        """Test ACI web service"""
        import json
        
        # Prepare test data
        test_json = json.dumps({
            "data": test_data.tolist()
        })
        
        # Make prediction
        result = service.run(test_json)
        return result

# Usage example
def deploy_model_to_azure():
    """Deploy model to Azure"""
    
    # Azure configuration
    subscription_id = "your-subscription-id"
    resource_group = "your-resource-group"
    workspace_name = "your-workspace-name"
    
    # Initialize deployment
    azure_deployment = AzureMLDeployment(
        subscription_id, resource_group, workspace_name
    )
    
    # Create model asset
    model_asset = azure_deployment.create_model_asset(
        model_name="wine-classifier",
        model_path="best_random_forest_model",
        run_id="your_best_run_id"
    )
    
    # Create endpoint
    endpoint = azure_deployment.create_endpoint(
        endpoint_name="wine-classification-endpoint"
    )
    
    # Create deployment
    deployment = azure_deployment.create_deployment(
        endpoint_name="wine-classification-endpoint",
        deployment_name="wine-classification-deployment",
        model_name="wine-classifier",
        model_version=model_asset.version
    )
    
    print(f"Model deployed successfully: {endpoint.scoring_uri}")
    return endpoint.scoring_uri
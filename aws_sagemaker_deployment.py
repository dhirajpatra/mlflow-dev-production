# aws_sagemaker_deployment.py
import boto3
import mlflow
import mlflow.sagemaker
from sagemaker import get_execution_role
import json
import tarfile
import os

class AWSDeployment:
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        
    def deploy_to_sagemaker(self, model_uri, app_name, region_name, 
                           execution_role_arn, instance_type="ml.t2.medium"):
        """Deploy MLflow model to SageMaker"""
        
        # Deploy using MLflow SageMaker integration
        deployment = mlflow.sagemaker.deploy(
            app_name=app_name,
            model_uri=model_uri,
            region_name=region_name,
            mode="create",
            execution_role_arn=execution_role_arn,
            instance_type=instance_type,
            instance_count=1,
            vpc_config=None,
            flavor=None
        )
        
        return deployment
    
    def create_sagemaker_model(self, model_name, model_uri, execution_role):
        """Create SageMaker model from MLflow model"""
        
        # Build and push container image
        image_uri = mlflow.sagemaker.build_image(model_uri)
        
        # Create model
        response = self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn=execution_role
        )
        
        return response
    
    def create_endpoint_configuration(self, config_name, model_name, 
                                    instance_type="ml.t2.medium"):
        """Create endpoint configuration"""
        
        response = self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        return response
    
    def create_endpoint(self, endpoint_name, config_name):
        """Create SageMaker endpoint"""
        
        response = self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        # Wait for endpoint to be in service
        waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        return response
    
    def invoke_endpoint(self, endpoint_name, payload):
        """Invoke SageMaker endpoint"""
        
        runtime_client = boto3.client('sagemaker-runtime', 
                                    region_name=self.region_name)
        
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        return result

# Lambda deployment
lambda_deployment_code = """
import json
import mlflow
import mlflow.sklearn
import boto3
import os

# Load model once during cold start
model = None

def lambda_handler(event, context):
    global model
    
    if model is None:
        # Load model from S3
        model_uri = os.environ['MLFLOW_MODEL_URI']
        model = mlflow.sklearn.load_model(model_uri)
    
    try:
        # Parse input
        body = json.loads(event['body'])
        input_data = body['data']
        
        # Make prediction
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Prepare response
        response_body = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
"""

class AWSLambdaDeployment:
    def __init__(self, region_name='us-east-1'):
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        
    def create_deployment_package(self, model_uri, function_name):
        """Create Lambda deployment package"""
        
        # Create temporary directory
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Save model
            model_path = os.path.join(temp_dir, 'model.pkl')
            import joblib
            joblib.dump(model, model_path)
            
            # Create lambda function
            lambda_file = os.path.join(temp_dir, 'lambda_function.py')
            with open(lambda_file, 'w') as f:
                f.write(lambda_deployment_code)
            
            # Create deployment package
            package_path = f"{function_name}.zip"
            shutil.make_archive(
                package_path.replace('.zip', ''), 
                'zip', 
                temp_dir
            )
            
            return package_path
    
    def deploy_lambda_function(self, function_name, model_uri, execution_role_arn):
        """Deploy model as Lambda function"""
        
        # Create deployment package
        package_path = self.create_deployment_package(model_uri, function_name)
        
        # Read deployment package
        with open(package_path, 'rb') as f:
            zip_content = f.read()
        
        # Create Lambda function
        response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role=execution_role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Environment={
                'Variables': {
                    'MLFLOW_MODEL_URI': model_uri
                }
            },
            Timeout=30,
            MemorySize=512
        )
        
        return response

# Usage example for AWS deployment
def deploy_model_to_aws():
    """Deploy model to AWS"""
    
    # AWS configuration
    region_name = 'us-east-1'
    execution_role_arn = 'arn:aws:iam::account:role/SageMakerExecutionRole'
    
    # Initialize deployment
    aws_deployment = AWSDeployment(region_name=region_name)
    
    # Model URI from MLflow
    model_uri = "runs:/your_best_run_id/best_random_forest_model"
    
    # Deploy to SageMaker
    print("Deploying to SageMaker...")
    endpoint_name = "wine-classification-endpoint"
    
    # Create model
    model_response = aws_deployment.create_sagemaker_model(
        model_name="wine-classifier-model",
        model_uri=model_uri,
        execution_role=execution_role_arn
    )
    
    # Create endpoint configuration
    config_response = aws_deployment.create_endpoint_configuration(
        config_name="wine-classifier-config",
        model_name="wine-classifier-model"
    )
    
    # Create endpoint
    endpoint_response = aws_deployment.create_endpoint(
        endpoint_name=endpoint_name,
        config_name="wine-classifier-config"
    )
    
    print(f"SageMaker endpoint created: {endpoint_name}")
    
    # Deploy to Lambda
    print("Deploying to Lambda...")
    lambda_deployment = AWSLambdaDeployment(region_name=region_name)
    lambda_response = lambda_deployment.deploy_lambda_function(
        function_name="wine-classification-lambda",
        model_uri=model_uri,
        execution_role_arn=execution_role_arn
    )
    
    print(f"Lambda function created: {lambda_response['FunctionArn']}")
    
    return endpoint_name, lambda_response['FunctionArn']
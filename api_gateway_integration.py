# api_gateway_integration.py
import boto3
import json

class APIGatewayIntegration:
    def __init__(self, region_name='us-east-1'):
        self.apigateway_client = boto3.client('apigateway', region_name=region_name)
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        
    def create_rest_api(self, api_name, lambda_function_arn):
        """Create REST API with Lambda integration"""
        
        # Create API
        api_response = self.apigateway_client.create_rest_api(
            name=api_name,
            description='Wine Classification API'
        )
        api_id = api_response['id']
        
        # Get root resource
        resources = self.apigateway_client.get_resources(restApiId=api_id)
        root_resource_id = resources['items'][0]['id']
        
        # Create resource
        resource_response = self.apigateway_client.create_resource(
            restApiId=api_id,
            parentId=root_resource_id,
            pathPart='predict'
        )
        resource_id = resource_response['id']
        
        # Create method
        self.apigateway_client.put_method(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            authorizationType='NONE'
        )
        
        # Set integration
        self.apigateway_client.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f'arn:aws:apigateway:{self.apigateway_client.meta.region_name}:lambda:path/2015-03-31/functions/{lambda_function_arn}/invocations'
        )
        
        # Deploy API
        deployment_response = self.apigateway_client.create_deployment(
            restApiId=api_id,
            stageName='prod'
        )
        
        # Grant API Gateway permission to invoke Lambda
        self.lambda_client.add_permission(
            FunctionName=lambda_function_arn,
            StatementId='api-gateway-invoke',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com'
        )
        
        api_url = f"https://{api_id}.execute-api.{self.apigateway_client.meta.region_name}.amazonaws.com/prod/predict"
        
        return api_url
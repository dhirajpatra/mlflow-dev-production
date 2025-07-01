# aws_mlflow_server.py
import boto3
from mlflow.server import get_app

def setup_aws_mlflow_server():
    """Setup MLflow server with AWS backend"""
    # S3 bucket for artifacts
    s3_bucket = "your-mlflow-artifacts-bucket"
    
    # RDS database for tracking
    db_uri = "postgresql://username:password@rds-endpoint:5432/mlflow"
    
    # Start server with AWS backend
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", db_uri,
        "--default-artifact-root", f"s3://{s3_bucket}/mlflow-artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    return cmd

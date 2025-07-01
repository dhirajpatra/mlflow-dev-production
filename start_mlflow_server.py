# start_mlflow_server.py
import subprocess
import os

def start_mlflow_server():
    """Start MLflow tracking server locally"""
    # Create artifacts directory
    os.makedirs("mlflow_artifacts", exist_ok=True)
    
    # Start MLflow server
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlflow_artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    start_mlflow_server()
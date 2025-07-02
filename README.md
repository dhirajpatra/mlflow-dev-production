# MLflow End-to-End ML Pipeline

A comprehensive MLflow implementation demonstrating the complete machine learning lifecycle from model development to production deployment on Azure and AWS platforms.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd mlflow-project
pip install -r requirements.txt

# Follow the step-by-step instructions in run_steps.txt
cat run_steps.txt
```

## 📋 Prerequisites

- Python 3.8+
- MLflow 2.0+
- Azure CLI (for Azure deployment)
- AWS CLI (for AWS deployment)
- Docker (optional, for containerized deployments)

### Required Python Packages

```bash
pip install mlflow pandas scikit-learn boto3 azure-ml-core
```

### Cloud Setup

**Azure:**

```bash
az login
az account set --subscription <your-subscription-id>
```

**AWS:**

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

## 🏗️ Project Structure

```
├── models/                    # Trained models and artifacts
├── data/                     # Training and reference datasets
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── training/
│   │   ├── main_training.py          # Main training pipeline
│   │   └── model_utils.py            # Model utilities
│   ├── deployment/
│   │   ├── azure_deployment.py      # Azure ML deployment
│   │   └── aws_sagemaker_deployment.py  # AWS SageMaker deployment
│   ├── monitoring/
│   │   ├── model_monitoring.py      # Automated monitoring
│   │   └── ab_testing.py            # A/B testing framework
│   └── utils/
│       └── start_mlflow_server.py   # MLflow server setup
├── config/                   # Configuration files
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── run_steps.txt              # Step-by-step pipeline execution guide
└── README.md                # This file
```

## 🔧 Setup Instructions

### 1. MLflow Tracking Server

Start the MLflow tracking server locally:

```bash
python src/utils/start_mlflow_server.py
```

Or configure remote tracking server:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

### 2. Environment Configuration

Create a `.env` file with your configuration:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=your-s3-endpoint
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret

# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
```

## 🚂 Running the Pipeline

### Step-by-Step Execution

Follow the instructions in `run_steps.txt` for the complete pipeline:

```bash
# 1. Start MLflow server
python start_mlflow_server.py

# 2. Train models
python main_training.py

# 3. Deploy to cloud (choose one)
python azure_deployment.py
# OR
python aws_sagemaker_deployment.py

# 4. Set up monitoring
python -c "
from model_monitoring import AutomatedMonitoring
monitor = AutomatedMonitoring('your-endpoint-url', 'reference_data.csv')
monitor.run_monitoring_cycle()
"

# 5. Run A/B test
python -c "
from ab_testing import ABTestingFramework
ab_test = ABTestingFramework()
ab_test.setup_ab_test('model_uri_a', 'model_uri_b')
# Run predictions and analyze
"
```

### Automated Execution

For convenience, you can also run all steps sequentially:

```bash
# Execute each step from run_steps.txt
while IFS= read -r line; do
    if [[ $line == python* ]]; then
        echo "Executing: $line"
        eval "$line"
    fi
done < run_steps.txt
```

### Individual Components

**Training Only:**

```bash
python main_training.py --experiment-name "my-experiment"
```

**Deployment Only:**

```bash
# Azure
python azure_deployment.py --model-uri "models:/my-model/1"

# AWS
python aws_sagemaker_deployment.py --model-uri "models:/my-model/1"
```

## 📊 Model Development and Tracking

### Training Process

The training pipeline includes:

- Data preprocessing and feature engineering
- Model training with hyperparameter tuning
- Automatic experiment tracking with MLflow
- Model validation and metrics logging
- Model registration in MLflow Model Registry

### Key Features

- **Experiment Tracking**: All runs are automatically logged with parameters, metrics, and artifacts
- **Model Versioning**: Models are versioned and stored in the MLflow Model Registry
- **Artifact Storage**: Training artifacts stored in cloud storage (S3/Azure Blob)
- **Reproducibility**: Environment and code version tracking

## ☁️ Cloud Deployment

### Azure ML

Deploy models to Azure ML managed endpoints:

```python
from src.deployment.azure_deployment import AzureMLDeployer

deployer = AzureMLDeployer(
    subscription_id="your-subscription-id",
    resource_group="your-rg",
    workspace_name="your-workspace"
)

endpoint_url = deployer.deploy_model(
    model_uri="models:/my-model/1",
    endpoint_name="my-model-endpoint"
)
```

### AWS SageMaker

Deploy models to Amazon SageMaker:

```python
from src.deployment.aws_sagemaker_deployment import SageMakerDeployer

deployer = SageMakerDeployer(region="us-west-2")
endpoint_name = deployer.deploy_model(
    model_uri="models:/my-model/1",
    instance_type="ml.t2.medium"
)
```

## 📈 Monitoring and Production

### Automated Monitoring

The monitoring system tracks:

- **Data Drift**: Statistical tests for input data distribution changes
- **Model Performance**: Accuracy, precision, recall metrics over time
- **Infrastructure**: Latency, throughput, error rates
- **Business Metrics**: Custom KPIs and business-specific metrics

### A/B Testing

Compare model performance in production:

```python
from src.monitoring.ab_testing import ABTestingFramework

ab_test = ABTestingFramework()
results = ab_test.run_experiment(
    control_model="models:/my-model/1",
    treatment_model="models:/my-model/2",
    traffic_split=0.5,
    duration_days=7
)
```

## 🔍 Troubleshooting

### Common Issues

**MLflow Server Connection:**

```bash
# Check if server is running
curl http://localhost:5000/health

# Verify environment variable
echo $MLFLOW_TRACKING_URI
```

**Cloud Authentication:**

```bash
# Azure
az account show

# AWS
aws sts get-caller-identity
```

**Model Loading Issues:**

- Verify model URI format: `models:/model-name/version`
- Check MLflow Model Registry permissions
- Ensure artifact store is accessible

**Deployment Failures:**

- Verify cloud resource quotas and limits
- Check networking and security group configurations
- Review deployment logs in cloud console

### Getting Help

1. Check the [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
2. Review cloud provider documentation:
   - [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
   - [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
3. Open an issue in this repository

## 🛡️ Security Best Practices

- **Secrets Management**: Use cloud secret managers (Azure Key Vault, AWS Secrets Manager)
- **Access Control**: Implement proper IAM roles and permissions
- **Network Security**: Use VPNs and private endpoints where possible
- **Data Encryption**: Enable encryption at rest and in transit
- **Audit Logging**: Enable comprehensive audit trails

## 🚀 Advanced Features

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: MLflow Pipeline
on: [push]
jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run MLflow Pipeline
        run: |
          python src/training/main_training.py
          python src/deployment/azure_deployment.py
```

### Scaling Considerations

- Use remote MLflow tracking server for team collaboration
- Implement distributed training for large datasets
- Use cloud-native storage for artifacts (S3, Azure Blob)
- Consider MLflow Model Registry for centralized model management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🌐 Guide

Follow for guide [MLflow Documentation](https://mlflow.org/docs/latest/ml/getting-started/)
Follow other solutions and tutorial [Think Different](https://dhirajpatra.blogspot.com/)


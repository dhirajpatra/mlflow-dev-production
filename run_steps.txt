# 1. Start MLflow server
python start_mlflow_server.py

# 2. Train models
python main_training.py

# 3. Deploy to cloud
python azure_deployment.py  # or aws_sagemaker_deployment.py

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
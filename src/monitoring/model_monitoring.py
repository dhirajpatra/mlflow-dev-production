# model_monitoring.py
import mlflow
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

class ModelMonitor:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    def log_prediction_metrics(self, model_name, predictions, actuals, 
                              features, timestamp=None):
        """Log prediction metrics for monitoring"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        with mlflow.start_run(run_name=f"monitoring_{timestamp.strftime('%Y%m%d_%H%M%S')}"):
            # Set tags
            mlflow.set_tag("monitoring", "true")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("timestamp", timestamp.isoformat())
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, average='weighted')
            recall = recall_score(actuals, predictions, average='weighted')
            f1 = f1_score(actuals, predictions, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("sample_count", len(predictions))
            
            # Log feature statistics
            feature_stats = {
                'mean': features.mean().to_dict(),
                'std': features.std().to_dict(),
                'min': features.min().to_dict(),
                'max': features.max().to_dict()
            }
            
            for stat_name, stat_values in feature_stats.items():
                for feature_name, value in stat_values.items():
                    mlflow.log_metric(f"feature_{feature_name}_{stat_name}", value)
    
    def detect_data_drift(self, reference_data, current_data, threshold=0.05):
        """Detect data drift using statistical tests"""
        
        drift_results = {}
        
        for column in reference_data.columns:
            if column in current_data.columns:
                # Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(
                    reference_data[column], 
                    current_data[column]
                )
                
                drift_results[column] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
        
        return drift_results
    
    def generate_monitoring_report(self, model_name, days_back=7):
        """Generate monitoring report"""
        
        # Get recent monitoring runs
        experiment = mlflow.get_experiment_by_name("Wine_Classification_Experiment")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.monitoring = 'true' and tags.model_name = '{model_name}'",
            max_results=days_back * 24  # Assuming hourly monitoring
        )
        
        if runs.empty:
            return "No monitoring data found"
        
        # Create report
        report = {
            'model_name': model_name,
            'monitoring_period': f"Last {days_back} days",
            'total_predictions': runs['metrics.sample_count'].sum(),
            'average_accuracy': runs['metrics.accuracy'].mean(),
            'accuracy_trend': runs['metrics.accuracy'].tolist(),
            'timestamps': runs['tags.timestamp'].tolist()
        }
        
        return report

# Automated monitoring pipeline
class AutomatedMonitoring:
    def __init__(self, model_endpoint, reference_data_path):
        self.model_endpoint = model_endpoint
        self.reference_data = pd.read_csv(reference_data_path)
        self.monitor = ModelMonitor()
        
    def collect_predictions(self, time_window_hours=1):
        """Collect predictions from the last time window"""
        # This would typically connect to your prediction logs
        # For demonstration, we'll simulate this
        
        # Simulate prediction data
        n_samples = np.random.randint(50, 200)
        predictions = np.random.randint(0, 3, n_samples)
        actuals = np.random.randint(0, 3, n_samples)
        
        # Simulate feature data
        features = pd.DataFrame(
            np.random.randn(n_samples, len(self.reference_data.columns)),
            columns=self.reference_data.columns
        )
        
        return predictions, actuals, features
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        
        # Collect recent predictions
        predictions, actuals, features = self.collect_predictions()
        
        # Log metrics
        self.monitor.log_prediction_metrics(
            model_name="wine_classifier",
            predictions=predictions,
            actuals=actuals,
            features=features
        )
        
        # Check for data drift
        drift_results = self.monitor.detect_data_drift(
            self.reference_data,
            features
        )
        
        # Check for significant drift
        significant_drift = any(
            result['drift_detected'] for result in drift_results.values()
        )
        
        if significant_drift:
            print("ALERT: Data drift detected!")
            print("Drifted features:", [
                feature for feature, result in drift_results.items()
                if result['drift_detected']
            ])
            
            # Trigger retraining (would be implemented based on your pipeline)
            self.trigger_retraining()
        
        return drift_results
    
    def trigger_retraining(self):
        """Trigger model retraining"""
        print("Triggering model retraining pipeline...")
        # This would typically trigger your training pipeline
        # Could use Apache Airflow, AWS Step Functions, etc.
        pass

# Performance tracking
class PerformanceTracker:
    def __init__(self):
        self.prediction_times = []
        self.error_counts = {}
        
    def log_prediction_time(self, duration):
        """Log prediction latency"""
        self.prediction_times.append({
            'timestamp': datetime.now(),
            'duration_ms': duration * 1000
        })
        
        # Keep only last 1000 entries
        if len(self.prediction_times) > 1000:
            self.prediction_times = self.prediction_times[-1000:]
    
    def log_error(self, error_type, error_message):
        """Log prediction errors"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.set_tag("error_tracking", "true")
            mlflow.set_tag("error_type", error_type)
            mlflow.log_param("error_message", error_message)
            mlflow.log_metric("error_count", self.error_counts[error_type])
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        if not self.prediction_times:
            return {}
            
        durations = [p['duration_ms'] for p in self.prediction_times]
        
        return {
            'avg_latency_ms': np.mean(durations),
            'p95_latency_ms': np.percentile(durations, 95),
            'p99_latency_ms': np.percentile(durations, 99),
            'total_predictions': len(durations),
            'error_rate': sum(self.error_counts.values()) / len(durations) if durations else 0
        }
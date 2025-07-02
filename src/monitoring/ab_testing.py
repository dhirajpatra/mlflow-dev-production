# ab_testing.py
import mlflow
import numpy as np
import pandas as pd
from scipy import stats
import random
from datetime import datetime

class ABTestingFramework:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    def setup_ab_test(self, model_a_uri, model_b_uri, traffic_split=0.5):
        """Setup A/B test between two models"""
        
        # Load models
        self.model_a = mlflow.sklearn.load_model(model_a_uri)
        self.model_b = mlflow.sklearn.load_model(model_b_uri)
        self.traffic_split = traffic_split
        
        # Initialize tracking
        self.test_results = {
            'model_a': {'predictions': [], 'response_times': [], 'errors': 0},
            'model_b': {'predictions': [], 'response_times': [], 'errors': 0}
        }
        
        # Start MLflow run for A/B test
        self.ab_run = mlflow.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.set_tag("experiment_type", "ab_test")
        mlflow.log_param("traffic_split", traffic_split)
        mlflow.log_param("model_a_uri", model_a_uri)
        mlflow.log_param("model_b_uri", model_b_uri)
        
    def route_prediction(self, features):
        """Route prediction to model A or B based on traffic split"""
        
        # Determine which model to use
        use_model_a = random.random() < self.traffic_split
        
        try:
            start_time = datetime.now()
            
            if use_model_a:
                prediction = self.model_a.predict(features.reshape(1, -1))[0]
                model_used = 'model_a'
            else:
                prediction = self.model_b.predict(features.reshape(1, -1))[0]
                model_used = 'model_b'
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Log results
            self.test_results[model_used]['predictions'].append(prediction)
            self.test_results[model_used]['response_times'].append(response_time)
            
            return prediction, model_used
            
        except Exception as e:
            if use_model_a:
                self.test_results['model_a']['errors'] += 1
            else:
                self.test_results['model_b']['errors'] += 1
            raise e
    
    def analyze_ab_test(self, metric='accuracy', actuals=None):
        """Analyze A/B test results"""
        
        if metric == 'accuracy' and actuals is not None:
            # Calculate accuracy for both models
            model_a_accuracy = np.mean([
                pred == actual for pred, actual in 
                zip(self.test_results['model_a']['predictions'], actuals[:len(self.test_results['model_a']['predictions'])])
            ])
            
            model_b_accuracy = np.mean([
                pred == actual for pred, actual in 
                zip(self.test_results['model_b']['predictions'], actuals[:len(self.test_results['model_b']['predictions'])])
            ])
            
            # Statistical significance test
            n_a = len(self.test_results['model_a']['predictions'])
            n_b = len(self.test_results['model_b']['predictions'])
            
            # Z-test for proportions
            p_a = model_a_accuracy
            p_b = model_b_accuracy
            p_pool = (n_a * p_a + n_b * p_b) / (n_a + n_b)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
            z_score = (p_b - p_a) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Log results to MLflow
            mlflow.log_metric("model_a_accuracy", model_a_accuracy)
            mlflow.log_metric("model_b_accuracy", model_b_accuracy)
            mlflow.log_metric("accuracy_difference", p_b - p_a)
            mlflow.log_metric("z_score", z_score)
            mlflow.log_metric("p_value", p_value)
            mlflow.log_metric("statistical_significance", p_value < 0.05)
            
            return {
                'model_a_accuracy': model_a_accuracy,
                'model_b_accuracy': model_b_accuracy,
                'difference': p_b - p_a,
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'winner': 'model_b' if p_b > p_a and p_value < 0.05 else 'model_a' if p_a > p_b and p_value < 0.05 else 'inconclusive'
            }
        
        elif metric == 'response_time':
            # Analyze response times
            model_a_times = self.test_results['model_a']['response_times']
            model_b_times = self.test_results['model_b']['response_times']
            
            if model_a_times and model_b_times:
                # T-test for response times
                t_stat, p_value = stats.ttest_ind(model_a_times, model_b_times)
                
                avg_time_a = np.mean(model_a_times)
                avg_time_b = np.mean(model_b_times)
                
                mlflow.log_metric("model_a_avg_response_time", avg_time_a)
                mlflow.log_metric("model_b_avg_response_time", avg_time_b)
                mlflow.log_metric("response_time_difference", avg_time_b - avg_time_a)
                mlflow.log_metric("response_time_p_value", p_value)
                
                return {
                    'model_a_avg_time': avg_time_a,
                    'model_b_avg_time': avg_time_b,
                    'difference': avg_time_b - avg_time_a,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    def end_ab_test(self):
        """End A/B test and log final results"""
        
        # Log final counts
        mlflow.log_metric("model_a_total_predictions", len(self.test_results['model_a']['predictions']))
        mlflow.log_metric("model_b_total_predictions", len(self.test_results['model_b']['predictions']))
        mlflow.log_metric("model_a_errors", self.test_results['model_a']['errors'])
        mlflow.log_metric("model_b_errors", self.test_results['model_b']['errors'])
        
        mlflow.end_run()
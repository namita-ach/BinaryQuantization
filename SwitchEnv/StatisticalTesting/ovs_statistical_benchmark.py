#!/usr/bin/env python3
"""
Open vSwitch Anomaly Detection Benchmark with Statistical Runs
Compares Flow-Aware LSH vs Isolation Forest vs One-Class SVM
"""

import numpy as np
import pandas as pd
import time
import subprocess
import os
from collections import defaultdict
from sklearn.metrics import (classification_report, roc_auc_score, 
                           confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import psutil
import resource
import sys
import random
import gc
import tracemalloc
from memory_profiler import profile
import threading

# Constants
EMB_SIZE = 64
RESULTS_DIR = "resource-results"
TRAIN_FILE = "/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_training-set.csv"
TEST_FILE = "/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_testing-set.csv"

from SwitchEnv.StatisticalTesting.anomaly_detection import FlowAwareLSHAnomalyDetector, StreamingEmbedder
from SwitchEnv.StatisticalTesting.flow_processing import FlowFeatureExtractor, FlowAwareBinaryQuantizer
from SwitchEnv.StatisticalTesting.resource_metrics import ResourceMonitor

class StatisticalBenchmark:
    def __init__(self, embedding_size=EMB_SIZE):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.embedding_size = embedding_size
        self.seeds = [355, 1307, 6390, 9026, 2997, 9766, 1095, 4926, 276, 8706]
        
    def run_single_experiment(self, seed):
        monitor = ResourceMonitor()
        
        try:
            train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
            test_df = pd.read_csv(TEST_FILE, low_memory=False)
            print(f"Loaded datasets (seed: {seed})")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

        # Convert labels
        def fix_labels(df):
            if 'label' in df.columns:
                df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
            return df

        train_df = fix_labels(train_df)
        test_df = fix_labels(test_df)

        # Sample data
        normal_train = train_df[train_df['label'] == 0].sample(
            n=min(3000, len(train_df[train_df['label'] == 0])),
            random_state=seed
        )

        normal_test = test_df[test_df['label'] == 0].sample(
            n=min(800, len(test_df[test_df['label'] == 0])),
            random_state=seed
        )
        anomaly_test = test_df[test_df['label'] == 1].sample(
            n=min(400, len(test_df[test_df['label'] == 1])),
            random_state=seed
        )
        test_combined = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=seed)

        print(f"Training on {len(normal_train)} normal flows")
        print(f"Testing on {len(test_combined)} flows ({len(normal_test)} normal, {len(anomaly_test)} anomaly)")

        # Convert to lists
        train_flows = normal_train.to_dict('records')
        test_flows = test_combined.to_dict('records')
        y_true = test_combined['label'].tolist()

        # Initialize model
        embedder = StreamingEmbedder(embedding_size=self.embedding_size)
        
        # Measure training resources
        monitor.start_monitoring()
        start_time = time.time()
        embedder.train(train_flows, seed)
        train_time = time.time() - start_time
        train_resources = monitor.stop_monitoring()
        
        # Measure model sizes
        model_sizes = embedder.measure_model_sizes()
        
        # Measure testing resources
        def test_lsh_flow(flow):
            features = embedder.feature_extractor.get_optimized_features(flow)
            embedding = embedder.quantizer.quantize_flow(features)
            return embedder.detector.predict_anomaly(embedding)
        
        def test_baseline_flow(flow):
            features = embedder.feature_extractor.get_optimized_features(flow)
            feature_matrix = embedder._features_to_matrix([features])
            feature_matrix_scaled = embedder.scaler.transform(feature_matrix)
            if_pred = embedder.isolation_forest.predict(feature_matrix_scaled)
            svm_pred = embedder.one_class_svm.predict(feature_matrix_scaled)
            return if_pred[0], svm_pred[0]
        
        # Measure flow processing times
        lsh_flow_times = monitor.measure_flow_processing_time(test_lsh_flow, test_flows)
        baseline_flow_times = monitor.measure_flow_processing_time(test_baseline_flow, test_flows)
        
        # Full prediction
        monitor.start_monitoring()
        start_time = time.time()
        all_results = embedder.predict(test_flows)
        test_time = time.time() - start_time
        test_resources = monitor.stop_monitoring()

        # Evaluate models
        results_with_resources = {}
        for model_name, (predictions, scores) in all_results.items():
            y_pred = [1 if pred else 0 for pred in predictions]

            results_with_resources[model_name] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, scores),
                'predictions': predictions,
                'scores': scores
            }

        # Resource metrics
        resource_metrics = {
            'train_time_seconds': train_time,
            'test_time_seconds': test_time,
            'flows_processed': len(test_flows),
            'flows_per_second': len(test_flows) / test_time if test_time > 0 else 0,
            'train_memory_usage_kb': train_resources['memory_usage_kb'],
            'train_peak_memory_kb': train_resources['peak_memory_usage_kb'],
            'train_cpu_usage_percent': train_resources['cpu_usage_percent'],
            'test_memory_usage_kb': test_resources['memory_usage_kb'],
            'test_peak_memory_kb': test_resources['peak_memory_usage_kb'],
            'test_cpu_usage_percent': test_resources['cpu_usage_percent'],
            'lsh_avg_flow_time_ms': lsh_flow_times['avg_flow_processing_time_ms'],
            'baseline_avg_flow_time_ms': baseline_flow_times['avg_flow_processing_time_ms'],
            **model_sizes
        }

        return results_with_resources, resource_metrics

    def run_multiple_experiments(self):
        all_metrics = defaultdict(lambda: defaultdict(list))
        all_resources = defaultdict(list)
        
        for i, seed in enumerate(self.seeds):
            print(f"\n=== Experiment {i+1}/{len(self.seeds)} (Seed: {seed}) ===")
            
            result = self.run_single_experiment(seed)
            if result is None:
                continue
                
            results, resources = result
            
            # Collect performance metrics
            for model_name, metrics in results.items():
                for metric, value in metrics.items():
                    if metric not in ['predictions', 'scores']:
                        all_metrics[model_name][metric].append(value)
            
            # Collect resource metrics
            for metric, value in resources.items():
                all_resources[metric].append(value)
        
        # Save results
        self.save_results(all_metrics, all_resources)
        self.print_summary(all_metrics, all_resources)
        
        return all_metrics, all_resources

    def save_results(self, all_metrics, all_resources):
        # Save model metrics
        model_results = []
        for model_name, metrics in all_metrics.items():
            model_results.append({
                'Model': model_name,
                **{f'{metric}_Mean': np.mean(values) for metric, values in metrics.items()},
                **{f'{metric}_Std': np.std(values) for metric, values in metrics.items()}
            })
        pd.DataFrame(model_results).to_csv(f"{RESULTS_DIR}/model_results.csv", index=False)
        
        # Save resource metrics
        resource_results = {
            'Metric': list(all_resources.keys()),
            'Mean': [np.mean(values) for values in all_resources.values()],
            'Std': [np.std(values) for values in all_resources.values()]
        }
        pd.DataFrame(resource_results).to_csv(f"{RESULTS_DIR}/resource_results.csv", index=False)

    def print_summary(self, all_metrics, all_resources):
        print("\n=== FINAL SUMMARY ===")
        print("=" * 60)
        
        # Model performance
        print("\nModel Performance:")
        for model_name, metrics in all_metrics.items():
            print(f"\n{model_name}:")
            for metric, values in metrics.items():
                print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Resource usage
        print("\nResource Usage:")
        key_resources = ['train_time_seconds', 'test_time_seconds', 'flows_per_second',
                        'total_lsh_model_kb', 'total_baseline_models_kb']
        for metric in key_resources:
            if metric in all_resources:
                values = all_resources[metric]
                print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

def setup_ovs(): #Initialize Open vSwitch environment
    print("Setting up Open vSwitch...")
    try:
        subprocess.run(["sudo", "ovs-vsctl", "add-br", "ovs-br0"], check=True)
        print("Created OVS bridge 'ovs-br0'")
    except subprocess.CalledProcessError as e:
        print(f"OVS setup error: {e}")

def main():
    print("=== OVS Anomaly Detection Benchmark with Statistical Runs ===")
    
    # Initialize OVS
    setup_ovs()
    
    # Run benchmark
    benchmark = StatisticalBenchmark(embedding_size=EMB_SIZE)
    benchmark.run_multiple_experiments()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Open vSwitch Anomaly Detection Benchmark
Compares Flow-Aware LSH vs Isolation Forest vs One-Class SVM
"""

import numpy as np
import pandas as pd
import time
import subprocess
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import psutil
import resource
import sys
import os
from collections import defaultdict

from anomaly_detection import FlowAwareLSHAnomalyDetector, StreamingEmbedder
from flow_processing import FlowFeatureExtractor, FlowAwareBinaryQuantizer
from resource_metrics import ResourceMonitor

# Constants
EMB_SIZE = 64  # Embedding size for LSH
TRAIN_FILE = "/home/pes1ug22am100/Documents/BinaryQuantization/UNSW_NB15_training-set.csv"
TEST_FILE = "/home/pes1ug22am100/Documents/BinaryQuantization/UNSW_NB15_testing-set.csv"
RESULTS_DIR = "SwitchEnv/results"

class CSVStreamer:
    def __init__(self, file_path, chunk_size=100, speed_factor=1.0):
        """Stream CSV data with realistic timing"""
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.speed_factor = speed_factor
        self._reset()
        
    def _reset(self):
        try:
            self.df = pd.read_csv(self.file_path, chunksize=self.chunk_size)
            self.chunk_generator = iter(self.df)
            self.current_chunk = None
            self.current_index = 0
            self.start_time = time.time()
            self.last_ts = None
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            raise

    def get_next_flow(self):
        if self.current_chunk is None or self.current_index >= len(self.current_chunk):
            try:
                self.current_chunk = next(self.chunk_generator)
                self.current_index = 0
            except StopIteration:
                return None
        
        flow = self.current_chunk.iloc[self.current_index].to_dict()
        self.current_index += 1
        
        # Simulate real-time streaming using timestamps if available
        if 'ts' in flow:
            current_ts = flow['ts']
            if self.last_ts is not None and current_ts > self.last_ts:
                delay = (current_ts - self.last_ts) / self.speed_factor
                time.sleep(delay)
            self.last_ts = current_ts
            
        return flow

class AnomalyDetectorBenchmark:
    def __init__(self, embedding_size=EMB_SIZE):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.embedder = StreamingEmbedder(embedding_size=embedding_size)
        self.results = {
            'lsh': {'y_true': [], 'y_pred': [], 'scores': [], 'times': []},
            'iforest': {'y_true': [], 'y_pred': [], 'scores': [], 'times': []},
            'ocsvm': {'y_true': [], 'y_pred': [], 'scores': [], 'times': []}
        }
        
    def train(self, train_file=TRAIN_FILE):
        print(f"\nTraining models using {train_file}...")
        try:
            train_df = pd.read_csv(train_file)
            normal_train = train_df[train_df['label'] == 0].sample(n=3000)
            self.embedder.train(normal_train.to_dict('records'))
            print("Training completed successfully")
            return True
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False

    def process_flow(self, flow):
        """Process flow through all three models"""
        start_time = time.perf_counter()
        true_label = flow.get('label', 0)
        
        # Feature extraction
        features = self.embedder.feature_extractor.get_optimized_features(flow)
        feature_matrix = self.embedder._features_to_matrix([features])
        feature_matrix_scaled = self.embedder.scaler.transform(feature_matrix)
        
        # LSH Detection
        lsh_start = time.perf_counter()
        embedding = self.embedder.quantizer.quantize_flow(features)
        is_anomaly, confidence = self.embedder.detector.predict_anomaly(embedding)
        self.results['lsh']['times'].append(time.perf_counter() - lsh_start)
        self.results['lsh']['y_true'].append(true_label)
        self.results['lsh']['y_pred'].append(1 if is_anomaly else 0)
        self.results['lsh']['scores'].append(confidence)
        
        # Isolation Forest
        if_start = time.perf_counter()
        if_pred = self.embedder.isolation_forest.predict(feature_matrix_scaled)[0]
        if_score = self.embedder.isolation_forest.score_samples(feature_matrix_scaled)[0]
        self.results['iforest']['times'].append(time.perf_counter() - if_start)
        self.results['iforest']['y_true'].append(true_label)
        self.results['iforest']['y_pred'].append(1 if if_pred == -1 else 0)
        self.results['iforest']['scores'].append(-if_score)
        
        # One-Class SVM
        svm_start = time.perf_counter()
        svm_pred = self.embedder.one_class_svm.predict(feature_matrix_scaled)[0]
        svm_score = self.embedder.one_class_svm.score_samples(feature_matrix_scaled)[0]
        self.results['ocsvm']['times'].append(time.perf_counter() - svm_start)
        self.results['ocsvm']['y_true'].append(true_label)
        self.results['ocsvm']['y_pred'].append(1 if svm_pred == -1 else 0)
        self.results['ocsvm']['scores'].append(-svm_score)
        
        return time.perf_counter() - start_time

    def run_benchmark(self, test_file=TEST_FILE, speed_factor=1.0):
        """Run complete benchmark on test data"""
        print(f"\nStarting benchmark on {test_file}...")
        streamer = CSVStreamer(test_file, speed_factor=speed_factor)
        monitor = ResourceMonitor()
        processing_times = []
        
        monitor.start_monitoring()
        try:
            while True:
                flow = streamer.get_next_flow()
                if flow is None:
                    break
                    
                proc_time = self.process_flow(flow)
                processing_times.append(proc_time)
                
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        
        test_metrics = monitor.stop_monitoring()
        self.save_results()
        self.print_results(len(processing_times), test_metrics)
        
        return {
            'model_results': self.results,
            'system_metrics': {
                'processing_times': processing_times,
                'resource_usage': test_metrics
            }
        }

    def save_results(self):
        """Save results to CSV files"""
        for model_name, data in self.results.items():
            df = pd.DataFrame({
                'true_label': data['y_true'],
                'prediction': data['y_pred'],
                'score': data['scores'],
                'processing_time_ms': [t*1000 for t in data['times']]
            })
            df.to_csv(f"{RESULTS_DIR}/{model_name}_results.csv", index=False)

    def print_results(self, num_flows, test_metrics):
        """Print comprehensive benchmark results"""
        print("\n=== Benchmark Results ===")
        print(f"Total flows processed: {num_flows}")
        print(f"Total time: {test_metrics['execution_time_seconds']:.2f}s")
        print(f"Throughput: {num_flows/test_metrics['execution_time_seconds']:.1f} flows/sec\n")
        
        for model_name, data in self.results.items():
            print(f"\nModel: {model_name.upper()}")
            print(f"Avg processing time: {np.mean(data['times'])*1000:.2f}ms")
            print(classification_report(data['y_true'], data['y_pred'], 
                                     target_names=['Normal', 'Anomaly']))
            print(f"ROC AUC: {roc_auc_score(data['y_true'], data['scores']):.4f}")

def setup_ovs():
    """Initialize Open vSwitch environment"""
    print("Setting up Open vSwitch...")
    try:
        subprocess.run(["sudo", "ovs-vsctl", "add-br", "ovs-br0"], check=True)
        print("Created OVS bridge 'ovs-br0'")
    except subprocess.CalledProcessError as e:
        print(f"OVS setup error: {e}")

def main():
    print("=== OVS Anomaly Detection Benchmark ===")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Test file: {TEST_FILE}")
    
    # Initialize OVS (comment out if not needed)
    setup_ovs()
    
    # Run benchmark
    benchmark = AnomalyDetectorBenchmark(embedding_size=EMB_SIZE)
    
    if not benchmark.train():
        print("Failed to train models. Exiting.")
        return
    
    benchmark.run_benchmark(speed_factor=1.0)  # Set speed_factor >1 for faster playback

if __name__ == "__main__":
    main()
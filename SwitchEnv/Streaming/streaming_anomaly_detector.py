#!/usr/bin/env python3
"""
Streaming Network Flow Anomaly Detection System
Processes flows in real-time through a simulated switch interface
"""

import numpy as np
import pandas as pd
import time
import os
import queue
import threading
import json
from collections import defaultdict, deque
from sklearn.metrics import (classification_report, roc_auc_score, 
                           confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import psutil
import sys
import random
import gc
from datetime import datetime
import signal
import argparse

# Environment variables with defaults
EMB_SIZE = int(os.environ.get('EMBEDDING_SIZE', '64'))
STREAM_SAMPLES = int(os.environ.get('STREAM_SAMPLES', '1000'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', 'streaming-results')
TRAIN_FILE = os.environ.get('TRAIN_FILE', '/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_training-set.csv')
TEST_FILE = os.environ.get('TEST_FILE', '/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_testing-set.csv')
SLEEP_INTERVAL = float(os.environ.get('SLEEP_INTERVAL', '0.01'))  # seconds between flows

from anomaly_detection import FlowAwareLSHAnomalyDetector, StreamingEmbedder
from flow_processing import FlowFeatureExtractor, FlowAwareBinaryQuantizer
from resource_metrics import ResourceMonitor

class StreamingAnomalyDetector:
    def __init__(self, embedding_size=EMB_SIZE):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.embedding_size = embedding_size
        self.embedder = None
        self.resource_monitor = ResourceMonitor()
        
        # Streaming metrics
        self.flow_queue = queue.Queue(maxsize=1000)
        self.results_queue = queue.Queue()
        self.processed_flows = 0
        self.streaming_active = False
        self.start_time = None
        
        # Performance tracking
        self.predictions = []
        self.true_labels = []
        self.detection_scores = []
        self.processing_times = []
        self.memory_usage = []
        
        # Model results storage
        self.model_results = defaultdict(lambda: defaultdict(list))
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        print(f"\nReceived signal {sig}. Shutting down gracefully...")
        self.streaming_active = False
        self.generate_final_report()
        sys.exit(0)

    def load_and_prepare_data(self, seed=42):
        print("Loading datasets...")
        try:
            train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
            test_df = pd.read_csv(TEST_FILE, low_memory=False)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

        # Convert labels
        def fix_labels(df):
            if 'label' in df.columns:
                df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
            return df

        train_df = fix_labels(train_df)
        test_df = fix_labels(test_df)

        # Sample training data (normal flows only)
        normal_train = train_df[train_df['label'] == 0].sample(
            n=min(3000, len(train_df[train_df['label'] == 0])),
            random_state=seed
        )

        # Prepare streaming test data
        test_sample = test_df.sample(n=min(STREAM_SAMPLES, len(test_df)), random_state=seed)
        
        print(f"Training on {len(normal_train)} normal flows")
        print(f"Streaming {len(test_sample)} test flows")
        
        return normal_train.to_dict('records'), test_sample

    def train_models(self, train_flows, seed=42):
        print("Training anomaly detection models...")
        
        self.embedder = StreamingEmbedder(embedding_size=self.embedding_size)
        
        # Monitor training resources
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        self.embedder.train(train_flows, seed)
        
        train_time = time.time() - start_time
        train_resources = self.resource_monitor.stop_monitoring()
        
        model_sizes = self.embedder.measure_model_sizes()
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Model sizes: {model_sizes}")
        
        return {
            'train_time': train_time,
            'train_resources': train_resources,
            'model_sizes': model_sizes
        }

    def flow_producer(self, test_df): #Producer thread that feeds flows to the queue
        print("Starting flow producer...")
        
        for _, flow in test_df.iterrows():
            if not self.streaming_active:
                break
                
            flow_dict = flow.to_dict()
            try:
                self.flow_queue.put(flow_dict, timeout=1)
                time.sleep(SLEEP_INTERVAL)  # Simulate network delay
            except queue.Full:
                print("Warning: Flow queue full, dropping flow")
                continue
        
        print("Flow producer finished")

    def flow_processor(self): #Consumer thread that processes flows and detects anomalies
        print("Starting flow processor...")
        
        batch_flows = []
        batch_labels = []
        
        while self.streaming_active:
            try:
                flow = self.flow_queue.get(timeout=1)
                batch_flows.append(flow)
                batch_labels.append(flow.get('label', 0))
                
                # Process in batches for efficiency
                if len(batch_flows) >= BATCH_SIZE:
                    self.process_batch(batch_flows, batch_labels)
                    batch_flows = []
                    batch_labels = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing flow: {e}")
                continue
        
        # Process remaining flows
        if batch_flows:
            self.process_batch(batch_flows, batch_labels)
        
        print("Flow processor finished")

    def process_batch(self, flows, labels):
        """Process a batch of flows"""
        start_time = time.time()
        
        # Get predictions from all models
        try:
            all_results = self.embedder.predict(flows)
            
            batch_time = time.time() - start_time
            current_memory = psutil.Process().memory_info().rss / 1024  # KB
            
            # Store results for each model
            for model_name, (predictions, scores) in all_results.items():
                self.model_results[model_name]['predictions'].extend(predictions)
                self.model_results[model_name]['scores'].extend(scores)
                self.model_results[model_name]['labels'].extend(labels)
                self.model_results[model_name]['processing_times'].append(batch_time)
                self.model_results[model_name]['memory_usage'].append(current_memory)
            
            self.processed_flows += len(flows)
            
            # Print progress
            if self.processed_flows % (BATCH_SIZE * 10) == 0:
                elapsed = time.time() - self.start_time
                flows_per_sec = self.processed_flows / elapsed if elapsed > 0 else 0
                print(f"Processed {self.processed_flows} flows - {flows_per_sec:.2f} flows/sec")
                
        except Exception as e:
            print(f"Error processing batch: {e}")

    def start_streaming(self, test_df):
        print(f"Starting streaming detection for {len(test_df)} flows...")
        print(f"Batch size: {BATCH_SIZE}, Sleep interval: {SLEEP_INTERVAL}s")
        
        self.streaming_active = True
        self.start_time = time.time()
        
        # Start producer and consumer threads
        producer_thread = threading.Thread(target=self.flow_producer, args=(test_df,))
        processor_thread = threading.Thread(target=self.flow_processor)
        
        producer_thread.start()
        processor_thread.start()
        
        # Monitor streaming
        self.monitor_streaming()
        
        # Wait for threads to complete
        producer_thread.join()
        self.streaming_active = False
        processor_thread.join()
        
        print("Streaming detection completed")

    def monitor_streaming(self):
        while self.streaming_active and self.processed_flows < STREAM_SAMPLES:
            time.sleep(5)  # Check every 5 seconds
            
            if self.processed_flows >= STREAM_SAMPLES:
                print(f"Reached target of {STREAM_SAMPLES} samples")
                self.streaming_active = False
                break
                
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                rate = self.processed_flows / elapsed
                eta = (STREAM_SAMPLES - self.processed_flows) / rate if rate > 0 else 0
                print(f"Progress: {self.processed_flows}/{STREAM_SAMPLES} flows "
                      f"({rate:.2f} flows/sec, ETA: {eta:.1f}s)")

    def calculate_metrics(self, model_name, predictions, scores, labels):
        y_pred = [1 if pred else 0 for pred in predictions]
        y_true = labels
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else 0,
            'total_samples': len(y_true),
            'anomalies_detected': sum(y_pred),
            'true_anomalies': sum(y_true)
        }

    def generate_final_report(self):
        print("\n" + "="*60)
        print("STREAMING ANOMALY DETECTION RESULTS")
        print("="*60)
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total flows processed: {self.processed_flows}")
        print(f"Average processing rate: {self.processed_flows/total_time:.2f} flows/sec")
        
        # Calculate metrics for each model
        model_metrics = {}
        for model_name, results in self.model_results.items():
            if results['predictions']:
                metrics = self.calculate_metrics(
                    model_name, 
                    results['predictions'], 
                    results['scores'], 
                    results['labels']
                )
                model_metrics[model_name] = metrics
                
                print(f"\n{model_name} Performance:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"  Anomalies detected: {metrics['anomalies_detected']}/{metrics['total_samples']}")
                
                # Calculate average processing time
                avg_batch_time = np.mean(results['processing_times'])
                avg_flow_time = avg_batch_time / BATCH_SIZE * 1000  # ms per flow
                print(f"  Avg processing time: {avg_flow_time:.2f} ms per flow")
                
                # Memory usage
                avg_memory = np.mean(results['memory_usage'])
                print(f"  Avg memory usage: {avg_memory:.2f} KB")
        
        # Save results to files
        self.save_results(model_metrics, total_time)
        
        return model_metrics

    def save_results(self, model_metrics, total_time):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model performance results
        performance_data = []
        for model_name, metrics in model_metrics.items():
            performance_data.append({
                'Model': model_name,
                'Timestamp': timestamp,
                'Total_Time': total_time,
                'Flows_Processed': self.processed_flows,
                'Flows_Per_Second': self.processed_flows / total_time if total_time > 0 else 0,
                **metrics
            })
        
        pd.DataFrame(performance_data).to_csv(
            f"{RESULTS_DIR}/streaming_performance_{timestamp}.csv", 
            index=False
        )
        
        # Detailed results for each model
        for model_name, results in self.model_results.items():
            if results['predictions']:
                detailed_data = {
                    'Prediction': [1 if pred else 0 for pred in results['predictions']],
                    'Score': results['scores'],
                    'True_Label': results['labels'],
                    'Correct': [pred == label for pred, label in zip(
                        [1 if p else 0 for p in results['predictions']], 
                        results['labels']
                    )]
                }
                
                pd.DataFrame(detailed_data).to_csv(
                    f"{RESULTS_DIR}/streaming_detailed_{model_name}_{timestamp}.csv",
                    index=False
                )
        
        print(f"\nResults saved to {RESULTS_DIR}/")

    def run_streaming_detection(self, seed=42):
        print("="*60)
        print("STREAMING NETWORK FLOW ANOMALY DETECTION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Stream samples: {STREAM_SAMPLES}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Embedding size: {EMB_SIZE}")
        print(f"  Sleep interval: {SLEEP_INTERVAL}s")
        print(f"  Results directory: {RESULTS_DIR}")
        
        # Load and prepare data
        train_flows, test_df = self.load_and_prepare_data(seed)
        if train_flows is None or test_df is None:
            return
        
        # Train models
        training_info = self.train_models(train_flows, seed)
        
        # Start streaming detection
        self.start_streaming(test_df)
        
        # Generate final report
        final_metrics = self.generate_final_report()
        
        return final_metrics

def main():
    parser = argparse.ArgumentParser(description='Streaming Network Flow Anomaly Detection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--samples', type=int, help='Number of samples to process (overrides env var)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides env var)')
    parser.add_argument('--embedding-size', type=int, help='Embedding size (overrides env var)')
    
    args = parser.parse_args()
    
    # Override environment variables if provided
    if args.samples:
        global STREAM_SAMPLES
        STREAM_SAMPLES = args.samples
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
    if args.embedding_size:
        global EMB_SIZE
        EMB_SIZE = args.embedding_size
    
    # Create and run detector
    detector = StreamingAnomalyDetector(embedding_size=EMB_SIZE)
    detector.run_streaming_detection(seed=args.seed)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
OVS Streaming Network Flow Anomaly Detection System
Integrates with Open vSwitch to process flows in real-time
"""

import numpy as np
import pandas as pd
import time
import os
import queue
import threading
import json
import subprocess
import socket
import struct
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

# OVS Configuration
OVS_BRIDGE = os.environ.get('OVS_BRIDGE', 'br0')
OVS_CONTROLLER_PORT = int(os.environ.get('OVS_CONTROLLER_PORT', '6653'))
NETFLOW_PORT = int(os.environ.get('NETFLOW_PORT', '9995'))

from anomaly_detection import FlowAwareLSHAnomalyDetector, StreamingEmbedder
from flow_processing import FlowFeatureExtractor, FlowAwareBinaryQuantizer
from resource_metrics import ResourceMonitor

class OVSFlowInjector:
    """Injects CSV flow data into OVS as synthetic flows"""
    
    def __init__(self, bridge_name=OVS_BRIDGE):
        self.bridge_name = bridge_name
        self.flow_table = {}
        self.verify_ovs_setup()
    
    def verify_ovs_setup(self):
        """Verify OVS bridge exists and is configured"""
        try:
            result = subprocess.run(['ovs-vsctl', 'br-exists', self.bridge_name], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Bridge {self.bridge_name} doesn't exist, creating it...")
                subprocess.run(['ovs-vsctl', 'add-br', self.bridge_name], check=True)
            
            # Configure NetFlow export
            self.configure_netflow()
            print(f"OVS bridge {self.bridge_name} is ready")
            
        except subprocess.CalledProcessError as e:
            print(f"Error setting up OVS: {e}")
            sys.exit(1)
    
    def configure_netflow(self):
        """Configure NetFlow export for flow monitoring"""
        try:
            # Remove existing NetFlow config
            subprocess.run(['ovs-vsctl', 'clear', 'bridge', self.bridge_name, 'netflow'], 
                          capture_output=True)
            
            # Create NetFlow configuration
            netflow_cmd = [
                'ovs-vsctl', 'set', 'bridge', self.bridge_name,
                f'netflow=@nf', '--',
                '--id=@nf', 'create', 'netflow',
                f'targets="127.0.0.1:{NETFLOW_PORT}"',
                'active-timeout=1',
                'add-id-to-interface=false'
            ]
            subprocess.run(netflow_cmd, check=True)
            print(f"NetFlow configured on port {NETFLOW_PORT}")
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not configure NetFlow: {e}")
    
    def create_flow_entry(self, flow_data):
        """Create OVS flow entry from CSV data"""
        try:
            # Extract key fields from CSV flow
            src_ip = flow_data.get('srcip', '192.168.1.1')
            dst_ip = flow_data.get('dstip', '192.168.1.2')
            src_port = flow_data.get('sport', 80)
            dst_port = flow_data.get('dsport', 443)
            protocol = flow_data.get('proto', 'tcp')
            
            # Create flow match criteria
            match_criteria = []
            if src_ip:
                match_criteria.append(f"ip_src={src_ip}")
            if dst_ip:
                match_criteria.append(f"ip_dst={dst_ip}")
            if protocol.lower() == 'tcp':
                match_criteria.append("ip_proto=6")
                if src_port:
                    match_criteria.append(f"tcp_src={src_port}")
                if dst_port:
                    match_criteria.append(f"tcp_dst={dst_port}")
            elif protocol.lower() == 'udp':
                match_criteria.append("ip_proto=17")
                if src_port:
                    match_criteria.append(f"udp_src={src_port}")
                if dst_port:
                    match_criteria.append(f"udp_dst={dst_port}")
            
            return ','.join(match_criteria)
            
        except Exception as e:
            print(f"Error creating flow entry: {e}")
            return None
    
    def inject_flow(self, flow_data):
        """Inject a single flow into OVS"""
        try:
            match_criteria = self.create_flow_entry(flow_data)
            if not match_criteria:
                return False
            
            # Create temporary flow with short timeout to simulate traffic
            flow_cmd = [
                'ovs-ofctl', 'add-flow', self.bridge_name,
                f"priority=100,{match_criteria},idle_timeout=2,actions=output:1"
            ]
            
            subprocess.run(flow_cmd, capture_output=True, text=True)
            
            # Simulate packet generation for the flow
            self.simulate_packet_flow(flow_data)
            
            return True
            
        except Exception as e:
            print(f"Error injecting flow: {e}")
            return False
    
    def simulate_packet_flow(self, flow_data):
        """Simulate packet flow to generate statistics"""
        try:
            # Get packet and byte counts from CSV
            packets = flow_data.get('spkts', 1) + flow_data.get('dpkts', 1)
            bytes_count = flow_data.get('sbytes', 64) + flow_data.get('dbytes', 64)
            
            # For simulation, we'll create a dummy flow entry that generates stats
            # In a real scenario, this would be actual network traffic
            
        except Exception as e:
            print(f"Error simulating packet flow: {e}")

class OVSFlowCollector:
    """Collects flows from OVS via NetFlow"""
    
    def __init__(self, port=NETFLOW_PORT):
        self.port = port
        self.socket = None
        self.running = False
        self.flow_queue = queue.Queue()
    
    def start_collector(self):
        """Start NetFlow collector"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('127.0.0.1', self.port))
            self.socket.settimeout(1.0)
            self.running = True
            print(f"NetFlow collector started on port {self.port}")
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(1024)
                    flow_data = self.parse_netflow_packet(data)
                    if flow_data:
                        self.flow_queue.put(flow_data)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error in NetFlow collector: {e}")
                    
        except Exception as e:
            print(f"Error starting NetFlow collector: {e}")
    
    def parse_netflow_packet(self, data):
        """Parse NetFlow packet (simplified)"""
        try:
            # This is a simplified NetFlow parser
            # In practice, you'd need a full NetFlow v5/v9/IPFIX parser
            return {
                'timestamp': time.time(),
                'src_ip': '192.168.1.1',
                'dst_ip': '192.168.1.2',
                'src_port': 80,
                'dst_port': 443,
                'protocol': 'tcp',
                'packets': 1,
                'bytes': 64
            }
        except Exception as e:
            print(f"Error parsing NetFlow packet: {e}")
            return None
    
    def stop_collector(self):
        """Stop NetFlow collector"""
        self.running = False
        if self.socket:
            self.socket.close()

class OVSStreamingAnomalyDetector:
    def __init__(self, embedding_size=EMB_SIZE):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.embedding_size = embedding_size
        self.embedder = None
        self.resource_monitor = ResourceMonitor()
        
        # OVS components
        self.ovs_injector = OVSFlowInjector()
        self.ovs_collector = OVSFlowCollector()
        
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
        self.ovs_collector.stop_collector()
        self.cleanup_ovs_flows()
        self.generate_final_report()
        sys.exit(0)

    def cleanup_ovs_flows(self):
        """Clean up OVS flows on shutdown"""
        try:
            subprocess.run(['ovs-ofctl', 'del-flows', OVS_BRIDGE], 
                          capture_output=True, text=True)
            print("OVS flows cleaned up")
        except Exception as e:
            print(f"Error cleaning up OVS flows: {e}")

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
        print(f"Streaming {len(test_sample)} test flows through OVS")
        
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

    def ovs_flow_injector_thread(self, test_df):
        """Thread that injects CSV flows into OVS"""
        print("Starting OVS flow injection...")
        
        for _, flow in test_df.iterrows():
            if not self.streaming_active:
                break
                
            flow_dict = flow.to_dict()
            
            # Inject flow into OVS
            if self.ovs_injector.inject_flow(flow_dict):
                # Also add to processing queue with original labels
                try:
                    self.flow_queue.put(flow_dict, timeout=1)
                except queue.Full:
                    print("Warning: Flow queue full, dropping flow")
                    continue
            
            time.sleep(SLEEP_INTERVAL)
        
        print("OVS flow injection finished")

    def ovs_flow_collector_thread(self):
        """Thread that collects flows from OVS via NetFlow"""
        print("Starting OVS flow collection...")
        
        # Start NetFlow collector in separate thread
        collector_thread = threading.Thread(target=self.ovs_collector.start_collector)
        collector_thread.daemon = True
        collector_thread.start()
        
        # Process collected flows
        while self.streaming_active:
            try:
                if not self.ovs_collector.flow_queue.empty():
                    ovs_flow = self.ovs_collector.flow_queue.get(timeout=1)
                    # Process OVS flow data here if needed
                    print(f"Collected flow from OVS: {ovs_flow}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error collecting OVS flow: {e}")
                continue
        
        print("OVS flow collection finished")

    def flow_processor(self):
        """Consumer thread that processes flows and detects anomalies"""
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
                print(f"Processed {self.processed_flows} flows via OVS - {flows_per_sec:.2f} flows/sec")
                
        except Exception as e:
            print(f"Error processing batch: {e}")

    def start_streaming(self, test_df):
        print(f"Starting OVS streaming detection for {len(test_df)} flows...")
        print(f"OVS Bridge: {OVS_BRIDGE}")
        print(f"NetFlow Port: {NETFLOW_PORT}")
        print(f"Batch size: {BATCH_SIZE}, Sleep interval: {SLEEP_INTERVAL}s")
        
        self.streaming_active = True
        self.start_time = time.time()
        
        # Start all threads
        injector_thread = threading.Thread(target=self.ovs_flow_injector_thread, args=(test_df,))
        collector_thread = threading.Thread(target=self.ovs_flow_collector_thread)
        processor_thread = threading.Thread(target=self.flow_processor)
        
        injector_thread.start()
        collector_thread.start()
        processor_thread.start()
        
        # Monitor streaming
        self.monitor_streaming()
        
        # Wait for threads to complete
        injector_thread.join()
        self.streaming_active = False
        collector_thread.join()
        processor_thread.join()
        
        # Clean up
        self.cleanup_ovs_flows()
        
        print("OVS streaming detection completed")

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
                print(f"OVS Progress: {self.processed_flows}/{STREAM_SAMPLES} flows "
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
        print("OVS STREAMING ANOMALY DETECTION RESULTS")
        print("="*60)
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total flows processed: {self.processed_flows}")
        print(f"Average processing rate: {self.processed_flows/total_time:.2f} flows/sec")
        print(f"OVS Bridge used: {OVS_BRIDGE}")
        
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
                'OVS_Bridge': OVS_BRIDGE,
                **metrics
            })
        
        pd.DataFrame(performance_data).to_csv(
            f"{RESULTS_DIR}/ovs_streaming_performance_{timestamp}.csv", 
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
                    f"{RESULTS_DIR}/ovs_streaming_detailed_{model_name}_{timestamp}.csv",
                    index=False
                )
        
        print(f"\nResults saved to {RESULTS_DIR}/")

    def run_streaming_detection(self, seed=42):
        print("="*60)
        print("OVS STREAMING NETWORK FLOW ANOMALY DETECTION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Stream samples: {STREAM_SAMPLES}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Embedding size: {EMB_SIZE}")
        print(f"  Sleep interval: {SLEEP_INTERVAL}s")
        print(f"  OVS Bridge: {OVS_BRIDGE}")
        print(f"  NetFlow Port: {NETFLOW_PORT}")
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
    parser = argparse.ArgumentParser(description='OVS Streaming Network Flow Anomaly Detection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--samples', type=int, help='Number of samples to process (overrides env var)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides env var)')
    parser.add_argument('--embedding-size', type=int, help='Embedding size (overrides env var)')
    parser.add_argument('--bridge', type=str, help='OVS bridge name (overrides env var)')
    parser.add_argument('--netflow-port', type=int, help='NetFlow port (overrides env var)')
    
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
    if args.bridge:
        global OVS_BRIDGE
        OVS_BRIDGE = args.bridge
    if args.netflow_port:
        global NETFLOW_PORT
        NETFLOW_PORT = args.netflow_port
    
    # Create and run detector
    detector = OVSStreamingAnomalyDetector(embedding_size=EMB_SIZE)
    detector.run_streaming_detection(seed=args.seed)

if __name__ == "__main__":
    main()
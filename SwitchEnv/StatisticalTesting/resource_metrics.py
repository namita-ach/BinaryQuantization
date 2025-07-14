import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import psutil
import resource
import sys
import gc
import tracemalloc
from memory_profiler import profile
import threading
import time
from collections import defaultdict, deque
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import random

from SwitchEnv.StatisticalTesting.anomaly_detection import StreamingEmbedder

EMB_SIZE = 64

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.baseline_cpu_percent = None
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous monitoring of resources"""
        self.monitoring = True
        self.baseline_memory = self.process.memory_info().rss / 1024  # KB
        self.baseline_cpu_percent = self.process.cpu_percent()
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start memory tracing
        tracemalloc.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return resource usage statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        # Get final measurements
        current_memory = self.process.memory_info().rss / 1024  # KB
        end_time = time.time()
        
        # Memory tracing results
        current_trace, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # CPU usage
        final_cpu = self.process.cpu_percent()
        
        # Calculate statistics
        memory_usage = current_memory - self.baseline_memory if self.baseline_memory else current_memory
        peak_memory_usage = max(self.memory_samples) - self.baseline_memory if self.memory_samples else memory_usage
        avg_memory_usage = np.mean(self.memory_samples) - self.baseline_memory if self.memory_samples else memory_usage
        
        avg_cpu_usage = np.mean(self.cpu_samples) if self.cpu_samples else final_cpu
        peak_cpu_usage = max(self.cpu_samples) if self.cpu_samples else final_cpu
        
        # System resource usage
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        
        return {
            'memory_usage_kb': memory_usage,
            'peak_memory_usage_kb': peak_memory_usage,
            'avg_memory_usage_kb': avg_memory_usage,
            'traced_memory_kb': current_trace / 1024,
            'peak_traced_memory_kb': peak_trace / 1024,
            'cpu_usage_percent': avg_cpu_usage,
            'peak_cpu_usage_percent': peak_cpu_usage,
            'execution_time_seconds': end_time - self.start_time,
            'user_cpu_time': rusage.ru_utime,
            'system_cpu_time': rusage.ru_stime,
            'max_rss_kb': rusage.ru_maxrss,  # Maximum resident set size
            'page_faults': rusage.ru_majflt,  # Major page faults
            'context_switches': rusage.ru_nvcsw + rusage.ru_nivcsw,  # Voluntary + involuntary
            'io_operations': rusage.ru_inblock + rusage.ru_oublock,  # Input + output blocks
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                mem_info = self.process.memory_info()
                self.memory_samples.append(mem_info.rss / 1024)  # KB
                self.cpu_samples.append(self.process.cpu_percent())
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
                
    def measure_model_size(self, model_obj):
        """Measure the memory footprint of a model object"""
        return sys.getsizeof(model_obj) / 1024  # KB
        
    def measure_flow_processing_time(self, flow_processor_func, flows, *args, **kwargs):
        """Measure time to process individual flows"""
        processing_times = []
        
        for flow in flows[:min(100, len(flows))]:  # Sample first 100 flows
            start_time = time.perf_counter()
            flow_processor_func(flow, *args, **kwargs)
            end_time = time.perf_counter()
            processing_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return {
            'avg_flow_processing_time_ms': np.mean(processing_times),
            'min_flow_processing_time_ms': np.min(processing_times),
            'max_flow_processing_time_ms': np.max(processing_times),
            'std_flow_processing_time_ms': np.std(processing_times),
        }

def run_single_experiment(seed: int = 42):
    monitor = ResourceMonitor()
    
    try:
        # Try to load real data
        train_df = pd.read_csv('/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_training-set.csv', low_memory=False)
        test_df = pd.read_csv('/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_testing-set.csv', low_memory=False)
        print(f"Loaded real UNSW-NB15 dataset (seed: {seed})")
    except:
        print("Cannot read data - please ensure UNSW-NB15 CSV files are in the current directory")
        return None

    # Convert labels properly
    def fix_labels(df):
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        return df

    train_df = fix_labels(train_df)
    test_df = fix_labels(test_df)

    # Sample data for single experiment
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
    embedder = StreamingEmbedder(embedding_size=EMB_SIZE)
    
    # Measure training resources
    monitor.start_monitoring()
    start_time = time.time()
    embedder.train(train_flows, seed)
    train_time = time.time() - start_time
    train_resources = monitor.stop_monitoring()
    
    # Measure model sizes after training
    model_sizes = embedder.measure_model_sizes()
    
    # Measure testing resources for each model
    testing_resources = {}
    
    # Test LSH model
    def test_lsh_flow(flow):
        features = embedder.feature_extractor.get_optimized_features(flow)
        embedding = embedder.quantizer.quantize_flow(features)
        return embedder.detector.predict_anomaly(embedding)
    
    # Test baseline models
    def test_baseline_flow(flow):
        features = embedder.feature_extractor.get_optimized_features(flow)
        feature_matrix = embedder._features_to_matrix([features])
        feature_matrix_scaled = embedder.scaler.transform(feature_matrix)
        
        # Test both baseline models
        if_pred = embedder.isolation_forest.predict(feature_matrix_scaled)
        svm_pred = embedder.one_class_svm.predict(feature_matrix_scaled)
        return if_pred[0], svm_pred[0]
    
    # Measure individual flow processing times
    lsh_flow_times = monitor.measure_flow_processing_time(test_lsh_flow, test_flows)
    baseline_flow_times = monitor.measure_flow_processing_time(test_baseline_flow, test_flows)
    
    # Full prediction with resource monitoring
    monitor.start_monitoring()
    start_time = time.time()
    all_results = embedder.predict(test_flows)
    test_time = time.time() - start_time
    test_resources = monitor.stop_monitoring()

    # Evaluate all models
    print(f"\nFixed Model Results (seed: {seed}):")
    print("="*60)
    
    results_with_resources = {}

    for model_name, (predictions, scores) in all_results.items():
        y_pred = [1 if pred else 0 for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, scores)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Store results with resource metrics
        results_with_resources[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'scores': scores
        }

    # Add resource utilization metrics
    resource_metrics = {
        'train_time_seconds': train_time,
        'test_time_seconds': test_time,
        'flows_processed': len(test_flows),
        'flows_per_second': len(test_flows) / test_time if test_time > 0 else 0,
        
        # Training resources
        'train_memory_usage_kb': train_resources['memory_usage_kb'],
        'train_peak_memory_kb': train_resources['peak_memory_usage_kb'],
        'train_cpu_usage_percent': train_resources['cpu_usage_percent'],
        'train_peak_cpu_percent': train_resources['peak_cpu_usage_percent'],
        
        # Testing resources
        'test_memory_usage_kb': test_resources['memory_usage_kb'],
        'test_peak_memory_kb': test_resources['peak_memory_usage_kb'],
        'test_cpu_usage_percent': test_resources['cpu_usage_percent'],
        'test_peak_cpu_percent': test_resources['peak_cpu_usage_percent'],
        
        # Individual flow processing times
        'lsh_avg_flow_time_ms': lsh_flow_times['avg_flow_processing_time_ms'],
        'lsh_max_flow_time_ms': lsh_flow_times['max_flow_processing_time_ms'],
        'baseline_avg_flow_time_ms': baseline_flow_times['avg_flow_processing_time_ms'],
        'baseline_max_flow_time_ms': baseline_flow_times['max_flow_processing_time_ms'],
        
        # Model sizes
        **model_sizes,
        
        # System resource usage
        'page_faults': test_resources['page_faults'],
        'context_switches': test_resources['context_switches'],
        'io_operations': test_resources['io_operations'],
        
        # Memory efficiency ratios
        'memory_efficiency_lsh_vs_if': model_sizes['total_lsh_model_kb'] / model_sizes['isolation_forest_kb'],
        'memory_efficiency_lsh_vs_svm': model_sizes['total_lsh_model_kb'] / model_sizes['one_class_svm_kb'],
        'speed_efficiency_lsh_vs_baseline': lsh_flow_times['avg_flow_processing_time_ms'] / baseline_flow_times['avg_flow_processing_time_ms'],
    }

    print("="*60)
    print(f"Resource Utilization Summary:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Testing time: {test_time:.2f} seconds")
    print(f"Flows per second: {resource_metrics['flows_per_second']:.1f}")
    print(f"LSH model size: {model_sizes['total_lsh_model_kb']:.1f} KB")
    print(f"Baseline models size: {model_sizes['total_baseline_models_kb']:.1f} KB")
    print(f"Memory efficiency (LSH vs baselines): {resource_metrics['memory_efficiency_lsh_vs_if']:.2f}x")
    print(f"Speed efficiency (LSH vs baselines): {resource_metrics['speed_efficiency_lsh_vs_baseline']:.2f}x")
    print("="*60)

    return results_with_resources, resource_metrics


def run_multiple_experiments(num_runs: int = 5, base_seed: int = 42):
    all_accuracy_scores = defaultdict(list)
    all_precision_scores = defaultdict(list)
    all_recall_scores = defaultdict(list)
    all_f1_scores = defaultdict(list)
    all_roc_scores = defaultdict(list)
    all_resource_metrics = defaultdict(list)

    # seeds = []
    # for i in range(20):
    #     seeds.append(random.randint(1, 10000))
    seeds = [355, 1307, 6390, 9026, 2997, 9766, 1095, 4926, 276, 8706]

    for i, seed in enumerate(seeds):
        print(f"\nRunning experiment {i+1}/{len(seeds)} with seed={seed}")
        result = run_single_experiment(seed)
        if result is None:
            continue  # Skip if the experiment failed
            
        results_with_resources, resource_metrics = result

        # Reconstruct ground truth
        test_df = pd.read_csv('/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_testing-set.csv', low_memory=False)
        test_df = test_df[test_df['label'].isin([0, 1])]
        normal_test = test_df[test_df['label'] == 0].sample(n=min(800, len(test_df[test_df['label'] == 0])), random_state=seed)
        anomaly_test = test_df[test_df['label'] == 1].sample(n=min(400, len(test_df[test_df['label'] == 1])), random_state=seed)
        test_combined = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=seed)
        y_true = test_combined['label'].tolist()

        # Collect performance metrics
        for model_name, model_results in results_with_resources.items():
            all_accuracy_scores[model_name].append(model_results['accuracy'])
            all_precision_scores[model_name].append(model_results['precision'])
            all_recall_scores[model_name].append(model_results['recall'])
            all_f1_scores[model_name].append(model_results['f1_score'])
            all_roc_scores[model_name].append(model_results['roc_auc'])
            
        # Collect resource metrics
        for metric_name, value in resource_metrics.items():
            all_resource_metrics[metric_name].append(value)

    # Summary
    print("\n=== Final Summary Across Runs ===")
    for model_name in all_accuracy_scores:
        mean_accuracy = np.mean(all_accuracy_scores[model_name])
        std_accuracy = np.std(all_accuracy_scores[model_name])
        mean_precision = np.mean(all_precision_scores[model_name])
        std_precision = np.std(all_precision_scores[model_name])
        mean_recall = np.mean(all_recall_scores[model_name])
        std_recall = np.std(all_recall_scores[model_name])
        mean_f1 = np.mean(all_f1_scores[model_name])
        std_f1 = np.std(all_f1_scores[model_name])
        mean_roc = np.mean(all_roc_scores[model_name])
        std_roc = np.std(all_roc_scores[model_name])

        print(f"\nModel: {model_name}")
        print(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        print(f"Average Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Average ROC AUC: {mean_roc:.4f} ± {std_roc:.4f}")
        
    # Print resource utilization summary
    print("\n=== Resource Utilization Summary ===")
    key_metrics = ['train_time_seconds', 'test_time_seconds', 'flows_per_second', 
                   'total_lsh_model_kb', 'total_baseline_models_kb', 
                   'memory_efficiency_lsh_vs_if', 'speed_efficiency_lsh_vs_baseline']
    
    for metric in key_metrics:
        if metric in all_resource_metrics:
            mean_val = np.mean(all_resource_metrics[metric])
            std_val = np.std(all_resource_metrics[metric])
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    # Create comprehensive results DataFrame
    results_data = []
    for model_name in all_accuracy_scores:
        results_data.append({
            'Model': model_name,
            'Accuracy_Mean': np.mean(all_accuracy_scores[model_name]),
            'Accuracy_Std': np.std(all_accuracy_scores[model_name]),
            'Precision_Mean': np.mean(all_precision_scores[model_name]),
            'Precision_Std': np.std(all_precision_scores[model_name]),
            'Recall_Mean': np.mean(all_recall_scores[model_name]),
            'Recall_Std': np.std(all_recall_scores[model_name]),
            'F1_Mean': np.mean(all_f1_scores[model_name]),
            'F1_Std': np.std(all_f1_scores[model_name]),
            'ROC_Mean': np.mean(all_roc_scores[model_name]),
            'ROC_Std': np.std(all_roc_scores[model_name]),
        })
    
    # Add resource metrics to the first row (they're experiment-wide)
    if results_data:
        for metric_name, values in all_resource_metrics.items():
            results_data[0][f'{metric_name}_Mean'] = np.mean(values)
            results_data[0][f'{metric_name}_Std'] = np.std(values)
    
    return pd.DataFrame(results_data)
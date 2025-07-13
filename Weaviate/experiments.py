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
import threading
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import weaviate
from weaviate.util import generate_uuid5
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        self.monitoring = True
        self.baseline_memory = self.process.memory_info().rss / 1024
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        tracemalloc.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        current_memory = self.process.memory_info().rss / 1024
        end_time = time.time()
        current_trace, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        final_cpu = self.process.cpu_percent()
        memory_usage = current_memory - (self.baseline_memory or 0)
        peak_memory_usage = max(self.memory_samples, default=current_memory) - (self.baseline_memory or 0)
        avg_memory_usage = np.mean(self.memory_samples) - (self.baseline_memory or 0) if self.memory_samples else memory_usage
        avg_cpu_usage = np.mean(self.cpu_samples) if self.cpu_samples else final_cpu
        peak_cpu_usage = max(self.cpu_samples, default=final_cpu)
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
            'max_rss_kb': rusage.ru_maxrss,
            'page_faults': rusage.ru_majflt,
            'context_switches': rusage.ru_nvcsw + rusage.ru_nivcsw,
            'io_operations': rusage.ru_inblock + rusage.ru_oublock,
        }

    def _monitor_loop(self):
        while self.monitoring:
            try:
                mem = self.process.memory_info().rss / 1024
                cpu = self.process.cpu_percent(interval=None)
                self.memory_samples.append(mem)
                self.cpu_samples.append(cpu)
                time.sleep(0.1)
            except Exception:
                break


class WeaviateNetworkFlowFeatureExtractor:
    def __init__(self):
        self.numerical_features = ["dur", "sbytes", "dbytes", "spkts", "dpkts", "rate", "sttl", "dttl"]
        self.categorical_features = ["proto", "service", "state"]
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_features(self, flow):
        f = {}
        for feat in self.numerical_features:
            try:
                f[feat] = float(flow.get(feat, 0) or 0)
            except:
                f[feat] = 0.0
        for feat in self.categorical_features:
            f[feat] = str(flow.get(feat) or "")
        f['flow_id'] = flow.get('id') or generate_uuid5(json.dumps(flow))
        f['label'] = int(flow.get('label') or 0)
        f['is_anomaly'] = bool(f['label'] == 1)
        return f

    def get_feature_vector(self, flow):
        return np.array([float(flow.get(f, 0) or 0) for f in self.numerical_features])

    def fit_scaler(self, flows):
        X = np.stack([self.get_feature_vector(f) for f in flows])
        self.scaler.fit(X)
        self.is_fitted = True

    def transform_features(self, flows):
        if not self.is_fitted:
            raise ValueError("Scaler not fitted!")
        X = np.stack([self.get_feature_vector(f) for f in flows])
        return self.scaler.transform(X)


class WeaviateAnomalyDetector:
    def __init__(self, weaviate_url="http://localhost:8080", collection_name="NetworkFlow"):
        self.client = None
        self.weaviate_url = weaviate_url
        self.collection_name = collection_name
        self.feature_extractor = WeaviateNetworkFlowFeatureExtractor()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = OneClassSVM(nu=0.1)
        self.local_outlier_factor = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        self.pca = PCA(n_components=5)
        self.is_trained = False

    def connect(self):
        self.client = weaviate.Client(url=self.weaviate_url)
        return self.client.is_ready()

    def create_collection(self):
        if self.client.schema.exists(self.collection_name):
            self.client.schema.delete_class(self.collection_name)
        class_def = {
            "class": self.collection_name,
            "properties": [
                *[{"name": f, "dataType": ["number"]} for f in self.feature_extractor.numerical_features],
                *[{"name": f, "dataType": ["string"]} for f in self.feature_extractor.categorical_features],
                {"name": "flow_id", "dataType": ["string"]},
                {"name": "label", "dataType": ["int"]},
                {"name": "is_anomaly", "dataType": ["boolean"]},
                {"name": "flow_description", "dataType": ["string"]},
                {"name": "anomaly_score", "dataType": ["number"]},
                {"name": "training_set", "dataType": ["boolean"]},
            ],
            "vectorizer": "none"
        }
        self.client.schema.create_class(class_def)

    def train(self, flows):
        self.feature_extractor.fit_scaler(flows)
        X = self.feature_extractor.transform_features(flows)
        X_pca = self.pca.fit_transform(X)
        self.isolation_forest.fit(X_pca)
        self.one_class_svm.fit(X_pca)
        with self.client.batch as batch:
            for f in flows:
                obj = self.feature_extractor.extract_features(f)
                obj["flow_description"] = str(obj)
                obj["anomaly_score"] = 0.0
                obj["training_set"] = True
                batch.add_data_object(obj, self.collection_name)
        self.is_trained = True

    def predict(self, flows):
        X = self.feature_extractor.transform_features(flows)
        X_pca = self.pca.transform(X)
        results = {}
        results['isolation_forest'] = ([x == -1 for x in self.isolation_forest.predict(X_pca)], -self.isolation_forest.score_samples(X_pca))
        results['one_class_svm'] = ([x == -1 for x in self.one_class_svm.predict(X_pca)], -self.one_class_svm.score_samples(X_pca))
        results['local_outlier_factor'] = ([x == -1 for x in self.local_outlier_factor.fit_predict(X_pca)], -self.local_outlier_factor.negative_outlier_factor_)
        return results

    def measure_model_sizes(self):
        return {
            'if_kb': sys.getsizeof(self.isolation_forest) / 1024,
            'svm_kb': sys.getsizeof(self.one_class_svm) / 1024,
            'lof_kb': sys.getsizeof(self.local_outlier_factor) / 1024,
            'pca_kb': sys.getsizeof(self.pca) / 1024
        }


def run_single_weaviate_experiment(seed=42):
    mon = ResourceMonitor()
    df_train = pd.read_csv('UNSW_NB15_training-set.csv')
    df_test = pd.read_csv('UNSW_NB15_testing-set.csv')
    for df in [df_train, df_test]:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    normal_train = df_train[df_train['label'] == 0].sample(n=1000, random_state=seed)
    normal_test = df_test[df_test['label'] == 0].sample(n=400, random_state=seed)
    anomaly_test = df_test[df_test['label'] == 1].sample(n=200, random_state=seed)
    test_comb = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=seed)

    y_true = test_comb['label'].tolist()
    train_flows = normal_train.to_dict('records')
    test_flows = test_comb.to_dict('records')

    model = WeaviateAnomalyDetector()
    if not model.connect():
        raise RuntimeError("Weaviate not ready")
    model.create_collection()

    mon.start_monitoring()
    model.train(train_flows)
    train_res = mon.stop_monitoring()

    mon.start_monitoring()
    preds = model.predict(test_flows)
    test_res = mon.stop_monitoring()

    def test_flow(flow):
        x = model.feature_extractor.get_feature_vector(flow).reshape(1, -1)
        x = model.feature_extractor.scaler.transform(x)
        x = model.pca.transform(x)
        model.isolation_forest.predict(x)

    def measure_flow_time():
        times = []
        for flow in test_flows[:100]:
            start = time.perf_counter()
            test_flow(flow)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return {
            'avg_flow_processing_time_ms': np.mean(times),
            'min_flow_processing_time_ms': np.min(times),
            'max_flow_processing_time_ms': np.max(times),
            'std_flow_processing_time_ms': np.std(times),
        }

    flow_time_stats = measure_flow_time()

    results = []
    for name, (y_pred_raw, y_score) in preds.items():
        y_pred = [1 if x else 0 for x in y_pred_raw]
        row = {
            'model': name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_score),
            'conf_matrix': str(confusion_matrix(y_true, y_pred))
        }
        row.update(train_res)
        row.update(test_res)
        row.update(flow_time_stats)
        row.update(model.measure_model_sizes())
        row['train_time_seconds'] = train_res['execution_time_seconds']
        row['test_time_seconds'] = test_res['execution_time_seconds']
        row['flows_processed'] = len(test_flows)
        row['flows_per_second'] = len(test_flows) / test_res['execution_time_seconds']
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv('weaviate_anomaly_results.csv', index=False)
    print("Saved results to weaviate_anomaly_results.csv")


if __name__ == '__main__':
    run_single_weaviate_experiment()

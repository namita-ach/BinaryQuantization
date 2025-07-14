import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import sys

from flow_processing import FlowFeatureExtractor, FlowAwareBinaryQuantizer



EMB_SIZE = 64


class FlowAwareLSHAnomalyDetector:
    def __init__(self, embedding_size: int = 128, num_hash_functions: int = 20, num_tables: int = 16):
        self.embedding_size = embedding_size
        self.num_hash_functions = num_hash_functions
        self.num_tables = num_tables
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
        self.normal_embeddings = []
        self.hash_functions = []
        self.anomaly_threshold = None
        self.distance_stats = {}

    def _generate_hash_functions(self, seed):
        np.random.seed(seed)
        self.hash_functions = []

        for table_idx in range(self.num_tables):
            table_functions = []
            for func_idx in range(self.num_hash_functions):
                # Use different hash strategies for different tables
                if table_idx < self.num_tables // 3:
                    # Local hash functions (focus on nearby bits)
                    start_bit = np.random.randint(0, max(1, self.embedding_size - 25))
                    bit_positions = np.arange(start_bit, min(start_bit + 25, self.embedding_size))
                elif table_idx < 2 * self.num_tables // 3:
                    # Medium-range hash functions
                    start_bit = np.random.randint(0, max(1, self.embedding_size - 40))
                    bit_positions = np.arange(start_bit, min(start_bit + 40, self.embedding_size))
                else:
                    # Global hash functions (random bits across embedding)
                    bit_positions = np.random.choice(self.embedding_size,
                                                   min(self.embedding_size//2, 30),
                                                   replace=False)
                table_functions.append(bit_positions)
            self.hash_functions.append(table_functions)

    def _compute_lsh_hash(self, embedding: int, table_idx: int) -> str:
        hash_bits = []
        for func_idx in range(self.num_hash_functions):
            bit_positions = self.hash_functions[table_idx][func_idx]

            # Extract bits at specified positions and compute parity
            parity = 0
            for pos in bit_positions:
                if embedding & (1 << pos):
                    parity ^= 1
            hash_bits.append(str(parity))

        return ''.join(hash_bits)

    def fit(self, normal_embeddings: List[int], seed: int = 42):
        print(f"Building flow-aware LSH tables with {len(normal_embeddings)} embeddings...")

        self._generate_hash_functions(seed)
        self.normal_embeddings = normal_embeddings

        # Clear hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]

        # Build hash tables
        for embedding in normal_embeddings:
            for table_idx in range(self.num_tables):
                hash_key = self._compute_lsh_hash(embedding, table_idx)
                self.hash_tables[table_idx][hash_key].append(embedding)

        # Compute distance statistics for adaptive thresholding
        self._compute_distance_statistics()

        # Set adaptive threshold
        self._set_adaptive_threshold()

        # Print debugging info
        print(f"Distance stats: mean={self.distance_stats['mean']:.1f}, std={self.distance_stats['std']:.1f}")
        print(f"Percentiles: {[f'{p:.1f}' for p in self.distance_stats['percentiles']]}")

    def _compute_distance_statistics(self):
        if len(self.normal_embeddings) < 2:
            self.distance_stats = {'mean': 32, 'std': 8, 'percentiles': [16, 24, 32, 40, 48]}
            return

        # Sample embeddings for statistics
        sample_size = min(2000, len(self.normal_embeddings))
        sample_embeddings = np.random.choice(self.normal_embeddings, sample_size, replace=False)

        distances = []
        for i in range(min(1000, len(sample_embeddings))):
            for j in range(i+1, min(i+10, len(sample_embeddings))):
                dist = bin(sample_embeddings[i] ^ sample_embeddings[j]).count('1')
                distances.append(dist)

        if distances:
            distances = np.array(distances)
            self.distance_stats = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'percentiles': np.percentile(distances, [10, 25, 50, 75, 90])
            }
        else:
            self.distance_stats = {'mean': 32, 'std': 8, 'percentiles': [16, 24, 32, 40, 48]}

    def _set_adaptive_threshold(self):
        # Set threshold based on distance statistics
        # Use a more aggressive threshold to detect anomalies
        mean_dist = self.distance_stats['mean']
        std_dist = self.distance_stats['std']

        # Use mean - 0.5*std as threshold (more aggressive)
        self.anomaly_threshold = max(mean_dist - 2 * std_dist,
                                   self.distance_stats['percentiles'][1])  # 25th percentile as minimum

        print(f"Set anomaly threshold to: {self.anomaly_threshold:.1f}")

    def _hamming_distance(self, a: int, b: int) -> int:
        return bin(a ^ b).count('1')

    def predict_anomaly(self, embedding: int) -> Tuple[bool, float]:
        candidates = set()

        # Collect candidates from all tables
        for table_idx in range(self.num_tables):
            hash_key = self._compute_lsh_hash(embedding, table_idx)
            candidates.update(self.hash_tables[table_idx].get(hash_key, []))

        # Debugging: check candidate retrieval
        if len(candidates) == 0:
            # No exact hash matches - this is suspicious
            # Try to find approximate matches by checking nearby hashes
            for table_idx in range(min(3, self.num_tables)):  # Check first 3 tables
                hash_key = self._compute_lsh_hash(embedding, table_idx)
                # Check if any similar hash keys exist
                for existing_key in list(self.hash_tables[table_idx].keys())[:50]:  # Check first 50
                    # Simple similarity check - count different bits
                    if existing_key and hash_key:
                        diff_bits = sum(c1 != c2 for c1, c2 in zip(hash_key, existing_key))
                        if diff_bits <= 2:  # Allow 2 bit differences
                            candidates.update(self.hash_tables[table_idx][existing_key])

        if not candidates:
            # Still no candidates - definitely anomalous
            return True, 0.90

        # Find nearest neighbors
        k = min(20, len(candidates))
        distances = []

        for candidate in candidates:
            dist = self._hamming_distance(embedding, candidate)
            distances.append(dist)

        distances.sort()
        top_k_distances = distances[:k]

        # Compute anomaly score
        min_distance = min(top_k_distances)
        avg_distance = np.mean(top_k_distances)

        # Multi-criteria anomaly detection with more aggressive thresholds
        is_anomaly = False
        confidence = 0.0

        # Criterion 1: Minimum distance threshold (primary)
        if min_distance > self.anomaly_threshold:
            is_anomaly = True
            confidence = max(confidence, 0.7 + 0.3 * (min_distance / self.embedding_size))

        # Criterion 2: Average distance threshold (secondary)
        avg_threshold = self.anomaly_threshold * 1.1
        if avg_distance > avg_threshold:
            is_anomaly = True
            confidence = max(confidence, 0.6 + 0.4 * (avg_distance / self.embedding_size))

        # Criterion 3: Isolation check - if very few candidates found
        if len(candidates) < 5:
            is_anomaly = True
            confidence = max(confidence, 0.75)

        # Criterion 4: Check if embedding is significantly different from normal range
        if min_distance > self.distance_stats['percentiles'][3]:  # 75th percentile
            is_anomaly = True
            confidence = max(confidence, 0.65)

        # Ensure we have some minimum anomaly detection rate
        # If distance is above mean, there's some chance it's anomalous
        if min_distance > self.distance_stats['mean']:
            anomaly_prob = (min_distance - self.distance_stats['mean']) / self.distance_stats['std']
            if np.random.random() < min(anomaly_prob * 0.1, 0.3):  # 10% base rate, max 30%
                is_anomaly = True
                confidence = max(confidence, 0.55)

        # Normalize confidence score
        confidence = min(confidence, 1.0)

        return is_anomaly, confidence
    


class StreamingEmbedder:
    def __init__(self, embedding_size: int = EMB_SIZE):
        self.feature_extractor = FlowFeatureExtractor()
        self.quantizer = FlowAwareBinaryQuantizer(embedding_size)
        self.detector = FlowAwareLSHAnomalyDetector(embedding_size)

        # Baseline models for comparison
        self.isolation_forest = IsolationForest(contamination=0.15, n_estimators=100)
        self.one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.15)
        self.scaler = StandardScaler()

        self.is_trained = False

    def measure_model_sizes(self):
        """Measure memory footprint of different model components"""
        sizes = {}
        
        # LSH detector components
        sizes['lsh_hash_tables_kb'] = sys.getsizeof(self.detector.hash_tables) / 1024
        sizes['lsh_normal_embeddings_kb'] = sys.getsizeof(self.detector.normal_embeddings) / 1024
        sizes['lsh_hash_functions_kb'] = sys.getsizeof(self.detector.hash_functions) / 1024
        
        # Quantizer components
        sizes['quantizer_params_kb'] = sys.getsizeof(self.quantizer.quantization_params) / 1024
        
        # Feature extractor
        sizes['feature_extractor_kb'] = sys.getsizeof(self.feature_extractor) / 1024
        
        # Baseline models
        sizes['isolation_forest_kb'] = sys.getsizeof(self.isolation_forest) / 1024
        sizes['one_class_svm_kb'] = sys.getsizeof(self.one_class_svm) / 1024
        sizes['scaler_kb'] = sys.getsizeof(self.scaler) / 1024
        
        # Total model size
        sizes['total_lsh_model_kb'] = (sizes['lsh_hash_tables_kb'] + 
                                      sizes['lsh_normal_embeddings_kb'] + 
                                      sizes['lsh_hash_functions_kb'] + 
                                      sizes['quantizer_params_kb'] + 
                                      sizes['feature_extractor_kb'])
        
        sizes['total_baseline_models_kb'] = (sizes['isolation_forest_kb'] + 
                                            sizes['one_class_svm_kb'] + 
                                            sizes['scaler_kb'])
        
        return sizes

    def train(self, normal_flows: List[Dict], seed: int = 42):
        print(f"Training flow-aware system with {len(normal_flows)} normal flows...")

        # Set seeds for reproducibility
        np.random.seed(seed)
        self.isolation_forest.random_state = seed

        # Extract optimized features (removes harmful features)
        all_features = []
        for flow in normal_flows:
            features = self.feature_extractor.get_optimized_features(flow)
            all_features.append(features)

        # Fit quantizer
        self.quantizer.fit(all_features)

        # Generate embeddings
        normal_embeddings = []
        for features in all_features:
            embedding = self.quantizer.quantize_flow(features)
            normal_embeddings.append(embedding)

        # Debug: Check embedding diversity
        unique_embeddings = len(set(normal_embeddings))
        print(f"Generated {unique_embeddings} unique embeddings from {len(normal_embeddings)} flows")

        # Train LSH detector
        self.detector.fit(normal_embeddings, seed)

        # Prepare data for sklearn models
        feature_matrix = self._features_to_matrix(all_features)

        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        # Train baseline models
        print("Training baseline models...")
        self.isolation_forest.fit(feature_matrix_scaled)
        self.one_class_svm.fit(feature_matrix_scaled)

        self.is_trained = True
        print("Training completed successfully!")

    def _features_to_matrix(self, features_list: List[Dict]) -> np.ndarray:
        if not features_list:
            return np.array([])

        # Get all unique feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())

        feature_names = sorted(list(all_features))

        # Create matrix
        matrix = np.zeros((len(features_list), len(feature_names)))

        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(feature_names):
                matrix[i, j] = features.get(feature_name, 0)

        return matrix

    def predict(self, test_flows: List[Dict]) -> Dict[str, Tuple[List[bool], List[float]]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Extract optimized features
        all_features = []
        for flow in test_flows:
            features = self.feature_extractor.get_optimized_features(flow)
            all_features.append(features)

        results = {}

        # 1. Flow-aware LSH predictions
        lsh_predictions = []
        lsh_scores = []

        for features in all_features:
            embedding = self.quantizer.quantize_flow(features)
            is_anomaly, score = self.detector.predict_anomaly(embedding)
            lsh_predictions.append(is_anomaly)
            lsh_scores.append(score)

        results['Flow_Aware_LSH'] = (lsh_predictions, lsh_scores)

        # 2. Baseline models for comparison
        feature_matrix = self._features_to_matrix(all_features)
        feature_matrix_scaled = self.scaler.transform(feature_matrix)

        # Isolation Forest
        if_predictions = self.isolation_forest.predict(feature_matrix_scaled)
        if_scores = self.isolation_forest.score_samples(feature_matrix_scaled)
        if_predictions = [pred == -1 for pred in if_predictions]
        if_scores = [1 - (score + 0.5) for score in if_scores]
        results['Isolation_Forest'] = (if_predictions, if_scores)

        # One-Class SVM
        svm_predictions = self.one_class_svm.predict(feature_matrix_scaled)
        svm_scores = self.one_class_svm.score_samples(feature_matrix_scaled)
        svm_predictions = [pred == -1 for pred in svm_predictions]
        svm_scores = [1 - (score + 0.5) for score in svm_scores]
        results['One_Class_SVM'] = (svm_predictions, svm_scores)

        return results


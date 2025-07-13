import numpy as np
import pandas as pd
import hashlib
import struct
import time
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FlowFeatureExtractor:
    def __init__(self):
        self.feature_stats = {}
        self.service_encoder = {}
        self.state_encoder = {}

    def extract_comprehensive_features(self, flow_record: Dict) -> Dict:
        features = {}
        def safe_float(value, default=0.0):
            try:
                if value in ['-', '', None, 'nan']:
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default

        def safe_int(value, default=0):
            try:
                if value in ['-', '', None, 'nan']:
                    return default
                return int(float(value))
            except (ValueError, TypeError):
                return default

        # Basic flow features (these are the most important)
        features['duration'] = safe_float(flow_record.get('dur', 0))
        features['src_bytes'] = safe_int(flow_record.get('sbytes', 0))
        features['dst_bytes'] = safe_int(flow_record.get('dbytes', 0))
        features['src_packets'] = safe_int(flow_record.get('spkts', 0))
        features['dst_packets'] = safe_int(flow_record.get('dpkts', 0))

        # Protocol mapping
        proto = str(flow_record.get('proto', 'tcp')).lower()
        proto_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
        features['protocol'] = proto_map.get(proto, 1)

        # Service encoding
        service = str(flow_record.get('service', '-')).lower()
        if service not in self.service_encoder:
            self.service_encoder[service] = len(self.service_encoder) + 1
        features['service_encoded'] = self.service_encoder[service]

        # State encoding
        state = str(flow_record.get('state', 'other')).upper()
        if state not in self.state_encoder:
            self.state_encoder[state] = len(self.state_encoder) + 1
        features['state_encoded'] = self.state_encoder[state]

        # Derived features that are imporant for anomaly detection
        features['total_bytes'] = features['src_bytes'] + features['dst_bytes']
        features['total_packets'] = features['src_packets'] + features['dst_packets']

        # Avoid division by zero
        if features['total_packets'] > 0:
            features['bytes_per_packet'] = features['total_bytes'] / features['total_packets']
        else:
            features['bytes_per_packet'] = 0

        if features['duration'] > 0:
            features['bytes_per_second'] = features['total_bytes'] / features['duration']
            features['packets_per_second'] = features['total_packets'] / features['duration']
        else:
            features['bytes_per_second'] = 0
            features['packets_per_second'] = 0

        # Directional ratios (important for attack detection)
        if features['total_bytes'] > 0:
            features['src_bytes_ratio'] = features['src_bytes'] / features['total_bytes']
        else:
            features['src_bytes_ratio'] = 0.5

        if features['total_packets'] > 0:
            features['src_packets_ratio'] = features['src_packets'] / features['total_packets']
        else:
            features['src_packets_ratio'] = 0.5

        # Load features (these are often indicative of attacks)
        features['src_load'] = safe_float(flow_record.get('sload', 0))
        features['dst_load'] = safe_float(flow_record.get('dload', 0))
        features['load_ratio'] = 0
        if features['dst_load'] > 0:
            features['load_ratio'] = features['src_load'] / features['dst_load']

        # Additional behavioral features
        features['mean_packet_size'] = safe_float(flow_record.get('smean', 0))
        features['trans_depth'] = safe_int(flow_record.get('trans_depth', 0))

        # Network behavior indicators
        features['ct_src_dport_ltm'] = safe_int(flow_record.get('ct_src_dport_ltm', 0))
        features['ct_dst_sport_ltm'] = safe_int(flow_record.get('ct_dst_sport_ltm', 0))

        return features

class BinaryQuantizer:
    def __init__(self, embedding_size: int = 64):  # Increased size for better representation
        self.embedding_size = embedding_size
        self.quantization_params = {}
        self.is_fitted = False

    def fit(self, features_list: List[Dict]):
        print("Learning quantization parameters...")
        feature_collections = defaultdict(list)

        for features in features_list:
            for key, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
                    feature_collections[key].append(value)

        # Learn quantization parameters for each feature
        for feature_name, values in feature_collections.items():
            if len(values) < 10:  # Skip features with too few samples
                continue

            values = np.array(values)

            if feature_name in ['protocol', 'service_encoded', 'state_encoded']:
                unique_vals = np.unique(values)
                self.quantization_params[feature_name] = {
                    'type': 'categorical',
                    'values': unique_vals,
                    'bits': min(4, int(np.ceil(np.log2(len(unique_vals) + 1))))
                }
            else:
                # Continuous features- use robust percentile-based quantization
                # Remove outliers for better quantization
                q01, q99 = np.percentile(values, [1, 99])
                filtered_values = values[(values >= q01) & (values <= q99)]

                if len(filtered_values) > 0:
                    # Use more quantization levels for important features
                    if feature_name in ['total_bytes', 'total_packets', 'bytes_per_packet', 'duration']:
                        bits = 6  # 64 levels for important features
                    else:
                        bits = 4  # 16 levels for others

                    percentiles = np.linspace(0, 100, 2**bits)
                    thresholds = np.percentile(filtered_values, percentiles)

                    self.quantization_params[feature_name] = {
                        'type': 'continuous',
                        'thresholds': thresholds,
                        'bits': bits,
                        'min_val': q01,
                        'max_val': q99
                    }

        self.is_fitted = True
        print(f"Learned quantization for {len(self.quantization_params)} features")

    def _quantize_feature(self, value, feature_name):
        if feature_name not in self.quantization_params:
            return 0, 2  # Default: 0 value, 2 bits

        params = self.quantization_params[feature_name]

        if params['type'] == 'categorical':
            if value in params['values']:
                idx = np.where(params['values'] == value)[0][0]
                return idx, params['bits']
            else:
                return 0, params['bits']
        else:  # continuous
            # Clip to learned range
            clipped_value = np.clip(value, params['min_val'], params['max_val'])

            # Find quantization bin
            bin_idx = np.searchsorted(params['thresholds'], clipped_value)
            bin_idx = min(bin_idx, len(params['thresholds']) - 1)

            return bin_idx, params['bits']

    def quantize_flow(self, features: Dict) -> int:
        if not self.is_fitted:
            raise ValueError("Quantizer must be fitted before use")

        # Priority order for features (most important first)
        feature_priority = [
            'total_bytes', 'total_packets', 'duration', 'bytes_per_packet',
            'src_bytes_ratio', 'packets_per_second', 'protocol', 'service_encoded',
            'state_encoded', 'load_ratio', 'src_load', 'dst_load'
        ]

        binary_embedding = 0
        used_bits = 0

        for feature_name in feature_priority:
            if used_bits >= self.embedding_size:
                break

            if feature_name in features:
                quantized_value, bits_needed = self._quantize_feature(
                    features[feature_name], feature_name
                )

                if used_bits + bits_needed <= self.embedding_size:
                    # Shift existing bits and add new quantized value
                    binary_embedding = (binary_embedding << bits_needed) | quantized_value
                    used_bits += bits_needed

        return binary_embedding

    def get_quantization_analysis(self) -> Dict:
        analysis = {
            'num_features': len(self.quantization_params),
            'categorical_features': [],
            'continuous_features': [],
            'bits_per_feature': {},
            'feature_importance': {}
        }

        for feature_name, params in self.quantization_params.items():
            analysis['bits_per_feature'][feature_name] = params['bits']
            if params['type'] == 'categorical':
                analysis['categorical_features'].append(feature_name)
            else:
                analysis['continuous_features'].append(feature_name)

        return analysis

    def visualize_quantization_impact(self, features_list: List[Dict], embeddings: List[int]):
        if not self.is_fitted:
            return

        # Create feature importance analysis
        feature_names = list(self.quantization_params.keys())
        feature_matrix = np.zeros((len(features_list), len(feature_names)))

        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(feature_names):
                feature_matrix[i, j] = features.get(feature_name, 0)

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Feature distribution
        axes[0, 0].hist([len(self.quantization_params[f]['thresholds']) if f in self.quantization_params and self.quantization_params[f]['type'] == 'continuous' else len(self.quantization_params[f]['values']) if f in self.quantization_params else 0 for f in feature_names], bins=20)
        axes[0, 0].set_title('Quantization Levels Distribution')
        axes[0, 0].set_xlabel('Number of Quantization Levels')
        axes[0, 0].set_ylabel('Frequency')

        # 2. Bits allocation
        bits_data = [self.quantization_params[f]['bits'] for f in feature_names]
        axes[0, 1].bar(range(len(feature_names)), bits_data)
        axes[0, 1].set_title('Bits Allocated per Feature')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Bits')
        axes[0, 1].tick_params(axis='x', labelsize=8)

        # 3. Feature correlation with embeddings
        correlations = []
        for j in range(len(feature_names)):
            if np.std(feature_matrix[:, j]) > 0:
                corr = np.corrcoef(feature_matrix[:, j], embeddings)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        axes[0, 2].bar(range(len(feature_names)), correlations)
        axes[0, 2].set_title('Feature-Embedding Correlation')
        axes[0, 2].set_xlabel('Feature Index')
        axes[0, 2].set_ylabel('Absolute Correlation')
        axes[0, 2].tick_params(axis='x', labelsize=8)

        # 4. Embedding distribution
        axes[1, 0].hist(embeddings, bins=50, alpha=0.7)
        axes[1, 0].set_title('Binary Embedding Distribution')
        axes[1, 0].set_xlabel('Embedding Value')
        axes[1, 0].set_ylabel('Frequency')

        # 5. Feature importance heatmap
        if len(feature_names) > 1:
            feature_corr = np.corrcoef(feature_matrix.T)
            im = axes[1, 1].imshow(feature_corr, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=axes[1, 1])
        print("CORR MATRIX")
        print(feature_corr)

        # 6. Top features by bits
        top_indices = np.argsort(bits_data)[-10:]
        top_features = [feature_names[i] for i in top_indices]
        top_bits = [bits_data[i] for i in top_indices]

        axes[1, 2].barh(range(len(top_features)), top_bits)
        axes[1, 2].set_yticks(range(len(top_features)))
        axes[1, 2].set_yticklabels(top_features, fontsize=8)
        axes[1, 2].set_title('Top Features by Bits Allocated')
        axes[1, 2].set_xlabel('Bits')

        print("TOP FEATURES BY BIT")
        print(feature_names[i] for i in top_indices)

        plt.tight_layout()
        plt.savefig('Ablations/quantization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return correlations, feature_names

class LSHAnomalyDetector:
    def __init__(self, embedding_size: int = 64, num_hash_functions: int = 12, num_tables: int = 8):
        self.embedding_size = embedding_size
        self.num_hash_functions = num_hash_functions
        self.num_tables = num_tables
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
        self.normal_embeddings = []
        self.hash_functions = []

    def _generate_hash_functions(self, seed):
        # Generate random hash functions with given seed
        np.random.seed(seed)
        self.hash_functions = []
        for table_idx in range(self.num_tables):
            table_functions = []
            for func_idx in range(self.num_hash_functions):
                # Random bit positions for hash function
                bit_positions = np.random.choice(self.embedding_size,
                                               min(self.embedding_size//4, 16),
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
        print(f"Building LSH tables with {len(normal_embeddings)} embeddings...")

        # Generate hash functions with seed
        self._generate_hash_functions(seed)

        self.normal_embeddings = normal_embeddings

        # Clear hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]

        # Build hash tables
        for embedding in normal_embeddings:
            for table_idx in range(self.num_tables):
                hash_key = self._compute_lsh_hash(embedding, table_idx)
                self.hash_tables[table_idx][hash_key].append(embedding)

        # Compute statistics for adaptive thresholding
        self._compute_normal_statistics()

    def _compute_normal_statistics(self):
        if len(self.normal_embeddings) < 2:
            self.mean_distance = 16
            self.std_distance = 4
            return

        sample_size = min(1000, len(self.normal_embeddings))
        sample_embeddings = np.random.choice(self.normal_embeddings, sample_size, replace=False)

        distances = []
        for i in range(min(500, len(sample_embeddings))):
            for j in range(i+1, min(i+20, len(sample_embeddings))):
                dist = bin(sample_embeddings[i] ^ sample_embeddings[j]).count('1')
                distances.append(dist)

        if distances:
            self.mean_distance = np.mean(distances)
            self.std_distance = np.std(distances)
        else:
            self.mean_distance = 16
            self.std_distance = 4

    def _hamming_distance(self, a: int, b: int) -> int:
        return bin(a ^ b).count('1')

    def predict_anomaly(self, embedding: int) -> Tuple[bool, float]:
        candidates = set()

        # Collect candidates from all tables
        for table_idx in range(self.num_tables):
            hash_key = self._compute_lsh_hash(embedding, table_idx)
            candidates.update(self.hash_tables[table_idx].get(hash_key, []))

        if not candidates:
            # No similar embeddings found- likely anomalous
            return True, 1.0

        # Find k nearest neighbors in terms of Hamming distance
        k = min(10, len(candidates))
        distances = []

        for candidate in candidates:
            dist = self._hamming_distance(embedding, candidate)
            distances.append(dist)

        distances.sort()
        top_k_distances = distances[:k]

        # Compute anomaly score based on distance to knn
        avg_distance = np.mean(top_k_distances)

        # Adaptive threshold based on normal statistics
        threshold = self.mean_distance + 2 * self.std_distance

        # Normalize score
        normalized_score = min(avg_distance / (self.embedding_size * 0.75), 1.0)

        is_anomaly = avg_distance > threshold

        return is_anomaly, normalized_score

class StreamingEmbedder:
    def __init__(self, embedding_size: int = 64):
        self.feature_extractor = FlowFeatureExtractor()
        self.quantizer = BinaryQuantizer(embedding_size)
        self.detector = LSHAnomalyDetector(embedding_size)

        # Add new models
        self.isolation_forest = IsolationForest(contamination=0.1, n_estimators=100)
        self.one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.scaler = StandardScaler()

        self.is_trained = False

    def train(self, normal_flows: List[Dict], seed: int = 42):
        print(f"Training system with {len(normal_flows)} normal flows...")

        # Set seeds for reproducibility
        np.random.seed(seed)
        self.isolation_forest.random_state = seed

        # Extract features
        all_features = []
        for flow in normal_flows:
            features = self.feature_extractor.extract_comprehensive_features(flow)
            all_features.append(features)

        # Fit quantizer
        self.quantizer.fit(all_features)

        # Generate embeddings
        normal_embeddings = []
        for features in all_features:
            embedding = self.quantizer.quantize_flow(features)
            normal_embeddings.append(embedding)

        # Train LSH detector
        self.detector.fit(normal_embeddings, seed)

        # Prepare data for sklearn models (as a baseilne)
        feature_matrix = self._features_to_matrix(all_features)

        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        # Train Isolation Forest
        print("Training Isolation Forest...")
        self.isolation_forest.fit(feature_matrix_scaled)

        # Train One-Class SVM
        print("Training One-Class SVM...")
        self.one_class_svm.fit(feature_matrix_scaled)

        self.is_trained = True
        print("Training completed successfully!")

    def _features_to_matrix(self, features_list: List[Dict]) -> np.ndarray: # Convert list of feature dictionaries to numpy matrix
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

        # Extract features
        all_features = []
        for flow in test_flows:
            features = self.feature_extractor.extract_comprehensive_features(flow)
            all_features.append(features)

        # Prepare results dictionary
        results = {}

        # 1. Binary Embedding + LSH predictions
        lsh_predictions = []
        lsh_scores = []

        for features in all_features:
            embedding = self.quantizer.quantize_flow(features)
            is_anomaly, score = self.detector.predict_anomaly(embedding)
            lsh_predictions.append(is_anomaly)
            lsh_scores.append(score)

        results['Binary_LSH'] = (lsh_predictions, lsh_scores)

        # 2. Isolation Forest predictions
        feature_matrix = self._features_to_matrix(all_features)
        feature_matrix_scaled = self.scaler.transform(feature_matrix)

        if_predictions = self.isolation_forest.predict(feature_matrix_scaled)
        if_scores = self.isolation_forest.score_samples(feature_matrix_scaled)

        # Convert to boolean (True for anomaly) and normalize scores
        if_predictions = [pred == -1 for pred in if_predictions]
        if_scores = [1 - (score + 0.5) for score in if_scores]  # Normalize to [0, 1]

        results['Isolation_Forest'] = (if_predictions, if_scores)

        # 3. One-Class SVM predictions
        svm_predictions = self.one_class_svm.predict(feature_matrix_scaled)
        svm_scores = self.one_class_svm.score_samples(feature_matrix_scaled)

        # Convert to boolean (True for anomaly) and normalize scores
        svm_predictions = [pred == -1 for pred in svm_predictions]
        svm_scores = [1 - (score + 0.5) for score in svm_scores]  # Normalize to [0, 1]

        results['One_Class_SVM'] = (svm_predictions, svm_scores)

        return results


    def get_feature_analysis(self, flows: List[Dict]) -> Dict:
        """Analyze features extracted from flows"""
        all_features = []
        for flow in flows:
            features = self.feature_extractor.extract_comprehensive_features(flow)
            all_features.append(features)

        # Get feature statistics
        feature_stats = {}
        if all_features:
            feature_names = all_features[0].keys()
            for feature_name in feature_names:
                values = [f.get(feature_name, 0) for f in all_features]
                feature_stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'zero_ratio': sum(1 for v in values if v == 0) / len(values)
                }

        return feature_stats, all_features



    def run_feature_ablation(self, normal_flows: List[Dict], test_flows: List[Dict], y_true: List[int]):
        """Run ablation study by removing features one by one"""
        print("Running feature ablation study...")

        # Get baseline performance
        baseline_results = self.predict(test_flows)
        baseline_f1 = f1_score(y_true, [1 if pred else 0 for pred in baseline_results['Binary_LSH'][0]])

        # Get all features
        sample_features = self.feature_extractor.extract_comprehensive_features(normal_flows[0])
        all_feature_names = list(sample_features.keys())

        ablation_results = []

        for feature_to_remove in all_feature_names:
            print(f"Testing without feature: {feature_to_remove}")

            # Create modified feature extractor
            modified_extractor = FlowFeatureExtractor()

            # Train with modified features
            modified_normal_features = []
            for flow in normal_flows:
                features = modified_extractor.extract_comprehensive_features(flow)
                if feature_to_remove in features:
                    del features[feature_to_remove]
                modified_normal_features.append(features)

            # Create new quantizer and detector
            temp_quantizer = BinaryQuantizer(self.quantizer.embedding_size)
            temp_quantizer.fit(modified_normal_features)

            temp_embeddings = []
            for features in modified_normal_features:
                embedding = temp_quantizer.quantize_flow(features)
                temp_embeddings.append(embedding)

            temp_detector = LSHAnomalyDetector(self.detector.embedding_size)
            temp_detector.fit(temp_embeddings)

            # Test with modified features
            modified_test_features = []
            for flow in test_flows:
                features = modified_extractor.extract_comprehensive_features(flow)
                if feature_to_remove in features:
                    del features[feature_to_remove]
                modified_test_features.append(features)

            # Predict
            test_predictions = []
            for features in modified_test_features:
                embedding = temp_quantizer.quantize_flow(features)
                is_anomaly, score = temp_detector.predict_anomaly(embedding)
                test_predictions.append(is_anomaly)

            # Calculate performance
            y_pred = [1 if pred else 0 for pred in test_predictions]
            f1 = f1_score(y_true, y_pred)

            ablation_results.append({
                'removed_feature': feature_to_remove,
                'f1_score': f1,
                'f1_drop': baseline_f1 - f1,
                'relative_drop': (baseline_f1 - f1) / baseline_f1 if baseline_f1 > 0 else 0
            })

        return ablation_results, baseline_f1

def run_single_experiment(seed: int = 42):
    try:
        # Try to load real data
        train_df = pd.read_csv('UNSW_NB15_training-set.csv', low_memory=False)
        test_df = pd.read_csv('UNSW_NB15_testing-set.csv', low_memory=False)
        print(f"Loaded real UNSW-NB15 dataset (seed: {seed})")
    except:
        print("Cannot read data")
        return None

    # Convert labels properly
    def fix_labels(df):
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        return df

    train_df = fix_labels(train_df)
    test_df = fix_labels(test_df)

    # Sample balanced data for training (only normal flows) with different seed
    normal_train = train_df[train_df['label'] == 0].sample(
        n=min(5000, len(train_df[train_df['label'] == 0])),
        random_state=seed
    )

    # Sample balanced test data with different seed
    normal_test = test_df[test_df['label'] == 0].sample(
        n=min(1000, len(test_df[test_df['label'] == 0])),
        random_state=seed
    )
    anomaly_test = test_df[test_df['label'] == 1].sample(
        n=min(500, len(test_df[test_df['label'] == 1])),
        random_state=seed
    )
    test_combined = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=seed)

    print(f"Training on {len(normal_train)} normal flows")
    print(f"Testing on {len(test_combined)} flows ({len(normal_test)} normal, {len(anomaly_test)} anomaly)")

    # Convert to lists
    train_flows = normal_train.to_dict('records')
    test_flows = test_combined.to_dict('records')
    y_true = test_combined['label'].tolist()

    # Train model
    embedder = StreamingEmbedder(embedding_size=64)

    start_time = time.time()
    embedder.train(train_flows, seed)
    train_time = time.time() - start_time

    # Test model
    start_time = time.time()
    all_results = embedder.predict(test_flows)
    test_time = time.time() - start_time

    # Evaluate all models
    model_names = ['Binary_LSH', 'Isolation_Forest', 'One_Class_SVM']
    results_summary = {}

    print(f"\nModel Comparison Results (seed: {seed}):")
    print("="*60)

    for model_name in model_names:
        predictions, scores = all_results[model_name]
        y_pred = [1 if pred else 0 for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, scores)
        except:
            auc = 0.5

        results_summary[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc,
            'train_time': train_time,
            'test_time': test_time,
            'throughput': len(test_flows)/test_time
        }

        print(f"{model_name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC Score: {auc:.3f}")
        print()

    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
    print(f"Throughput: {len(test_flows)/test_time:.0f} flows/second")

    return results_summary

def run_ablation_study(seed: int = 42):
    try:
        train_df = pd.read_csv('UNSW_NB15_training-set.csv', low_memory=False)
        test_df = pd.read_csv('UNSW_NB15_testing-set.csv', low_memory=False)
        print("Loaded UNSW-NB15 dataset")
    except:
        print("Cannot read data")
        return None

    # Fix labels
    def fix_labels(df):
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        return df

    train_df = fix_labels(train_df)
    test_df = fix_labels(test_df)

    # Sample data
    normal_train = train_df[train_df['label'] == 0].sample(n=min(2000, len(train_df[train_df['label'] == 0])), random_state=seed)
    normal_test = test_df[test_df['label'] == 0].sample(n=min(500, len(test_df[test_df['label'] == 0])), random_state=seed)
    anomaly_test = test_df[test_df['label'] == 1].sample(n=min(250, len(test_df[test_df['label'] == 1])), random_state=seed)
    test_combined = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=seed)

    print(f"Training: {len(normal_train)} normal flows")
    print(f"Testing: {len(test_combined)} flows ({len(normal_test)} normal, {len(anomaly_test)} anomaly)")

    # Convert to lists
    train_flows = normal_train.to_dict('records')
    test_flows = test_combined.to_dict('records')
    y_true = test_combined['label'].tolist()

    # Initialize and train model
    embedder = StreamingEmbedder(embedding_size=64)
    embedder.train(train_flows, seed)

    # 1. Feature Analysis
    print("\n1. Analyzing extracted features...")
    feature_stats, all_features = embedder.get_feature_analysis(train_flows)

    # 2. Quantization Analysis
    print("\n2. Analyzing quantization process...")
    quant_analysis = embedder.quantizer.get_quantization_analysis()

    # Generate embeddings for visualization
    train_embeddings = []
    for features in all_features:
        embedding = embedder.quantizer.quantize_flow(features)
        train_embeddings.append(embedding)

    # 3. Visualize quantization impact
    print("\n3. Creating quantization visualizations...")
    correlations, feature_names = embedder.quantizer.visualize_quantization_impact(all_features, train_embeddings)
    print("\nFeature indices and their names:")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")


    # 4. Run ablation study
    print("\n4. Running feature ablation study...")
    ablation_results, baseline_f1 = embedder.run_feature_ablation(train_flows, test_flows, y_true)

    # 5. Visualize ablation results
    print("\n5. Creating ablation study visualizations...")
    visualize_ablation_results(ablation_results, baseline_f1, feature_stats)

    # 6. Get baseline performance
    print("\n6. Getting baseline performance...")
    results = embedder.predict(test_flows)
    y_pred = [1 if pred else 0 for pred in results['Binary_LSH'][0]]

    print(f"\nBaseline Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1-Score: {baseline_f1:.3f}")
    print(f"AUC: {roc_auc_score(y_true, results['Binary_LSH'][1]):.3f}")

    # Print feature importance
    print(f"\nQuantization Analysis:")
    print(f"Total features quantized: {quant_analysis['num_features']}")
    print(f"Categorical features: {len(quant_analysis['categorical_features'])}")
    print(f"Continuous features: {len(quant_analysis['continuous_features'])}")

    return ablation_results, feature_stats, quant_analysis

def visualize_ablation_results(ablation_results, baseline_f1, feature_stats):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Feature importance (by F1 drop)
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df = ablation_df.sort_values('f1_drop', ascending=False)

    top_10 = ablation_df.head(10)
    axes[0, 0].barh(range(len(top_10)), top_10['f1_drop'])
    axes[0, 0].set_yticks(range(len(top_10)))
    axes[0, 0].set_yticklabels(top_10['removed_feature'], fontsize=8)
    axes[0, 0].set_title('Top 10 Most Important Features\n(by F1 Score Drop)')
    axes[0, 0].set_xlabel('F1 Score Drop')
    print("FEATURE IMPORTANCE")
    print(top_10['removed_feature'])

    # 2. Relative importance
    axes[0, 1].barh(range(len(top_10)), top_10['relative_drop'])
    axes[0, 1].set_yticks(range(len(top_10)))
    axes[0, 1].set_yticklabels(top_10['removed_feature'], fontsize=8)
    axes[0, 1].set_title('Top 10 Most Important Features\n(by Relative F1 Drop)')
    axes[0, 1].set_xlabel('Relative F1 Score Drop')
    print("RELATIVE FEATURE IMPORTANCE")
    print(top_10['relative_drop'])

    # 3. F1 scores without each feature
    axes[1, 0].scatter(range(len(ablation_df)), ablation_df['f1_score'], alpha=0.6)
    axes[1, 0].axhline(y=baseline_f1, color='r', linestyle='--', label=f'Baseline F1: {baseline_f1:.3f}')
    axes[1, 0].set_title('F1 Scores with Feature Removal')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    print("F1 scores with each feature removed:")
    for idx, (feature, f1) in enumerate(zip(ablation_df['removed_feature'], ablation_df['f1_score'])):
        print(f"{idx}: Removed '{feature}' -> F1 Score: {f1:.3f}")

    # 4. Distribution of F1 drops
    axes[1, 1].hist(ablation_df['f1_drop'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Distribution of F1 Score Drops')
    axes[1, 1].set_xlabel('F1 Score Drop')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('Ablations/ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Ablation study visualizations saved!")

if __name__ == "__main__":
    print("Starting single-run anomaly detection experiment with ablation study...")
    run_ablation_study(seed=42)
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

EMB_SIZE = 64


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

        # State encoding - but we'll remove this based on ablation study
        state = str(flow_record.get('state', 'other')).upper()
        if state not in self.state_encoder:
            self.state_encoder[state] = len(self.state_encoder) + 1
        features['state_encoded'] = self.state_encoder[state]

        # Derived features that are important for anomaly detection
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

        # Load features - but we'll remove these based on ablation study
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

    def get_optimized_features(self, flow_record: Dict) -> Dict:
        """Extract only the most important features based on ablation study"""
        all_features = self.extract_comprehensive_features(flow_record)

        # Based on ablation study, keep only the most important features
        # and remove the harmful ones
        optimized_features = {
            # Top 6 most important features
            'total_packets': all_features['total_packets'],
            'protocol': all_features['protocol'],
            'bytes_per_packet': all_features['bytes_per_packet'],
            'duration': all_features['duration'],
            'total_bytes': all_features['total_bytes'],
            'packets_per_second': all_features['packets_per_second'],

            # Additional useful features that don't hurt performance
            'service_encoded': all_features['service_encoded'],
            'dst_load': all_features['dst_load'],
            'mean_packet_size': all_features['mean_packet_size'],
        }

        # Remove harmful features identified in ablation study:
        # - state_encoded, src_load, load_ratio, src_bytes_ratio

        return optimized_features

class FlowAwareBinaryQuantizer:
    def __init__(self, embedding_size: int = EMB_SIZE): # can change here
        self.embedding_size = embedding_size
        self.quantization_params = {}
        self.is_fitted = False

    def fit(self, features_list: List[Dict]):
        print("Learning flow-aware quantization parameters...")
        feature_collections = defaultdict(list)

        for features in features_list:
            for key, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_collections[key].append(value)

        # Learn quantization parameters for each feature
        for feature_name, values in feature_collections.items():
            if len(values) < 10:  # Skip features with too few samples
                continue

            values = np.array(values)

            if feature_name in ['protocol', 'service_encoded']:
                unique_vals = np.unique(values)
                self.quantization_params[feature_name] = {
                    'type': 'categorical',
                    'values': unique_vals,
                    'bits': min(8, max(2, int(np.ceil(np.log2(len(unique_vals) + 1)))))
                }
            else:
                # More aggressive quantization to create better separation
                if feature_name in ['total_packets', 'total_bytes']:
                    # For volume features, use log-scale quantization
                    log_values = np.log10(values + 1)
                    bits = 10  # More bits for important features
                    percentiles = np.linspace(0, 100, 2**bits)
                    thresholds = np.percentile(log_values, percentiles)

                    self.quantization_params[feature_name] = {
                        'type': 'log_continuous',
                        'thresholds': thresholds,
                        'bits': bits
                    }
                elif feature_name in ['bytes_per_packet', 'packets_per_second']:
                    # For rate features, use adaptive quantization
                    # Remove extreme outliers more aggressively
                    q01, q99 = np.percentile(values, [1, 99])
                    filtered_values = values[(values >= q01) & (values <= q99)]

                    if len(filtered_values) > 0:
                        bits = 8
                        percentiles = np.linspace(0, 100, 2**bits)
                        thresholds = np.percentile(filtered_values, percentiles)

                        self.quantization_params[feature_name] = {
                            'type': 'adaptive_continuous',
                            'thresholds': thresholds,
                            'bits': bits,
                            'min_val': q01,
                            'max_val': q99
                        }
                elif feature_name == 'duration':
                    # For duration, use log-scale if there's high variance
                    if np.std(values) > np.mean(values):
                        log_values = np.log10(values + 1)
                        bits = 8
                        percentiles = np.linspace(0, 100, 2**bits)
                        thresholds = np.percentile(log_values, percentiles)

                        self.quantization_params[feature_name] = {
                            'type': 'log_continuous',
                            'thresholds': thresholds,
                            'bits': bits
                        }
                    else:
                        # Regular quantization
                        bits = 6
                        percentiles = np.linspace(0, 100, 2**bits)
                        thresholds = np.percentile(values, percentiles)

                        self.quantization_params[feature_name] = {
                            'type': 'continuous',
                            'thresholds': thresholds,
                            'bits': bits
                        }
                else:
                    # Default quantization for other features
                    bits = 6
                    percentiles = np.linspace(0, 100, 2**bits)
                    thresholds = np.percentile(values, percentiles)

                    self.quantization_params[feature_name] = {
                        'type': 'continuous',
                        'thresholds': thresholds,
                        'bits': bits
                    }

        self.is_fitted = True
        print(f"Learned quantization for {len(self.quantization_params)} features")

    def _quantize_feature(self, value, feature_name):
        if feature_name not in self.quantization_params:
            return 0, 2  # Default

        params = self.quantization_params[feature_name]

        if params['type'] == 'categorical':
            if value in params['values']:
                idx = np.where(params['values'] == value)[0][0]
                return idx, params['bits']
            else:
                return 0, params['bits']

        elif params['type'] == 'log_continuous':
            log_value = np.log10(value + 1)
            bin_idx = np.searchsorted(params['thresholds'], log_value)
            bin_idx = min(bin_idx, len(params['thresholds']) - 1)
            return bin_idx, params['bits']

        elif params['type'] == 'adaptive_continuous':
            clipped_value = np.clip(value, params['min_val'], params['max_val'])
            bin_idx = np.searchsorted(params['thresholds'], clipped_value)
            bin_idx = min(bin_idx, len(params['thresholds']) - 1)
            return bin_idx, params['bits']

        else:  # continuous
            bin_idx = np.searchsorted(params['thresholds'], value)
            bin_idx = min(bin_idx, len(params['thresholds']) - 1)
            return bin_idx, params['bits']

    def quantize_flow(self, features: Dict) -> int:
        if not self.is_fitted:
            raise ValueError("Quantizer must be fitted before use")

        # Priority order based on ablation study results
        feature_priority = [
            'total_packets',     # Most important
            'protocol',          # Second most important
            'bytes_per_packet',  # Third most important
            'duration',          # Fourth most important
            'total_bytes',       # Fifth most important
            'packets_per_second', # Sixth most important
            'service_encoded',
            'dst_load',
            'mean_packet_size',
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
                    binary_embedding = (binary_embedding << bits_needed) | quantized_value
                    used_bits += bits_needed

        return binary_embedding

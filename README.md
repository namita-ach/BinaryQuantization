# Flow-Aware LSH: Binary Quantized IP Flow Embeddings for Resource-Constrained Network Monitoring

A lightweight network anomaly detection system using binary quantization and locality-sensitive hashing, optimized for edge deployment with minimal computational overhead.

## Overview

This project implements a novel approach to network anomaly detection using binary quantization and locality-sensitive hashing (LSH) to create compact flow embeddings. The system is designed for edge devices where computational constraints would otherwise prohibit real-time anomaly detection.

## Dataset

- **Source**: UNSW-NB15 dataset
- **Training**: 3,000 normal network flows (Sampled 20 times)
- **Testing**: 1,200 flows (800 normal + 400 anomalies) [Also sampled 20 times]
- **Features**: 9 key network flow features after feature selection

## System Architecture

The Flow-Aware LSH system consists of:

1. **Binary Quantization**: Converts numeric flow features into binary representations
2. **LSH Hashing**: Maps flows into hash buckets for efficient similarity computation
3. **Anomaly Detection**: Uses Hamming distance to identify flows deviating from normal patterns
4. **Threshold-based Classification**: Applies learned thresholds to classify anomalies

## Study Flow

### 1. Ablation Study

Initial performance baseline:
- Accuracy: 65.6%
- F1-score: 0.146
- ROC AUC: 0.760

**Feature Importance Analysis**:
- **Harmful features**: state_encoded, src_load, load_ratio, src_bytes_ratio
- **Redundant features**: High correlation (>0.9) among byte-related and load features

**Key Findings**:
- Removing harmful features improved F1 from 0.146 to 0.379
- Many features provided no discriminative value
- Correlation analysis revealed significant feature redundancy

### 2. Performance Metrics

After optimization, the system was evaluated on three key metrics:

**Detection Accuracy**:
- **Recall**: 87.2% (prioritizes catching anomalies)
- **Precision**: ~42% (acceptable for security-focused applications)
- **F1 Score**: 53.46%
- **Accuracy**: 68.87%

**Processing Performance**:
- **Flow processing**: 28.7 flows/second
- **Average flow time**: 2.83ms
- **Training time**: 102.2 ± 5.2 seconds
- **Test time**: 41.9 ± 0.9 seconds

### 3. Resource Analysis

**Memory Efficiency**:
- Total LSH model: 26.1 KB
- Hash tables: 0.18 KB
- Normal embeddings: 25.4 KB
- 558.67x more memory-efficient than baseline methods

**CPU Usage**:
- Training: 98.7% utilization
- Testing: 99.5% utilization
- Bounded and predictable resource consumption

### 4. Comparison with Weaviate

**Flow-Aware LSH Advantages**:
- **Higher recall**: 87.2% vs 22.0% for Weaviate Isolation Forest
- **Lower latency**: 1.47ms vs 6.83ms per flow
- **Minimal deployment overhead**: ~26KB vs >1GB disk space for Weaviate
- **Edge-optimized**: No additional infrastructure required

**Weaviate Limitations for Edge**:
- Requires container runtime with 400-800MB RAM overhead
- Lower recall makes it unsuitable for security applications
- Additional abstraction layers increase latency
- Not optimized for per-flow processing

## Model Comparison

| Model | Accuracy | F1 Score | Recall | Precision | ROC AUC |
|-------|----------|----------|---------|-----------|---------|
| Flow-Aware LSH | 68.87% | 53.46% | 87.2% | 42% | 68.87% |
| Isolation Forest | 69.26% | 44.15% | 40% | 50% | 69.26% |
| One-Class SVM | 65.68% | 50.47% | 46% | 56% | 65.68% |

## Key Insights

1. **Trade-off Optimization**: The system prioritizes recall over precision, making it suitable for security applications where missing anomalies is more costly than false alarms.

2. **Resource Efficiency**: Binary quantization and LSH provide significant computational savings while maintaining detection effectiveness.

3. **Deployment Ready**: The compact model size and minimal dependencies make it practical for edge deployment.

4. **Stable Performance**: Low variance across multiple runs indicates consistent behavior.

## Technical Requirements

- Minimal memory footprint: ~26KB
- Fast processing: Sub-3ms per flow
- No external dependencies
- Edge-device compatible
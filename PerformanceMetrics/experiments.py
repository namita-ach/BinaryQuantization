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

from anomaly_detection import StreamingEmbedder

EMB_SIZE = 64


def run_single_experiment(seed: int = 42):
    try:
        # Try to load real data
        train_df = pd.read_csv('/home/pes1ug22am100/Documents/BinaryQuantization/PerformanceMetrics/UNSW_NB15_training-set.csv', low_memory=False)
        test_df = pd.read_csv('/home/pes1ug22am100/Documents/BinaryQuantization/PerformanceMetrics/UNSW_NB15_testing-set.csv', low_memory=False)
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

    # Train model
    embedder = StreamingEmbedder(embedding_size= EMB_SIZE)

    start_time = time.time()
    embedder.train(train_flows, seed)
    train_time = time.time() - start_time

    # Test model
    start_time = time.time()
    all_results = embedder.predict(test_flows)
    test_time = time.time() - start_time

    # Evaluate all models
    print(f"\nFixed Model Results (seed: {seed}):")
    print("="*60)

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

    print("="*60)
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Testing time: {test_time:.2f} seconds")
    print("="*60)

    return all_results

def run_multiple_experiments(num_runs: int = 5, base_seed: int = 42):
    all_accuracy_scores = defaultdict(list)
    all_precision_scores = defaultdict(list)
    all_recall_scores = defaultdict(list)
    all_f1_scores = defaultdict(list)
    all_roc_scores = defaultdict(list)

    seeds = [355, 1307, 6390, 9026, 2997, 9766, 1095, 4926, 276, 8706] # for consistency
    # print(f"Total unique seeds: {len(seeds)}")
    # seeds=[]
    # for i in range(20):
    #     seeds.append(random.randint(1, 10000))

    for i, seed in enumerate(seeds):
        print(f"\nRunning experiment {i+1}/{len(seeds)} with seed={seed}")
        results = run_single_experiment(seed)
        if results is None:
            continue  # Skip if the experiment failed

        # Reconstruct ground truth
        test_df = pd.read_csv('/home/pes1ug22am100/Documents/BinaryQuantization/PerformanceMetrics/UNSW_NB15_testing-set.csv', low_memory=False)
        test_df = test_df[test_df['label'].isin([0, 1])]
        normal_test = test_df[test_df['label'] == 0].sample(n=min(800, len(test_df[test_df['label'] == 0])), random_state=seed)
        anomaly_test = test_df[test_df['label'] == 1].sample(n=min(400, len(test_df[test_df['label'] == 1])), random_state=seed)
        test_combined = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=seed)
        y_true = test_combined['label'].tolist()

        for model_name, (predictions, scores) in results.items():
            y_pred = [1 if pred else 0 for pred in predictions]
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, scores)

            all_accuracy_scores[model_name].append(accuracy)
            all_precision_scores[model_name].append(precision)
            all_recall_scores[model_name].append(recall)
            all_f1_scores[model_name].append(f1)
            all_roc_scores[model_name].append(roc_auc)

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

    return pd.DataFrame({
        'Model': list(all_accuracy_scores.keys()),
        'Accuracy Mean': [np.mean(v) for v in all_accuracy_scores.values()],
        'Accuracy Std': [np.std(v) for v in all_accuracy_scores.values()],
        'Precision Mean': [np.mean(v) for v in all_precision_scores.values()],
        'Precision Std': [np.std(v) for v in all_precision_scores.values()],
        'Recall Mean': [np.mean(v) for v in all_recall_scores.values()],
        'Recall Std': [np.std(v) for v in all_recall_scores.values()],
        'F1 Mean': [np.mean(v) for v in all_f1_scores.values()],
        'F1 Std': [np.std(v) for v in all_f1_scores.values()],
        'ROC Mean': [np.mean(v) for v in all_roc_scores.values()],
        'ROC Std': [np.std(v) for v in all_roc_scores.values()],
    })

if __name__ == "__main__":
    # Run a single experiment or multiple
    results_df = run_multiple_experiments(num_runs=10, base_seed=42)

    # Optionally, save to CSV
    if results_df is not None:
        results_df.to_csv(f"PerfromanceMetrics/results-{EMB_SIZE}.csv", index=False)
        print(f"Saved summary to results-{EMB_SIZE}.csv")
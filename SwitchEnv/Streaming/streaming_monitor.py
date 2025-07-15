#!/usr/bin/env python3
"""
Real-time Streaming Anomaly Detection Monitor
Provides live monitoring of the streaming detection process
"""

import os
import time
import json
import argparse
import threading
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import psutil
import subprocess
import sys

class StreamingMonitor:
    def __init__(self, results_dir="streaming-results", refresh_interval=2):
        self.results_dir = results_dir
        self.refresh_interval = refresh_interval
        self.monitoring = False
        self.process_pid = None
        self.anomaly_count = 0
        self.last_anomaly_time = None
        self.anomaly_rate_history = deque(maxlen=50)  # Track anomaly detection rate
        
        # Metrics tracking
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.start_time = None
        
        # Display settings
        self.display_width = 80
        
    def find_streaming_process(self):
        """Find the streaming detector process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'streaming_anomaly_detector.py' in ' '.join(proc.info['cmdline']):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_process_metrics(self, pid):
        """Get resource metrics for the process"""
        try:
            proc = psutil.Process(pid)
            return {
                'cpu_percent': proc.cpu_percent(),
                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                'num_threads': proc.num_threads(),
                'status': proc.status()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def get_latest_results(self):
        """Get the latest results from the results directory"""
        if not os.path.exists(self.results_dir):
            return None
        
        # Find the most recent files
        csv_files = [f for f in os.listdir(self.results_dir) if f.endswith('.csv')]
        if not csv_files:
            return None
        
        # Try to read the latest performance file
        perf_files = [f for f in csv_files if 'streaming_performance' in f]
        if perf_files:
            latest_perf = max(perf_files, key=lambda x: os.path.getmtime(os.path.join(self.results_dir, x)))
            try:
                df = pd.read_csv(os.path.join(self.results_dir, latest_perf))
                return df.to_dict('records')
            except:
                pass
        
        return None
    
    def print_header(self):
        """Print the monitor header"""
        print("\033[2J\033[H")  # Clear screen and move to top
        print("=" * self.display_width)
        print("STREAMING ANOMALY DETECTION MONITOR".center(self.display_width))
        print("=" * self.display_width)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_dir}")
        print("=" * self.display_width)
    
    def print_process_status(self, pid, metrics):
        """Print process status information"""
        print(f"\nProcess Status (PID: {pid}):")
        print("-" * 40)
        
        if metrics:
            print(f"Status: {metrics['status']}")
            print(f"CPU Usage: {metrics['cpu_percent']:.1f}%")
            print(f"Memory Usage: {metrics['memory_mb']:.1f} MB")
            print(f"Threads: {metrics['num_threads']}")
        else:
            print("Process not found or access denied")
    
    def print_progress_bar(self, current, total, width=50):
        """Print a progress bar"""
        if total == 0:
            return
        
        progress = min(current / total, 1.0)
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = progress * 100
        
        print(f"Progress: [{bar}] {percentage:.1f}% ({current}/{total})")
    
    def print_performance_metrics(self, results):
        """Print performance metrics"""
        if not results:
            print("\nNo performance data available yet...")
            return
        
        print(f"\nPerformance Metrics:")
        print("-" * 40)
        
        for result in results:
            model_name = result.get('Model', 'Unknown')
            print(f"\n{model_name}:")
            
            # Core metrics
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            for metric in metrics:
                value = result.get(metric.title(), result.get(metric, 0))
                print(f"  {metric.title()}: {value:.4f}")
            
            # Processing metrics
            flows_processed = result.get('Flows_Processed', 0)
            flows_per_sec = result.get('Flows_Per_Second', 0)
            total_time = result.get('Total_Time', 0)
            
            print(f"  Flows Processed: {flows_processed}")
            print(f"  Processing Rate: {flows_per_sec:.2f} flows/sec")
            print(f"  Total Time: {total_time:.2f} seconds")
            
            # Detection metrics
            anomalies_detected = result.get('anomalies_detected', 0)
            total_samples = result.get('total_samples', 0)
            if total_samples > 0:
                detection_rate = (anomalies_detected / total_samples) * 100
                print(f"  Anomaly Rate: {detection_rate:.2f}% ({anomalies_detected}/{total_samples})")
    
    def print_live_stats(self, pid):
        """Print live statistics"""
        current_time = time.time()
        
        # Get process metrics
        proc_metrics = self.get_process_metrics(pid)
        if proc_metrics:
            self.metrics_history['cpu'].append(proc_metrics['cpu_percent'])
            self.metrics_history['memory'].append(proc_metrics['memory_mb'])
            self.metrics_history['timestamp'].append(current_time)
        
        print(f"\nLive Statistics:")
        print("-" * 40)
        
        if self.metrics_history['cpu']:
            avg_cpu = np.mean(list(self.metrics_history['cpu']))
            avg_memory = np.mean(list(self.metrics_history['memory']))
            print(f"Avg CPU Usage: {avg_cpu:.1f}%")
            print(f"Avg Memory Usage: {avg_memory:.1f} MB")
        
        # Show trend
        if len(self.metrics_history['cpu']) > 1:
            cpu_trend = list(self.metrics_history['cpu'])[-10:]  # Last 10 readings
            trend_str = "".join(["▲" if i > 0 else "▼" if i < 0 else "─" 
                               for i in np.diff(cpu_trend)])
            print(f"CPU Trend: {trend_str}")
    
    def monitor_file_system(self):
        """Monitor file system for new results"""
        if not os.path.exists(self.results_dir):
            return
        
        files = os.listdir(self.results_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        print(f"\nResult Files ({len(csv_files)}):")
        print("-" * 40)
        
        for file in sorted(csv_files)[-5:]:  # Show last 5 files
            file_path = os.path.join(self.results_dir, file)
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            time_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            print(f"  {file[:50]:<50} {size:>8} bytes {time_str}")
    
    def estimate_completion(self, current_flows, target_flows, flows_per_sec):
        """Estimate completion time"""
        if flows_per_sec <= 0 or current_flows >= target_flows:
            return "Unknown"
        
        remaining_flows = target_flows - current_flows
        eta_seconds = remaining_flows / flows_per_sec
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f} seconds"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f} minutes"
        else:
            return f"{eta_seconds/3600:.1f} hours"

    def colorize_status(self, status):
        """Add colors to status indicators"""
        colors = {
            'running': '\033[92m',  # Green
            'sleeping': '\033[93m', # Yellow
            'stopped': '\033[91m',  # Red
            'zombie': '\033[91m',   # Red
            'RESET': '\033[0m'      # Reset
        }
        
        color = colors.get(status.lower(), colors['RESET'])
        return f"{color}{status}{colors['RESET']}"
    
    def print_footer(self):
        """Print monitor footer"""
        print("\n" + "=" * self.display_width)
        print("Press Ctrl+C to stop monitoring".center(self.display_width))
        print("=" * self.display_width)
    
    def run_monitor(self):
        """Main monitoring loop"""
        self.monitoring = True
        self.start_time = time.time()
        
        print("Starting streaming monitor...")
        print("Looking for streaming process...")
        
        while self.monitoring:
            try:
                # Find the streaming process
                pid = self.find_streaming_process()
                
                if pid:
                    if self.process_pid != pid:
                        self.process_pid = pid
                        print(f"Found streaming process: PID {pid}")
                    
                    # Clear screen and show header
                    self.print_header()
                    
                    # Get process metrics
                    proc_metrics = self.get_process_metrics(pid)
                    
                    # Print process status with colors
                    print(f"\n⚙️  Process Status (PID: {pid}):")
                    print("-" * 40)
                    
                    if proc_metrics:
                        status_colored = self.colorize_status(proc_metrics['status'])
                        print(f"Status: {status_colored}")
                        print(f"CPU Usage: {proc_metrics['cpu_percent']:.1f}%")
                        print(f"Memory Usage: {proc_metrics['memory_mb']:.1f} MB")
                        print(f"Threads: {proc_metrics['num_threads']}")
                        
                        # Add CPU/Memory bars
                        cpu_bar = "█" * int(proc_metrics['cpu_percent'] / 5) + "░" * (20 - int(proc_metrics['cpu_percent'] / 5))
                        mem_bar = "█" * int(min(proc_metrics['memory_mb'] / 50, 20)) + "░" * (20 - int(min(proc_metrics['memory_mb'] / 50, 20)))
                        print(f"CPU: [{cpu_bar}] {proc_metrics['cpu_percent']:.1f}%")
                        print(f"MEM: [{mem_bar}] {proc_metrics['memory_mb']:.1f} MB")
                    else:
                        print("❌ Process not found or access denied")
                    
                    # Get flow progress
                    current_flows, target_flows, flows_per_sec = self.get_flow_progress()
                    
                    # Show progress bar
                    print(f"\n📊 Processing Progress:")
                    print("-" * 40)
                    self.print_progress_bar(current_flows, target_flows)
                    print(f"Flows: {current_flows:,}/{target_flows:,} | Rate: {flows_per_sec:.2f}/sec")
                    
                    if current_flows > 0 and current_flows < target_flows:
                        eta = self.estimate_completion(current_flows, target_flows, flows_per_sec)
                        print(f"🕒 ETA: {eta}")
                    elif current_flows >= target_flows:
                        print("✅ Processing completed!")
                    
                    # Show anomaly status
                    self.print_anomaly_status()
                    
                    # Get latest results and show performance
                    results = self.get_latest_results()
                    if results:
                        print(f"\n📈 Model Performance:")
                        print("-" * 40)
                        for result in results:
                            model_name = result.get('Model', 'Unknown')
                            accuracy = result.get('accuracy', 0)
                            precision = result.get('precision', 0)
                            recall = result.get('recall', 0)
                            f1 = result.get('f1_score', 0)
                            roc_auc = result.get('roc_auc', 0)
                            
                            # Create performance indicators
                            acc_indicator = "🟢" if accuracy > 0.9 else "🟡" if accuracy > 0.7 else "🔴"
                            f1_indicator = "🟢" if f1 > 0.8 else "🟡" if f1 > 0.6 else "🔴"
                            
                            print(f"{model_name}:")
                            print(f"  {acc_indicator} Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
                            print(f"  {f1_indicator} Recall: {recall:.4f} | F1-Score: {f1:.4f}")
                            print(f"  📊 ROC-AUC: {roc_auc:.4f}")
                            
                            # Show detection stats
                            anomalies_detected = result.get('anomalies_detected', 0)
                            total_samples = result.get('total_samples', 0)
                            if total_samples > 0:
                                detection_rate = (anomalies_detected / total_samples) * 100
                                print(f"  🚨 Anomaly Rate: {detection_rate:.2f}% ({anomalies_detected}/{total_samples})")
                            print()
                    
                    # Show live stats with trend
                    self.print_live_stats(pid)
                    
                    # Monitor file system
                    print(f"\n📁 Result Files:")
                    print("-" * 40)
                    if os.path.exists(self.results_dir):
                        files = os.listdir(self.results_dir)
                        csv_files = [f for f in files if f.endswith('.csv')]
                        anomaly_files = [f for f in csv_files if 'anomalous_flows' in f]
                        perf_files = [f for f in csv_files if 'streaming_performance' in f]
                        detailed_files = [f for f in csv_files if 'streaming_detailed' in f]
                        
                        print(f"📊 Performance files: {len(perf_files)}")
                        print(f"🚨 Anomaly files: {len(anomaly_files)}")
                        print(f"📋 Detailed files: {len(detailed_files)}")
                        
                        # Show latest files
                        if csv_files:
                            print("\nLatest files:")
                            for file in sorted(csv_files)[-3:]:  # Show last 3 files
                                file_path = os.path.join(self.results_dir, file)
                                size = os.path.getsize(file_path)
                                mtime = os.path.getmtime(file_path)
                                time_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
                                
                                # Add file type indicator
                                if 'anomalous_flows' in file:
                                    icon = "🚨"
                                elif 'streaming_performance' in file:
                                    icon = "📊"
                                elif 'streaming_detailed' in file:
                                    icon = "📋"
                                else:
                                    icon = "📄"
                                
                                print(f"  {icon} {file[:45]:<45} {size:>6} bytes {time_str}")
                    else:
                        print("❌ Results directory not found")
                    
                    # Show system resources
                    print(f"\n💻 System Resources:")
                    print("-" * 40)
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage(self.results_dir if os.path.exists(self.results_dir) else '/')
                    
                    print(f"System CPU: {cpu_percent:.1f}%")
                    print(f"System Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB used)")
                    print(f"Disk Usage: {disk.percent:.1f}% ({disk.free/1024/1024/1024:.1f}GB free)")
                    
                    # Show network activity if available
                    try:
                        net_io = psutil.net_io_counters()
                        print(f"Network: {net_io.bytes_sent/1024/1024:.1f}MB sent, {net_io.bytes_recv/1024/1024:.1f}MB recv")
                    except:
                        pass
                    
                    self.print_footer()
                    
                else:
                    print(f"❌ No streaming process found. Waiting... ({datetime.now().strftime('%H:%M:%S')})")
                    self.process_pid = None
                    
                    # Show available processes for debugging
                    streaming_processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if 'python' in proc.info['name'].lower():
                                cmdline = ' '.join(proc.info['cmdline'])
                                if 'streaming' in cmdline or 'anomaly' in cmdline:
                                    streaming_processes.append((proc.info['pid'], proc.info['name'], cmdline))
                        except:
                            continue
                    
                    if streaming_processes:
                        print("Found related processes:")
                        for pid, name, cmdline in streaming_processes[:3]:
                            print(f"  PID {pid}: {name} - {cmdline[:60]}...")
                
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped by user")
                self.monitoring = False
                break
            except Exception as e:
                print(f"❌ Error in monitoring: {e}")
                time.sleep(self.refresh_interval)

    def get_flow_progress(self):
        """Get current flow processing progress"""
        # Try to read from log files or process output
        target_flows = int(os.environ.get('STREAM_SAMPLES', 1000))
        
        # Look for progress in latest results
        results = self.get_latest_results()
        if results:
            for result in results:
                current_flows = result.get('Flows_Processed', 0)
                flows_per_sec = result.get('Flows_Per_Second', 0)
                return current_flows, target_flows, flows_per_sec
        
        return 0, target_flows, 0

    def print_anomaly_status(self):
        """Print real-time anomaly detection status"""
        anomaly_count = self.monitor_anomalies()
        
        print(f"\n🚨 Anomaly Detection Status:")
        print("-" * 40)
        print(f"Total Anomalies Detected: {anomaly_count}")
        
        if self.last_anomaly_time:
            time_since = datetime.now() - self.last_anomaly_time.replace(tzinfo=None)
            if time_since.total_seconds() < 60:
                print(f"Last Anomaly: {time_since.seconds} seconds ago")
            else:
                print(f"Last Anomaly: {self.last_anomaly_time.strftime('%H:%M:%S')}")
        else:
            print("Last Anomaly: None detected")
        
        # Show anomaly rate
        if len(self.anomaly_rate_history) > 1:
            recent_rate = np.mean(list(self.anomaly_rate_history)[-10:])
            print(f"Recent Anomaly Rate: {recent_rate:.2f} anomalies/min")

    def monitor_anomalies(self):
        """Monitor anomaly detection in real-time"""
        if not os.path.exists(self.results_dir):
            return 0
        
        # Find anomaly files
        anomaly_files = [f for f in os.listdir(self.results_dir) if 'anomalous_flows' in f and f.endswith('.csv')]
        
        total_anomalies = 0
        latest_anomaly_time = None
        
        for file in anomaly_files:
            try:
                file_path = os.path.join(self.results_dir, file)
                df = pd.read_csv(file_path)
                total_anomalies += len(df)
                
                # Get latest anomaly timestamp
                if 'timestamp' in df.columns and not df.empty:
                    latest_time = pd.to_datetime(df['timestamp']).max()
                    if latest_anomaly_time is None or latest_time > latest_anomaly_time:
                        latest_anomaly_time = latest_time
            except:
                continue
        
        # Update anomaly tracking
        if total_anomalies > self.anomaly_count:
            self.last_anomaly_time = latest_anomaly_time
            self.anomaly_count = total_anomalies
        
        return total_anomalies
    
    def run_summary(self):
        """Show summary of completed runs"""
        if not os.path.exists(self.results_dir):
            print(f"Results directory {self.results_dir} not found")
            return
        
        print("=" * 80)
        print("STREAMING DETECTION SUMMARY")
        print("=" * 80)
        
        # Find all performance files
        csv_files = [f for f in os.listdir(self.results_dir) if f.endswith('.csv')]
        perf_files = [f for f in csv_files if 'streaming_performance' in f]
        
        if not perf_files:
            print("No performance results found")
            return
        
        print(f"Found {len(perf_files)} completed runs:")
        print()
        
        for i, file in enumerate(sorted(perf_files), 1):
            print(f"Run {i}: {file}")
            file_path = os.path.join(self.results_dir, file)
            
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    model = row['Model']
                    accuracy = row.get('accuracy', 0)
                    f1 = row.get('f1_score', 0)
                    flows_processed = row.get('Flows_Processed', 0)
                    flows_per_sec = row.get('Flows_Per_Second', 0)
                    
                    print(f"  {model}: Accuracy={accuracy:.4f}, F1={f1:.4f}, "
                          f"Processed={flows_processed}, Rate={flows_per_sec:.2f}/sec")
                print()
            except Exception as e:
                print(f"  Error reading file: {e}")
                print()

def main():
    parser = argparse.ArgumentParser(description='Streaming Anomaly Detection Monitor')
    parser.add_argument('--results-dir', default='streaming-results',
                       help='Results directory to monitor')
    parser.add_argument('--refresh-interval', type=float, default=2.0,
                       help='Refresh interval in seconds')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of completed runs instead of live monitoring')
    
    args = parser.parse_args()
    
    monitor = StreamingMonitor(
        results_dir=args.results_dir,
        refresh_interval=args.refresh_interval
    )
    
    if args.summary:
        monitor.run_summary()
    else:
        try:
            monitor.run_monitor()
        except KeyboardInterrupt:
            print("\nMonitor stopped")

if __name__ == "__main__":
    main()
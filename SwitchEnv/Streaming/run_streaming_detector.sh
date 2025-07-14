#!/bin/bash

# Streaming Anomaly Detection Configuration Script
# This script sets up environment variables and runs the streaming detector

# Default configuration
DEFAULT_STREAM_SAMPLES=1000
DEFAULT_BATCH_SIZE=100
DEFAULT_EMBEDDING_SIZE=64
DEFAULT_SLEEP_INTERVAL=0.01
DEFAULT_RESULTS_DIR="streaming-results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Streaming Anomaly Detection${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --samples NUM         Number of samples to process (default: $DEFAULT_STREAM_SAMPLES)"
    echo "  -b, --batch-size NUM      Batch size for processing (default: $DEFAULT_BATCH_SIZE)"
    echo "  -e, --embedding-size NUM  Embedding size (default: $DEFAULT_EMBEDDING_SIZE)"
    echo "  -i, --interval FLOAT      Sleep interval between flows in seconds (default: $DEFAULT_SLEEP_INTERVAL)"
    echo "  -r, --results-dir PATH    Results directory (default: $DEFAULT_RESULTS_DIR)"
    echo "  -t, --train-file PATH     Training dataset file path"
    echo "  --test-file PATH          Test dataset file path"
    echo "  --seed NUM                Random seed (default: 42)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  STREAM_SAMPLES            Number of samples to process"
    echo "  BATCH_SIZE                Batch size for processing"
    echo "  EMBEDDING_SIZE            Embedding size"
    echo "  SLEEP_INTERVAL            Sleep interval between flows"
    echo "  RESULTS_DIR               Results directory"
    echo "  TRAIN_FILE                Training dataset file path"
    echo "  TEST_FILE                 Test dataset file path"
    echo ""
    echo "Examples:"
    echo "  $0 --samples 500 --batch-size 50"
    echo "  $0 -s 2000 -b 200 -e 128"
    echo "  STREAM_SAMPLES=1500 $0"
}

# Parse command line arguments
STREAM_SAMPLES=${STREAM_SAMPLES:-$DEFAULT_STREAM_SAMPLES}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
EMBEDDING_SIZE=${EMBEDDING_SIZE:-$DEFAULT_EMBEDDING_SIZE}
SLEEP_INTERVAL=${SLEEP_INTERVAL:-$DEFAULT_SLEEP_INTERVAL}
RESULTS_DIR=${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--samples)
            STREAM_SAMPLES="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--embedding-size)
            EMBEDDING_SIZE="$2"
            shift 2
            ;;
        -i|--interval)
            SLEEP_INTERVAL="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -t|--train-file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required files
if [[ -z "$TRAIN_FILE" ]]; then
    TRAIN_FILE="/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_training-set.csv"
fi

if [[ -z "$TEST_FILE" ]]; then
    TEST_FILE="/home/pes1ug22am100/Documents/IP-flow-vector-embeddings/BinaryQuantization/Refactored-resource/UNSW_NB15_testing-set.csv"
fi

# Check if files exist
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo -e "${RED}Error: Training file not found: $TRAIN_FILE${NC}"
    echo -e "${YELLOW}Please set the TRAIN_FILE environment variable or use --train-file option${NC}"
    exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
    echo -e "${RED}Error: Test file not found: $TEST_FILE${NC}"
    echo -e "${YELLOW}Please set the TEST_FILE environment variable or use --test-file option${NC}"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Print configuration
print_header
echo -e "${GREEN}Configuration:${NC}"
echo "  Stream samples: $STREAM_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Embedding size: $EMBEDDING_SIZE"
echo "  Sleep interval: ${SLEEP_INTERVAL}s"
echo "  Results directory: $RESULTS_DIR"
echo "  Training file: $TRAIN_FILE"
echo "  Test file: $TEST_FILE"
echo "  Random seed: $SEED"
echo ""

# Set environment variables
export STREAM_SAMPLES
export BATCH_SIZE
export EMBEDDING_SIZE
export SLEEP_INTERVAL
export RESULTS_DIR
export TRAIN_FILE
export TEST_FILE

# Check if Python script exists
PYTHON_SCRIPT="streaming_anomaly_detector.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo -e "${RED}Error: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Check Python dependencies
echo -e "${YELLOW}Checking Python dependencies...${NC}"
python3 -c "
import sys
required_modules = ['numpy', 'pandas', 'sklearn', 'psutil']
missing_modules = []

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print(f'Missing required modules: {missing_modules}')
    print('Please install them using: pip install ' + ' '.join(missing_modules))
    sys.exit(1)
else:
    print('All required modules are available')
"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Please install missing dependencies${NC}"
    exit 1
fi

# Run the streaming detector
echo -e "${GREEN}Starting streaming anomaly detection...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop gracefully${NC}"
echo ""

python3 "$PYTHON_SCRIPT" --seed "$SEED"

# Check if the run was successful
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Streaming detection completed successfully!${NC}"
    echo -e "${BLUE}Results saved to: $RESULTS_DIR/${NC}"
    
    # Show latest results
    if [[ -d "$RESULTS_DIR" ]]; then
        echo -e "${GREEN}Latest result files:${NC}"
        ls -la "$RESULTS_DIR"/*.csv 2>/dev/null | tail -5
    fi
else
    echo -e "${RED}Streaming detection failed${NC}"
    exit 1
fi
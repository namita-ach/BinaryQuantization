#!/bin/bash

# Enhanced Streaming Anomaly Detection Configuration Script
# This script sets up virtual environment, installs dependencies, and runs the streaming detector

# Default configuration
DEFAULT_STREAM_SAMPLES=1000
DEFAULT_BATCH_SIZE=100
DEFAULT_EMBEDDING_SIZE=64
DEFAULT_SLEEP_INTERVAL=0.01
DEFAULT_RESULTS_DIR="streaming-results"
VENV_NAME="ovs_env"

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
    echo "  --skip-venv               Skip virtual environment setup (use existing)"
    echo "  --clean-install           Remove existing virtual environment and reinstall"
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
    echo "  $0 --clean-install --samples 1500"
    echo "  STREAM_SAMPLES=1500 $0"
}

setup_virtual_environment() {
    echo -e "${YELLOW}Setting up virtual environment...${NC}"
    
    # Remove existing virtual environment if clean install is requested
    if [[ "$CLEAN_INSTALL" == "true" && -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_NAME"
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}Creating virtual environment '$VENV_NAME'...${NC}"
        python3 -m venv "$VENV_NAME"
        if [[ $? -ne 0 ]]; then
            echo -e "${RED}Failed to create virtual environment${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Virtual environment '$VENV_NAME' already exists${NC}"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Failed to activate virtual environment${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Virtual environment activated${NC}"
    
    # Upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip
    
    # Install required packages
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install pandas scikit-learn numpy psutil memory_profiler
    
    # Additional packages that might be needed based on common usage
    echo -e "${YELLOW}Installing additional packages...${NC}"
    pip install matplotlib seaborn scipy
    
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Failed to install required packages${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All packages installed successfully${NC}"
}

check_dependencies() {
    echo -e "${YELLOW}Checking Python dependencies...${NC}"
    
    # Use the virtual environment python
    "$VENV_NAME/bin/python" -c "
import sys
required_modules = ['numpy', 'pandas', 'sklearn', 'psutil', 'queue', 'threading', 'json', 'subprocess', 'socket', 'struct', 'collections', 'datetime', 'signal', 'argparse']
missing_modules = []

for module in required_modules:
    try:
        if module == 'sklearn':
            # Check for scikit-learn
            import sklearn
        else:
            __import__(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print(f'Missing required modules: {missing_modules}')
    sys.exit(1)
else:
    print('All required modules are available')
"

    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Some dependencies are missing. Please check the installation.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All dependencies are satisfied${NC}"
}

# Parse command line arguments
STREAM_SAMPLES=${STREAM_SAMPLES:-$DEFAULT_STREAM_SAMPLES}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
EMBEDDING_SIZE=${EMBEDDING_SIZE:-$DEFAULT_EMBEDDING_SIZE}
SLEEP_INTERVAL=${SLEEP_INTERVAL:-$DEFAULT_SLEEP_INTERVAL}
RESULTS_DIR=${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}
SEED=42
SKIP_VENV=false
CLEAN_INSTALL=false

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
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --clean-install)
            CLEAN_INSTALL=true
            shift
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
echo "  Virtual environment: $VENV_NAME"
echo "  Skip venv setup: $SKIP_VENV"
echo "  Clean install: $CLEAN_INSTALL"
echo ""

# Setup virtual environment unless skipped
if [[ "$SKIP_VENV" != "true" ]]; then
    setup_virtual_environment
else
    echo -e "${YELLOW}Skipping virtual environment setup${NC}"
    # Still activate if it exists
    if [[ -d "$VENV_NAME" ]]; then
        source "$VENV_NAME/bin/activate"
        echo -e "${GREEN}Activated existing virtual environment${NC}"
    else
        echo -e "${RED}Virtual environment '$VENV_NAME' not found. Run without --skip-venv first.${NC}"
        exit 1
    fi
fi

# Check dependencies
check_dependencies

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

# Run the streaming detector with sudo while keeping the virtual environment
echo -e "${GREEN}Starting streaming anomaly detection...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop gracefully${NC}"
echo -e "${BLUE}Running with sudo to ensure proper permissions...${NC}"
echo ""

# Check if we need sudo
if [[ $EUID -ne 0 ]]; then
    echo -e "${YELLOW}Running with sudo while preserving virtual environment...${NC}"
    sudo -E "$VENV_NAME/bin/python" "$PYTHON_SCRIPT" --seed "$SEED"
else
    echo -e "${GREEN}Already running as root${NC}"
    "$VENV_NAME/bin/python" "$PYTHON_SCRIPT" --seed "$SEED"
fi

# Check if the run was successful
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Streaming detection completed successfully!${NC}"
    echo -e "${BLUE}Results saved to: $RESULTS_DIR/${NC}"
    
    # Show latest results
    if [[ -d "$RESULTS_DIR" ]]; then
        echo -e "${GREEN}Latest result files:${NC}"
        ls -la "$RESULTS_DIR"/*.csv 2>/dev/null | tail -5
    fi
    
    # Show virtual environment info
    echo -e "${BLUE}Virtual environment location: $(pwd)/$VENV_NAME${NC}"
    echo -e "${YELLOW}To manually activate: source $VENV_NAME/bin/activate${NC}"
else
    echo -e "${RED}Streaming detection failed${NC}"
    exit 1
fi
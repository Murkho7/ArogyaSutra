#!/bin/bash

MODEL="iit-patna-cse-ai/arogyasutraV1E3.5"
NAME="qwen"
CUDA_DEVICES=${3:-"0"}  # Default to GPU 4,5,6,7 if not provided
PORT=${4:-10010}  # Default starting port is 10010 if not provided

# Store all background process PIDs
PIDS=()

# Signal handling function
cleanup() {
    echo "Stopping all servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill -TERM "$pid"
        fi
    done
    
    # Wait for processes to exit
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -KILL "$pid"
        fi
    done
    
    exit 0
}

# Set signal handlers
trap cleanup SIGINT SIGTERM

# Split CUDA_DEVICES by comma into array
IFS=',' read -ra DEVICE_ARRAY <<< "$CUDA_DEVICES"

# Iterate over device array and start server for each device
for ((i=0; i<${#DEVICE_ARRAY[@]}-1; i++)); do
    DEVICE=${DEVICE_ARRAY[$i]}
    CURRENT_PORT=$((PORT + DEVICE))
    
    echo "Starting server on GPU $DEVICE with port $CURRENT_PORT"
    CUDA_VISIBLE_DEVICES=$DEVICE python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $CURRENT_PORT \
        --model $MODEL \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.7 \
        --limit-mm-per-prompt image=10 \
        --served-model-name $NAME &
    
    # Record process PID
    PIDS+=($!)
done

# The last device does not use &, but still needs to record PID
LAST_DEVICE=${DEVICE_ARRAY[-1]}
LAST_PORT=$((PORT + LAST_DEVICE))

echo "Starting server on GPU $LAST_DEVICE with port $LAST_PORT"
CUDA_VISIBLE_DEVICES=$LAST_DEVICE python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port $LAST_PORT \
    --model $MODEL \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.7 \
    --limit-mm-per-prompt image=10 \
    --served-model-name $NAME &

# Record the last process PID
PIDS+=($!)

# Wait for all processes
wait

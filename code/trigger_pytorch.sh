#!/bin/bash

# Initialize conda for Git Bash by sourcing the conda.sh script.
source /c/Users/giann/anaconda3/etc/profile.d/conda.sh

# (Optional) Activate your environment if you haven't already done so.
conda activate fl_pytorch_env

# Check if a number of clients was provided; if not, default to 20.
NUM_CLIENTS=${1:-10}

echo "Starting $NUM_CLIENTS clients..."

# Loop from 0 to NUM_CLIENTS-1 and start each client in the background.
for (( cid=0; cid<NUM_CLIENTS; cid++ )); do
    echo "Starting client with CID: $cid"
    /c/Users/giann/anaconda3/envs/fl_pytorch_env/python.exe pytorch/client_pytorch.py --cid "$cid" &
done

# Wait for all background processes to complete.
wait

echo "All clients have finished."

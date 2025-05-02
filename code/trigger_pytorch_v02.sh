#!/bin/bash

# Initialize conda for Git Bash by sourcing the conda.sh script.
source /c/Users/igallios/AppData/Local/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate fl_pytorch_env_20250401

# Total clients to run from this machine
NUM_CLIENTS=10

# Starting CID (30 since first machine ran 0–29)
START_CID=0

echo "Starting $NUM_CLIENTS clients starting from CID $START_CID..."

# Loop from START_CID to START_CID+NUM_CLIENTS-1
for (( cid=START_CID; cid<START_CID+NUM_CLIENTS; cid++ )); do
    echo "Starting client with CID: $cid"
    /c/Users/igallios/AppData/Local/anaconda3/envs/fl_pytorch_env_20250401/python.exe pytorch/client_pytorch.py --cid "$cid" &
done

# Wait for all background processes to complete
wait

echo "All clients (from $START_CID to $((START_CID + NUM_CLIENTS - 1))) have finished."

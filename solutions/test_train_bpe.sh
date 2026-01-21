#!/bin/bash

BASE_CMD="uv run train_bpe.py"

PARAMS=(
    "--dataset-name TinyStoriesV2-GPT4- --vocab-size 10000 --num-process 64"
    "--dataset-name owt_ --vocab-size 32000 --num-process 512"
)

for i in "${!PARAMS[@]}"; do
    echo "[ param ${i}: ${PARAMS[i]} ]"
    
    FULL_CMD="$BASE_CMD ${PARAMS[i]}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="test_train_bpe_${TIMESTAMP}_config_${i}.log"
    
    /usr/bin/time -v $FULL_CMD 2>&1 | tee $LOG_FILE
done
#!/bin/bash

CONFIG_FILE="codebase/hyperparameters.yml"

# Extract config names (lines ending with :)
configs=$(grep '^[a-zA-Z0-9_-]\+:' $CONFIG_FILE | sed 's/://')

for config in $configs
do
    python3 codebase/agent.py "$config" --train

done
#!/bin/bash

# The purpose of this script is to reset the access times of all files in the
# /models directory. The reason to do that is that the flist system within Zos
# checks the atime to decide if a file can be cleaned from the cache. By running
# this script periodically, we can ensure that the model files are never
# removed. 
# 
# In a fresh VM, the model files won't be loaded at all yet, so this script will
# cause Zos to download them in that case.

# Read first byte of every file in /models
if [ -d "/models" ]; then
    for model_file in /models/*; do
        if [ -f "$model_file" ]; then
            if ! head -c 1 "$model_file" >/dev/null 2>&1; then
                echo "Warning: Failed to read first byte of $model_file" >&2
            fi
        fi
    done
fi

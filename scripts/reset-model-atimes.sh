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
for dir in /models /.venv; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -exec head -c 1 {} \; -print0 2>/dev/null | while IFS= read -r -d '' file; do
            echo "Warning: Failed to read first byte of $file" >&2
        done
    fi
done

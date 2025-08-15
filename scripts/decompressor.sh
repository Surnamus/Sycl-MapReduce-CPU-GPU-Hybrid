#!/bin/bash

#fun fact ovo sam bozjom voljom napisao neznam sta se desava
PROJECT="$HOME/project"
COMPRESSED="$PROJECT/dataset/compressed"
UNCOMPRESSED="$PROJECT/dataset/uncompressed"

mkdir -p "$UNCOMPRESSED"

# Unzip all zip files, skip duplicates
while IFS= read -r -d '' zipfile; do
    echo "Unzipping: $zipfile"
    # List files in zip and extract only if they don't exist
    unzip -Z1 "$zipfile" | while read -r f; do
        if [ ! -e "$UNCOMPRESSED/$(basename "$f")" ]; then
            unzip -jq "$zipfile" "$f" -d "$UNCOMPRESSED"
        else
            echo "Skipping duplicate: $(basename "$f")"
        fi
    done
done < <(find "$COMPRESSED" -name "*.zip" -print0)

echo "All unique zip files extracted."

#!/bin/bash

PROJECT="$HOME/project"
COMPRESSED="$PROJECT/dataset/compressed"
UNCOMPRESSED="$PROJECT/dataset/uncompressed"

mkdir -p "$UNCOMPRESSED"

find "$COMPRESSED" -name "*.zip" -print0 | while IFS= read -r -d '' zipfile; do
    echo "Unzipping: $zipfile"
    unzip -q "$zipfile" -d "$UNCOMPRESSED"
done

# Optional: flatten (move files from subdirectories to uncompressed)
find "$UNCOMPRESSED" -mindepth 2 -type f -exec mv -t "$UNCOMPRESSED" {} +

# Optional: remove empty dirs created inside uncompressed
find "$UNCOMPRESSED" -type d -empty -delete

echo "✅ All zip files extracted into $UNCOMPRESSED."

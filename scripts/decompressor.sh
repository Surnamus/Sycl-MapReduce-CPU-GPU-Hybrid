#!/bin/bash

#fun fact ovo sam bozjom voljom napisao neznam sta se desava
PROJECT="$HOME/project"
COMPRESSED="$PROJECT/dataset/compressed"
UNCOMPRESSED="$PROJECT/dataset/uncompressed"

mkdir -p "$UNCOMPRESSED"

find "$COMPRESSED" -name "*.zip" -print0 | while IFS= read -r -d '' zipfile; do
    echo "Unzipping: $zipfile"
    unzip -q "$zipfile" -d "$UNCOMPRESSED"
done

find "$UNCOMPRESSED" -mindepth 2 -type f -exec mv -t "$UNCOMPRESSED" {} +

find "$UNCOMPRESSED" -type d -empty -delete

echo " All zip files extracted ."
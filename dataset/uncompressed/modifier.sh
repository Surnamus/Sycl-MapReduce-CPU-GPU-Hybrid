#!/bin/bash

# main project path
PROJECT="$HOME/project"
DATASET="$PROJECT/dataset"
UNCOMPRESSED="$DATASET/uncompressed"
MODIFIED="$DATASET/modified"

# create modified directory if it doesn't exist
mkdir -p "$MODIFIED"

# loop over .fna files in uncompressed
for file in "$UNCOMPRESSED"/*.fna; do
    # skip if no files found
    [ -e "$file" ] || continue

    # get base name without extension
    base=$(basename "$file" .fna)

    # target txt file
    output="$MODIFIED/$base.txt"

    # remove first line and save as .txt in modified
    tail -n +2 "$file" > "$output"

    echo "Processed: $file → $output"
done

echo "Done."

#!/bin/bash

PROJECT="$HOME/project"
DATASET="$PROJECT/dataset"
FOLDER="$DATASET/modified"

mkdir -p "$FOLDER"

N=50  # number of characters to keep

# truncate all .txt files in modified folder recursively
find "$FOLDER" -type f -name "*.txt" | while read -r file; do
    echo "Truncating '$file' to $N characters..."
    # read first N characters into a variable
    content=$(head -c $N "$file")
    # overwrite the file in-place
    printf "%s" "$content" > "$file"
done

echo "Done truncating all files."

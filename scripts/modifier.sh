#!/bin/bash
#iz .fna u .txt iako je .fna samo fancy text sa headerom
#lakse je samo gledati i pratiti .txt fajlove tbh
set -e

PROJECT="$HOME/project"
DATASET="$PROJECT/dataset"
UNCOMPRESSED="$DATASET/uncompressed"
MODIFIED="$DATASET/modified"

# Clean out old files and recreate folder
rm -rf "$MODIFIED"
mkdir -p "$MODIFIED"

find "$UNCOMPRESSED" -type f -name "*.fna" | while read -r file; do
    base=$(basename "$file" .fna)
    outfile="$MODIFIED/${base}.txt"
    echo "Modifying: $file → $outfile"
    grep -v "^>" "$file" | tr -d '\n\r' > "$outfile"
done

echo " Done."

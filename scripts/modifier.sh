#!/bin/bash
#iz .fna u .txt iako je .fna samo fancy text sa headerom
#lakse je samo gledati i pratiti .txt fajlove tbh
PROJECT="$HOME/project"
DATASET="$PROJECT/dataset"
UNCOMPRESSED="$DATASET/uncompressed"
MODIFIED="$DATASET/modified"

mkdir -p "$MODIFIED"

for file in "$UNCOMPRESSED"/*.fna; do
    [ -e "$file" ] || continue

    base=$(basename "$file" .fna)

    output="$MODIFIED/$base.txt"

    tail -n +2 "$file" > "$output"

    echo "Processed: $file → $output"
done

echo "Done."

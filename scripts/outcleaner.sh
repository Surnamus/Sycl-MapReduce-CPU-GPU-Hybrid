#!/bin/bash

# Files to clear
file1="/home/user/project/output.txt"
file2="/home/user/project/verifyme.txt"

# Truncate the files
truncate -s 0 "$file1" "$file2"

echo "Contents of $file1 and $file2 have been cleared."

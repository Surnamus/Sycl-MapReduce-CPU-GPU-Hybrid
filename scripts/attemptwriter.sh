#!/bin/bash
# scripts/attemptwriter.sh
# Usage: ./attemptwriter.sh N K LS BS DEV EXIT_STATUS

LOGDIR="$HOME/project/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/attempts.log"

# Arguments
N="$1"
K="$2"
LS="$3"
BS="$4"
DEV="$5"
STATUS="$6"

# Determine next attempt number by counting existing attempts in the log
if [ -f "$LOGFILE" ]; then
    attempt=$(( $(grep -c '^Attempt' "$LOGFILE") + 1 ))
else
    attempt=1
fi

# Log header for this attempt
echo "Attempt $attempt: N=$N, K=$K, LS=$LS, BS=$BS, DEV=$DEV" >> "$LOGFILE"

# Determine result
if [ "$STATUS" -eq 0 ]; then
    echo "Attempt $attempt: Success" >> "$LOGFILE"
else
    if [ "$STATUS" -ge 128 ]; then
        signal=$((STATUS - 128))
        echo "Attempt $attempt: Fail -syserror (signal $signal)" >> "$LOGFILE"
    else
        echo "Attempt $attempt: Fail -missing" >> "$LOGFILE"
    fi
fi

echo "--------------------------------" >> "$LOGFILE"

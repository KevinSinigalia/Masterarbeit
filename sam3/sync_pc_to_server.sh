#!/bin/bash

# Definition der Pfade
LOCAL_PATH="./"
REMOTE_USER="udsvr"
REMOTE_HOST="iai-hpc2.iai.kit.edu"
REMOTE_PATH="/srv/udsvr/Masterarbeit/sam3"

echo "---------------------------------------------------"
echo "Starte Synchronisation: Lokal -> Server (KIT HPC)"
echo "Ziel: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "---------------------------------------------------"

# Der eigentliche rsync Befehl
# --exclude: Diese Ordner/Dateien werden NICHT hochgeladen
rsync -avzP \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='venv/' \
    --exclude='.env' \
    --exclude='sam3' \
    "$LOCAL_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "---------------------------------------------------"
echo "Fertig! Dein Code ist jetzt auf dem Server."

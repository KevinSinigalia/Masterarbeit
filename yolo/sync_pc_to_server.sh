#!/bin/bash

LOCAL_PATH="./"
REMOTE_USER="udsvr"
REMOTE_HOST="iai-aida022.iai.kit.edu"
REMOTE_PATH="/srv/udsvr/Masterarbeit/yolo"

echo "---------------------------------------------------"
echo "Starte Synchronisation: Lokal -> Server (KIT)"
echo "Ziel: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "---------------------------------------------------"

# Synchronisiert ALLES ohne Excludes
rsync -avzP "$LOCAL_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "---------------------------------------------------"
echo "Fertig! Alle Dateien wurden uebertragen."

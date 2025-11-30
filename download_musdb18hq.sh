#!/bin/bash

# Check argument count
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

data_dir=$1
data_dir="$(realpath -s "$data_dir")"
echo "Data directory is set to $data_dir"

MUSDB_DIR="$data_dir/musdb18hq"
MUSDB_ZIP="$data_dir/musdb18hq.zip"
MUSDB_URL="https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"

# Detect OS and available download tool
if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget -O"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl -L -o"
else
    echo "[Error] Neither wget nor curl found on this system. Please install one."
    exit 1
fi

echo "Using download command: $DOWNLOAD_CMD"

# Check if data already exists
if [ -d "$MUSDB_DIR" ]; then
    echo "[Warning] Skip download of MUSDB18HQ. The directory $MUSDB_DIR already exists."
else

    # Download ZIP if needed
    if [ -f "$MUSDB_ZIP" ]; then
        echo "[Warning] $MUSDB_ZIP already exists. Skip download."
    else
        echo "Downloading MUSDB18HQ dataset to $MUSDB_ZIP"
        $DOWNLOAD_CMD "$MUSDB_ZIP" "$MUSDB_URL"
    fi

    # Create target directory
    mkdir -p "$MUSDB_DIR"

    echo "Extracting MUSDB18HQ into $MUSDB_DIR"

    # Count files
    n_files=$(unzip -l "$MUSDB_ZIP" | tail -n 1 | xargs echo -n | cut -d' ' -f2)

    i=0
    unzip "$MUSDB_ZIP" -d "$MUSDB_DIR" | while IFS= read -r line; do
        if [[ "$line" == *"inflating:"* ]]; then
            i=$((i+1))
            printf "\rExtracted %d / %d files..." "$i" "$n_files"
        fi
    done

    # Remove ZIP after extraction
    rm "$MUSDB_ZIP"
fi

echo "Done."

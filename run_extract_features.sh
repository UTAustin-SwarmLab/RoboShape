#!/bin/bash
# Script to run save_encoder_features.py with correct arguments

# Directory containing scene folders (ensure this points to the correct location)
DATA_DIR="/home/obiwan/mirac/point/scannet_data_manual/train"

# Output directory for features
OUTPUT_DIR="/home/obiwan/mirac/sonata/encoder_features"

echo "Starting feature extraction..."
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"

python save_encoder_features.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

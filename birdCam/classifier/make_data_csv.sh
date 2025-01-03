#!/bin/bash

# ================================
# Script to Generate CSV and Remove Small JPEG Files
# Skips and deletes JPEG files smaller than 1KB
# ================================

# Define the CSV file path
CSV_FILE="bird_data.csv"

# Define the CSV file header
echo "class id,filepaths,labels,data set,scientific name" > "$CSV_FILE"

# Define class IDs and labels
declare -A classes
# Example: classes=( ["sparrow"]=0 ["tit"]=1 ["robin"]=2 )
classes=( ["sparrow"]=0 ["tit"]=1 ["robin"]=2 )

# Base data directory
BASE_DIR=~/data/birds2024

# Log file for deleted files
DELETION_LOG="deleted_files.log"

# Initialize the deletion log
echo "List of deleted files - $(date)" > "$DELETION_LOG"

# Function to delete small files and log them
delete_small_files() {
    local directory="$1"
    echo "Deleting JPEG files smaller than 1KB in: $directory"

    # Find all .jpeg and .jpg files (case-insensitive) smaller than 1024 bytes
    find "$directory" -type f \( -iname "*.jpeg" -o -iname "*.jpg" \) -size -1024c | while read -r filepath; do
        if [ -f "$filepath" ]; then
            echo "Deleting: $filepath"
            echo "$filepath" >> "$DELETION_LOG"
            rm "$filepath"
            # Check if deletion was successful
            if [ $? -eq 0 ]; then
                echo "Successfully deleted: $filepath"
            else
                echo "Failed to delete: $filepath" >&2
            fi
        fi
    done
}

# Function to process and append file info to CSV
process_files() {
    local directory="$1"
    local class_id="$2"
    local label="$3"
    local dataset="$4"
    local scientific_name="$5"

    echo "Processing JPEG files in: $directory"

    # Find all .jpeg and .jpg files (case-insensitive) larger than 1KB
    find "$directory" -type f \( -iname "*.jpeg" -o -iname "*.jpg" \) -size +1024c | while read -r filepath; do
        echo "$class_id,$filepath,$label,$dataset,$scientific_name" >> "$CSV_FILE"
    done
}

# Loop through 'train' and 'test' datasets
for dataset in train test; do
    echo "Processing dataset: $dataset"
    for label in "${!classes[@]}"; do
        class_id=${classes[$label]}
        scientific_name="$label"

        # Define the directory path for the current dataset and label
        dir="$BASE_DIR/$dataset/$label"

        # Check if the directory exists
        if [ -d "$dir" ]; then
            # Step 1: Delete JPEG files smaller than 1KB and log them
            delete_small_files "$dir"

            # Step 2: Process remaining JPEG files larger than 1KB and append to CSV
            process_files "$dir" "$class_id" "$label" "$dataset" "$scientific_name"
        else
            echo "Warning: Directory does not exist - $dir"
        fi
    done
done

echo "CSV generation and cleanup completed successfully."
echo "Deleted files are listed in: $DELETION_LOG"

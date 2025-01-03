#!/bin/bash

# Directories
output_dir=~/birds
train_dir=~/data/birds2024/train
test_dir=~/data/birds2024/test

# Create the necessary directories
mkdir -p "$train_dir/sparrow" "$train_dir/tit" "$test_dir/sparrow" "$test_dir/tit" "$train_dir/robin" "$test_dir/robin"

# Loop over sparrow and tit directories
for bird in sparrow tit robin; do
    # Get all images in the current bird directory
    images=("$output_dir/$bird"/*.jpeg)  # Replace .jpg with your image format

    # Shuffle the images randomly
    shuffle_array() {
        local i j tmp
        for ((i = ${#array[*]}-1; i > 0; i--)); do
            j=$((RANDOM % (i+1)))
            tmp=${array[$i]}
            array[$i]=${array[$j]}
            array[$j]=$tmp
        done
    }
    shuffle_array images

    # Split into train and test sets (10% test, 90% train)
    test_count=$(( ${#images[*]} / 10 ))
    train_count=$(( ${#images[*]} - test_count ))

    # Copy images to train and test directories
    for ((i=0; i<test_count; i++)); do
        cp "${images[$i]}" "$test_dir/$bird"
    done

    for ((i=test_count; i<${#images[*]}; i++)); do
        cp "${images[$i]}" "$train_dir/$bird"
    done
done

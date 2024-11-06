#!/bin/bash

# Step 1: Clean up previous builds
echo "Cleaning up..."
rm -rf ./bench
rm -f template template.o

# Step 2: Compile the CUDA program using mcc
echo "Compiling vector_add.mu..."
mcc -o template vector_add.mu -lmusart -L/usr/local/musa/lib 

# Step 3: Run the dataset
echo "Running the dataset..."
bash run_dataset

echo "Done!"

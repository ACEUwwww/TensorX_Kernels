#!/bin/bash

# Step 1: Clean up previous builds
echo "Cleaning up..."
rm -f template template.o

# Step 2: Compile the CUDA program using mcc
echo "Compiling matrix multiplication.mu..."
mcc -o template matrix_multiplication_half.mu -lmusart -L/usr/local/musa/lib --offload-arch=mp_22
./template
echo "Done!"

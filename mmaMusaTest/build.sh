rm -rf ./template
mcc -o template mma_musa.mu -lmusart --offload-arch=mp_22 -L/usr/local/musa/lib 
./template
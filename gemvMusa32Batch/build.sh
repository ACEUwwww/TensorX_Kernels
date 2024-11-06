rm -rf ./template
mcc -o template gemv_32.mu -lmusart --offload-arch=mp_22 -L/usr/local/musa/lib 
./template
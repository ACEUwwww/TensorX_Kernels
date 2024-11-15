rm -rf ./gemv_32batch_asm.o ./gemv_32batch_asm.out ./gemv_32batch_dst.o ./gemv_32batch_output.s ./out
python3 ../asm_render/asm_render.py ./gemv_32batch.s gemv_32batch_output.s --cbody body.mu --arch QY2 -d '{"trans_a":true, "trans_b":false, "thread_per_block":128,"io_dtype":"float","offline_reorder":false,"oline_reorder":true,"large_load":true,"tile_m":128,"tile_n":127}'
/usr/local/musa/bin/llvm-mc --arch=mtgpu --filetype=obj -mcpu=mp_22 gemv_32batch_output.s -o gemv_32batch_asm.o
/usr/local/musa/bin/lld -flavor gnu --no-undefined -shared -plugin /usr/local/bin/../lib/LLVMgold.so -plugin-opt=mcpu=mp_22 -plugin-opt=O0 gemv_32batch_asm.o -o gemv_32batch_asm.out
mcc --musa-path=/usr/local/musa -mtgpu --cuda-host-only -c host.mu -o gemv_32batch_dst.o -Xclang -fcuda-include-gpubinary -Xclang gemv_32batch_asm.out
mcc --musa-path=/usr/local/musa gemv_32batch_dst.o -o out -lmusart
./out
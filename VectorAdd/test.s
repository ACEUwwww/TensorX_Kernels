	.text
	.protected	_Z6vecAddPfS_S_i
	.globl	_Z6vecAddPfS_S_i
	.p2align	8
	.type	_Z6vecAddPfS_S_i,@function
_Z6vecAddPfS_S_i:
		[N-sD---]		FOP.MOV R1, SH23
		[N-wD--o]		DMA.LD.B32 AR1, SH[16:17], R1, 4, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD---]		INT.BYP R[1:2].s64, AR0.s32
		[NswD---]		BIT.ROT.LSR AR0, SH24, R1
		[Nw-D---]		BIT.LSL.OR R2, R2, C45, AR0
		[Ns-D---]		BIT.ROT.LSL R1, C45, R1
		[N--D---]		FOP.MOV R3, SH14
		[N-sD---]		FOP.MOV R4, SH15
		[NwsD---]		INT.BYP I[0:1].s64, R[1:2].s64
		[NswD---]		INT.ADD R[1:2].s64, R[3:4].s64, I[0:1].s64
		[NwsD---]		DMA.ST.B32 AR1, R[1:2], _, 4, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-wD---]		CTRL.END
.Lfunc_end0:
	.size	_Z6vecAddPfS_S_i, .Lfunc_end0-_Z6vecAddPfS_S_i

	.addrsig
	.mtgpu_metadata.start
---
mtgpu.kernels:
  - .arch_id:        220
    .args:
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         14
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         16
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         18
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         20
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           threadIdx.x
          .offset:         0
          .reg_bank:       2
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           1
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           threadIdx.y
          .offset:         1
          .reg_bank:       2
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           2
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           threadIdx.z
          .offset:         1
          .reg_bank:       2
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           3
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Global Spill Base
          .offset:         0
          .reg_bank:       3
          .size:           8
          .type:           18
        .input_mode:     0
        .kind:           19
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Core Number
          .offset:         2
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           23
    .async_bar_count: 0
    .attr_reg_count: 2
    .attrs:          {}
    .barrier_coef:   4294967295
    .clusterX:       0
    .clusterY:       0
    .clusterZ:       0
    .coef_reg_count: 0
    .const_calc_kernel: !str ''
    .constcalc_attr_reg_count: 0
    .constcalc_temp_reg_count: 0
    .global_atomic_count: 0
    .has_barrier:    false
    .has_cluster_barrier: false
    .has_sqmma_conv: false
    .imm_info:
      .offset:         21
      .reg_bank:       3
      .size:           16
      .values:
        - 65536
        - 262144
        - 20
        - 30
    .indirect_call_flag: false
    .internal_reg_count: 2
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           _Z6vecAddPfS_S_i
    .private_memory_size: 0
    .shared_memory_size: 20
    .shared_reg_count: 25
    .slot_reg_count: 0
    .temp_reg_count: 5
    .wave_mode:      0
mtgpu.version:
  - 1
  - 0
...

	.mtgpu_metadata.end

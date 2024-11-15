
	.text
	.protected	gemv_32_batch
	.globl	gemv_32_batch
	.p2align	8
	.type	gemv_32_batch,@function
gemv_32_batch:




; Program start
; Get R2
[N-sD---] LDMA.RD.B32.BST1      R2, _, _, 16, BURST_STRIDE=1
[NswD---] FOP.MOV               R0, AR0
[NwsD---] BIT.LUT.AND           R6, SH32, R0
[N-wD---] BIT.ROT.LSR           R7, C59, R0

; Prologue =====================================================================================
; Fill slot robust register
; buffer_size = (R2 + 1) * ((SH13 / 32) * 1024) * 2bytes
[N--D---] FOP.MOV               SL0, SH8
[N--D---] FOP.MOV               SL1, SH9
[N-sD---] FOP.MOV               R1, SH13
[NswD---] BIT.AND.LSL           R1, C79, SH45, R1
[N-sD---] INT.ADD               R41.s32, C33.s32, R2.s32
[NwwD---] INT.MUL               SL2.s32, R41.s32, R1.s32

; rowA = R6 / 4
; R43 = R6 % 4 * 8 + R7 * 32
; ldg_offset_in_A = rowA * K + R43
; *(float4*)R[32:35] = *(float4*)&SH[4:6][ldg_offset_in_A]; 
; ldg_offset_in_B = (bx * (SH13 / 32) + R7) * 1024 + R6 * 8
; *(float4*)&RegisterB = *(float4*)&SH[8:10][ldg_offset_in_B];
[N-sD---] BIT.ROT.LSR           R42, C45, R6
[NswD---] BIT                   R43, SH37, SH43, R0, C0, lut_prog=58596, sh_op=LSL, msk_op=BYP, rot_op=LSR, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2                    
[NwsD---] INT.MADD              R45.s32, SH13.s32, R43.s32, R42.s32
[NswD---] INT.MUL               R45.s32, C45.s32, R45.s32
[Nw-Ds--] DMA.LD.B128           R[32:35], SH[4:5], R45, _, _, _, _, SH[4:6]

[N-sD---] BIT.ROT.LSR           R1, C80, SH13
[NswD---] INT.MADD              R1.s32, R2.s32, R7.s32, R1.s32
[NwsD---] INT.MUL               R41.s32, C39.s32, R6.s32 
[NswD---] INT.MADD              R46.s32, SH38.s32, R41.s32, R1.s32      
[Nw-Ds--] DMA.LD.B128           R[16:19], SH[8:9], R46, _, _, _, _, SL[0:2]

; R40 = 304 * (R6 / 64) + (R6 % 4) * 64 + (R6 % 64) / 4 * 4 + f(R6 % 4) + R7 * 608
; abcdefgh -> ghcdef00 :  R1 = Rotate_left(((R6 << 26) | R6) & 0xF0000003, 6)
[NwsD---] BIT.ROT.LSR           R1, C79, R6
[NswD---] INT.MUL               R40.s32, SH34.s32, R1.s32
[NwsD---] BIT                   R1, SH39, SH42, R6, C0, lut_prog=43176, sh_op=LSL, msk_op=BYP, rot_op=ROL, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2
[NswD---] INT.ADD               R40.s32, R1.s32, R40.s32
[NwsD---] BIT.LUT.AND           R1, C50, R6
[NswD---] INT.ADD               R41.s32, C33.s32, R1.s32
[NwsD---] INT.MUL               R1.s32, R1.s32, R41.s32
[NswD---] INT.MADD              R40.s32, C61.s32, R40.s32, R1.s32  
[Nw-D---] INT.MADD              R40.s32, SH35.s32, R40.s32, R7.s32

; share_idx = R6 * 2 + R6 / 64 * 16 + R7 * 1216
; share_idx2 = R6 * 2 + R6 / 64 * 48 + R7 * 1216
[Ns-D---] INT.MADD              R41.s32, SH35.s32, R6.s32, R7.s32
[N-sD---] BIT.ROT.LSR           R1, C79, R6
[NwwD---] INT.MADD              R50.s32, C46.s32, R41.s32, R1.s32
[NwwD---] INT.MADD              R51.s32, C48.s32, R41.s32, R1.s32

; R[52:59] is to reduce the 
[Nw-D---] FOP.MOV               R24, C0
[N--D---] FOP.MOV               R25, C0
[N--D---] FOP.MOV               R26, C0
[N--D---] FOP.MOV               R27, C0
[N--D---] FOP.MOV               R28, C0
[N--D---] FOP.MOV               R29, C0
[N--D---] FOP.MOV               R30, C0
[N-sD---] FOP.MOV               R31, C0
[N-wD---] FOP.MOV               R52, C0
[N--D---] FOP.MOV               R53, C0
[N--D---] FOP.MOV               R54, C0
[N--D---] FOP.MOV               R55, C0
[N--D---] FOP.MOV               R56, C0
[N--D---] FOP.MOV               R57, C0
[N--D---] FOP.MOV               R58, C0
[Ns-D---] FOP.MOV               R59, C0

; Loop set
; Hotloop ====================================================================================
[NwsD---] BIT.ROT.LSR           R49, C59, SH13 
[NswD---] INT.ADD               R47.s32, SH33.s32, R45.s32
[N-sD---] INT.ADD               R48.s32, C35.s32, R46.s32

HOT_LOOP_START:
; Double Buffer Pang
[Nw-D-sw] DMA.LD.B128           R[36:39], SH[4:5], R47, _, _, _, _, SH[4:6]
[N-wDws-] DMA.LD.B128           R[20:23], SH[8:9], R48, _, _, _, _, SL[0:2]

; Load from share Ping  
[NwwD---] LDMA.WR.B128.BST1     R[32:35], R40, X4, 128, BURST_STRIDE=16
[N--D---] LDMA.RD.B32.BST2      R[8:9], R50, X4, 128, BURST_STRIDE=1216
[N-sD---] LDMA.RD.B32.BST2      R[10:11], R51, X4, 736, BURST_STRIDE=1216

; Compute Ping  
[N-wD---] MMA.323216            R52[4:1][2:4].f32, R16[2:1][1:1].f16.row, R8[2:1][1:1].f16.col, R52[4:1][2:4].f32
[Ns-D---] INT.ADD               R45.s32, SH33.s32, R47.s32
[N-sD---] INT.ADD               R46.s32, C35.s32, R48.s32
[N--D--s] MMA.323216            R24[4:1][2:4].f32, R18[2:1][1:1].f16.row, R10[2:1][1:1].f16.col, R24[4:1][2:4].f32

; Double Buffers Ping
[Nw-Ds-w] DMA.LD.B128           R[32:35], SH[4:5], R45, _, _, _, _, SH[4:6]
[N-wDsw-] DMA.LD.B128           R[16:19], SH[8:9], R46, _, _, _, _, SL[0:2]

; Load from share Pang
[N--D---] LDMA.WR.B128.BST1     R[36:39], R40, X4, 4992, BURST_STRIDE=16
[N--D---] LDMA.RD.B32.BST2      R[12:13], R50, X4, 4992, BURST_STRIDE=1216
[N-sD---] LDMA.RD.B32.BST2      R[14:15], R51, X4, 5600, BURST_STRIDE=1216

; Compute Pang  
[N-wD---] MMA.323216            R52[4:1][2:4].f32, R20[2:1][1:1].f16.row, R12[2:1][1:1].f16.col, R52[4:1][2:4].f32
[Ns-D---] INT.ADD               R47.s32, SH33.s32, R45.s32
[N-sD---] INT.ADD               R48.s32, C35.s32, R46.s32
[N--D--s] MMA.323216            R24[4:1][2:4].f32, R22[2:1][1:1].f16.row, R14[2:1][1:1].f16.col, R24[4:1][2:4].f32

; Loop End  
[NswD---] INT.ADD               R49.s32, C33.neg.s32, R49.s32
[NwsD---] INT.BYP.TG            p0, R49.s32, C0.s32
[N-wD---] CTRL.BR.ALLP0T        HOT_LOOP_START

; Epilogue ====================================================================================
; Add across two registerC
[N-sDw-w] INT.BYP.TE            p0, R7.s32, C0.s32
[Ns-D---] INT.MUL               R1.s32, C46.s32, R6.s32
[N--D---] FOP.ADD               R24, R52, R24
[N--D---] FOP.ADD               R25, R53, R25
[N--D---] FOP.ADD               R26, R54, R26
[N--D---] FOP.ADD               R27, R55, R27
[N--D---] FOP.ADD               R28, R56, R28
[N--D---] FOP.ADD               R29, R57, R29
[N--D---] FOP.ADD               R30, R58, R30
[N-sD---] FOP.ADD               R31, R59, R31

; Reduce Sum Part
[NwwDo--] !P0 LDMA.WR.B128.BST2 R[24:31], R1, X4, 128, BURST_STRIDE=16
[N--D---] AP.BARRIER            LM0

[N-sD---] P0 LDMA.RD.B128.BST2 R[8:15], R1, X4, 128, BURST_STRIDE=16
[N-wD---] P0 FOP.ADD            R17.l, R8, R24
[N--D---] P0 FOP.ADD            R17.h, R9, R25
[N--D---] P0 FOP.ADD            R18.l, R10, R26
[N--D---] P0 FOP.ADD            R18.h, R11, R27
[N--D---] P0 FOP.ADD            R19.l, R12, R28
[N--D---] P0 FOP.ADD            R19.h, R13, R29
[N--D---] P0 FOP.ADD            R20.l, R14, R30
[N-sD---] P0 FOP.ADD            R20.h, R15, R31
[N-wD---] AP.BARRIER            LM0

; Interleave shuffle inside warps for memory coalescing
[N-sD---] P0 BIT                R16, SH40, SH41, R6, C0, lut_prog=43176, sh_op=LSL, msk_op=BYP, rot_op=ROL, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2
[N-wD---] P0 SHUFFLE.IDX        R21, R17, R16, 12319
[N--D---] P0 SHUFFLE.IDX        R22, R18, R16, 12319
[N--D---] P0 SHUFFLE.IDX        R23, R19, R16, 12319
[N-sD---] P0 SHUFFLE.IDX        R24, R20, R16, 12319

; Store
; R25 = (R6 % 64) / 4
; R26 = (R6 / 64) * 16 + R6 % 4 * 4 + bx * 32
; 0abcdefg -> 000afg00 
; 00000000 0abcdefg
; 11111111 11000000
; 00000abc defg0000
[N-wD---] P0 BIT.AND.LSR        R25, C45, SH31, R6
[N-sD---] P0 BIT                R26, SH36, SH44, R6, C0, lut_prog=58596, sh_op=LSL, msk_op=BYP, rot_op=LSR, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2 
[NswD---] P0 INT.MADD           R26.s32, SH30.s32, R26.s32, R2.s32
[NwsD---] P0 INT.MADD           R27.s32, SH14.s32, R26.s32, R25.s32
[NswD---] P0 INT.MUL            R27.s32, C45.s32, R27.s32
[Nw-Ds--] P0 DMA.ST.B64         R[21:22], SH[0:1], R27, _, _, _, _, SH[0:2]
[NwsD---] P0 FOP.MOV            R1, SH30
[NswD---] P0 INT.MADD           R28.s32, SH14.s32, R27.s32, R1.s32
[Nw-Do--] P0 DMA.ST.B64         R[23:24], SH[0:1], R28, _, _, _, _, SH[0:2]

; Program end   
[NwwDwww] CTRL.NOP  
[N--D---] AP.BARRIER            LM0
[N--D---] CTRL.END

.Lfunc_end0:
	.size	gemv_32_batch, .Lfunc_end0-gemv_32_batch

    .type	llvm.mtgpu.kernel.gemv_32_batch.sm,@object
    .local	llvm.mtgpu.kernel.gemv_32_batch.sm
    .comm	llvm.mtgpu.kernel.gemv_32_batch.sm,9728,4

	.addrsig
	.mtgpu_metadata.start
---
mtgpu.kernels:
  - .arch_id:        220
    .args:
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         0
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         2
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         3
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         4
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         6
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         7
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         8
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         10
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         11
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         12
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         13
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         14
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
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
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         18
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         19
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_addr
          .offset:         20
          .reg_bank:       3
          .size:           8
          .type:           10
        .input_mode:     0
        .kind:           33
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_size
          .offset:         22
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           34
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Global Spill Base
          .offset:         23
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
          .offset:         25
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           23
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_x
          .offset:         26
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           13
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_y
          .offset:         27
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           14
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_z
          .offset:         28
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           15
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockDim_x
          .offset:         29
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           9
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_x
          .offset:         4
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           5
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_y
          .offset:         5
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           6
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_z
          .offset:         6
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           7
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           threadIdx_x
          .offset:         0
          .reg_bank:       2
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           1
    .async_bar_count: 0
    .attr_reg_count: 1
    .attrs:          {}
    .barrier_coef:   0
    .clusterX:       0
    .clusterY:       0
    .clusterZ:       0
    .coef_reg_count: 0
    .const_calc_kernel: !str ''
    .constcalc_temp_reg_count: 0
    .global_atomic_count: 0
    .has_barrier:    true
    .has_cluster_barrier: false
    .has_texture: false
    .has_sqmma_conv:    false
    .imm_info:
      .offset:         30
      .reg_bank:       3
      .size:           64
      .values:
        - 32
        - 63
        - 127
        - 128
        - 304
        - 608
        - 1026
        - 1282
        - 2048
        - 6662
        - 6915
        - 3758096387
        - 4026531843
        - 4294967168
        - 4294967232
        - 4294967264
    .indirect_call_flag: false
    .internal_reg_count: 0
    .local_mem_used_in_cluster: 0
    .name:           gemv_32_batch
    .private_memory_size: 0
    .shared_memory_size: 9856
    .shared_reg_count: 46
    .slot_reg_count: 3
    .temp_reg_count: 60
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

	.mtgpu_metadata.end

    
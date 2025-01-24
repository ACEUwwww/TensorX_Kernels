<kernel_name>
gemv_32_batch
</kernel_name>

<params>
C, robust*
size_C, int
_0, int
A, robust*
size_A, int
_1, int
B, robust*
size_B, int
_2, int
_M, int 
_K, int 
_N, int
debug_output, robust*
sizeDebug, int
_3, int
</params>

<smem>
9728
</smem>

<regs>
0-1  : tx, compute_temp
2-3  : bidx, bidy
2-3  : bidxy<0-1>
4-5  : debug_offset, debug_temp
6-7  : lane_idx, warp_idx
8-11 : RegisterA_ping<0-3>
12-15: RegisterA_pang<0-3>
16-19: RegisterB_ping<0-3>
20-23: RegisterB_pang<0-3>
24-31: RegisterC<0-7>
32-35: sts_temp_ping<0-3>
36-39: sts_temp_pang<0-3>
40   : sts_offset_share
41   : compute_temp2
42-44: row, colA, colB
45   : ldg_offset_in_A_ping
46   : ldg_offset_in_B_ping
47   : ldg_offset_in_A_pang
48   : ldg_offset_in_B_pang
49   : slice_k
50   : lds_share_idx
51   : lds_share_idx2
52-59: RegisterC_2<0-7>
8-15 : reduce_sum<0-7>
16   : shuffle_idx
17-20: halfRegisterC<0-3>
21-24: halfRegisterC_store<0-3>
25-26: rowC, colC
27   : stg_offset_in_C_1
28   : stg_offset_in_C_2
s:0-2: robust_reg_B<0-2>
</regs>

; Program start
; Get bidx
[N-sD---] LDMA.RD.B32.BST1      bidx, _, _, 16, BURST_STRIDE=1
[NswD---] FOP.MOV               tx, threadIdx_x
[NwsD---] BIT.LUT.AND           lane_idx, ConstI127, tx
[N-wD---] BIT.ROT.LSR           warp_idx, ConstI7, tx

; Prologue =====================================================================================
; Fill slot robust register
; buffer_size = (bidx + 1) * ((_K / 32) * 1024) * 2bytes
[N--D---] FOP.MOV               robust_reg_B[0], B[0]
[N--D---] FOP.MOV               robust_reg_B[1], B[1]
[N-sD---] FOP.MOV               compute_temp, _K
[NswD---] BIT.AND.LSL           compute_temp, ConstI6, ConstXFFFFFFE0, compute_temp
[N-sD---] INT.ADD               compute_temp2.s32, ConstI1.s32, bidx.s32
[NwwD---] INT.MUL               robust_reg_B[2].s32, compute_temp2.s32, compute_temp.s32

; rowA = lane_idx / 4
; colA = lane_idx % 4 * 8 + warp_idx * 32
; ldg_offset_in_A = rowA * K + colA
; *(float4*)sts_temp_ping = *(float4*)&A[ldg_offset_in_A]; 
; ldg_offset_in_B = (bx * (_K / 32) + warp_idx) * 1024 + lane_idx * 8
; *(float4*)&RegisterB[0] = *(float4*)&B[ldg_offset_in_B];
[N-sD---] BIT.ROT.LSR           row, ConstI2, lane_idx
[NswD---] BIT                   colA, ConstX0502, ConstXFFFFFF80, tx, C0, lut_prog=58596, sh_op=LSL, msk_op=BYP, rot_op=LSR, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2                    
[NwsD---] INT.MADD              ldg_offset_in_A_ping.s32, _K.s32, colA.s32, row.s32
[NswD---] INT.MUL               ldg_offset_in_A_ping.s32, ConstI2.s32, ldg_offset_in_A_ping.s32
[Nw-Ds--] DMA.LD.B128           sts_temp_ping[0:3], A[0:1], ldg_offset_in_A_ping, _, _, _, _, A[0:2]

[N-sD---] BIT.ROT.LSR           compute_temp, ConstI5, _K
[NswD---] INT.MADD              compute_temp.s32, bidx.s32, warp_idx.s32, compute_temp.s32
[NwsD---] INT.MUL               compute_temp2.s32, ConstI16.s32, lane_idx.s32 
[NswD---] INT.MADD              ldg_offset_in_B_ping.s32, ConstI2048.s32, compute_temp2.s32, compute_temp.s32      
[Nw-Ds--] DMA.LD.B128           RegisterB_ping[0:3], B[0:1], ldg_offset_in_B_ping, _, _, _, _, robust_reg_B[0:2]

; sts_offset_share = 304 * (lane_idx / 64) + (lane_idx % 4) * 64 + (lane_idx % 64) / 4 * 4 + f(lane_idx % 4) + warp_idx * 608
; abcdefgh -> ghcdef00 :  compute_temp = Rotate_left(((lane_idx << 26) | lane_idx) & 0xF0000003, 6)
[NwsD---] BIT.ROT.LSR           compute_temp, ConstI6, lane_idx
[NswD---] INT.MUL               sts_offset_share.s32, ConstI304.s32, compute_temp.s32
[NwsD---] BIT                   compute_temp, ConstX00001A06, ConstXF0000003, lane_idx, C0, lut_prog=43176, sh_op=LSL, msk_op=BYP, rot_op=ROL, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2
[NswD---] INT.ADD               sts_offset_share.s32, compute_temp.s32, sts_offset_share.s32
[NwsD---] BIT.LUT.AND           compute_temp, ConstI3, lane_idx
[NswD---] INT.ADD               compute_temp2.s32, ConstI1.s32, compute_temp.s32
[NwsD---] INT.MUL               compute_temp.s32, compute_temp.s32, compute_temp2.s32
[NswD---] INT.MADD              sts_offset_share.s32, ConstI4.s32, sts_offset_share.s32, compute_temp.s32  
[Nw-D---] INT.MADD              sts_offset_share.s32, ConstI608.s32, sts_offset_share.s32, warp_idx.s32

; share_idx = lane_idx * 2 + lane_idx / 64 * 16 + warp_idx * 1216
; share_idx2 = lane_idx * 2 + lane_idx / 64 * 48 + warp_idx * 1216
[Ns-D---] INT.MADD              compute_temp2.s32, ConstI608.s32, lane_idx.s32, warp_idx.s32
[N-sD---] BIT.ROT.LSR           compute_temp, ConstI6, lane_idx
[NwwD---] INT.MADD              lds_share_idx.s32, ConstI8.s32, compute_temp2.s32, compute_temp.s32
[NwwD---] INT.MADD              lds_share_idx2.s32, ConstI24.s32, compute_temp2.s32, compute_temp.s32

; RegisterC_2 is to reduce the 
[Nw-D---] FOP.MOV               RegisterC[0], ConstF0
[N--D---] FOP.MOV               RegisterC[1], ConstF0
[N--D---] FOP.MOV               RegisterC[2], ConstF0
[N--D---] FOP.MOV               RegisterC[3], ConstF0
[N--D---] FOP.MOV               RegisterC[4], ConstF0
[N--D---] FOP.MOV               RegisterC[5], ConstF0
[N--D---] FOP.MOV               RegisterC[6], ConstF0
[N-sD---] FOP.MOV               RegisterC[7], ConstF0
[N-wD---] FOP.MOV               RegisterC_2[0], ConstF0
[N--D---] FOP.MOV               RegisterC_2[1], ConstF0
[N--D---] FOP.MOV               RegisterC_2[2], ConstF0
[N--D---] FOP.MOV               RegisterC_2[3], ConstF0
[N--D---] FOP.MOV               RegisterC_2[4], ConstF0
[N--D---] FOP.MOV               RegisterC_2[5], ConstF0
[N--D---] FOP.MOV               RegisterC_2[6], ConstF0
[Ns-D---] FOP.MOV               RegisterC_2[7], ConstF0

; Loop set
; Hotloop ====================================================================================
[NwsD---] BIT.ROT.LSR           slice_k, ConstI7, _K 
[NswD---] INT.ADD               ldg_offset_in_A_pang.s32, ConstI128.s32, ldg_offset_in_A_ping.s32
[N-sD---] INT.ADD               ldg_offset_in_B_pang.s32, ConstI4096.s32, ldg_offset_in_B_ping.s32

HOT_LOOP_START:
; Double Buffer Pang
[Nw-D-sw] DMA.LD.B128           sts_temp_pang[0:3], A[0:1], ldg_offset_in_A_pang, _, _, _, _, A[0:2]
[N-wDws-] DMA.LD.B128           RegisterB_pang[0:3], B[0:1], ldg_offset_in_B_pang, _, _, _, _, robust_reg_B[0:2]

; Load from share Ping  
[NwwD---] LDMA.WR.B128.BST1     sts_temp_ping[0:3], sts_offset_share, X4, 128, BURST_STRIDE=16
[N--D---] LDMA.RD.B32.BST2      RegisterA_ping[0:1], lds_share_idx, X4, 128, BURST_STRIDE=1216
[N-sD---] LDMA.RD.B32.BST2      RegisterA_ping[2:3], lds_share_idx2, X4, 736, BURST_STRIDE=1216

; Compute Ping  
[N-wD---] MMA.323216            RegisterC_2[4:1][2:4].f32, RegisterB_ping0[2:1][1:1].f16.row, RegisterA_ping0[2:1][1:1].f16.col, RegisterC_2[4:1][2:4].f32
[Ns-D---] INT.ADD               ldg_offset_in_A_ping.s32, ConstI128.s32, ldg_offset_in_A_pang.s32
[N-sD---] INT.ADD               ldg_offset_in_B_ping.s32, ConstI4096.s32, ldg_offset_in_B_pang.s32
[N--D--s] MMA.323216            RegisterC[4:1][2:4].f32, RegisterB_ping2[2:1][1:1].f16.row, RegisterA_ping2[2:1][1:1].f16.col, RegisterC[4:1][2:4].f32

; Double Buffers Ping
[Nw-Ds-w] DMA.LD.B128           sts_temp_ping[0:3], A[0:1], ldg_offset_in_A_ping, _, _, _, _, A[0:2]
[N-wDsw-] DMA.LD.B128           RegisterB_ping[0:3], B[0:1], ldg_offset_in_B_ping, _, _, _, _, robust_reg_B[0:2]

; Load from share Pang
[N--D---] LDMA.WR.B128.BST1     sts_temp_pang[0:3], sts_offset_share, X4, 4992, BURST_STRIDE=16
[N--D---] LDMA.RD.B32.BST2      RegisterA_pang[0:1], lds_share_idx, X4, 4992, BURST_STRIDE=1216
[N-sD---] LDMA.RD.B32.BST2      RegisterA_pang[2:3], lds_share_idx2, X4, 5600, BURST_STRIDE=1216

; Compute Pang  
[N-wD---] MMA.323216            RegisterC_2[4:1][2:4].f32, RegisterB_pang0[2:1][1:1].f16.row, RegisterA_pang0[2:1][1:1].f16.col, RegisterC_2[4:1][2:4].f32
[Ns-D---] INT.ADD               ldg_offset_in_A_pang.s32, ConstI128.s32, ldg_offset_in_A_ping.s32
[N-sD---] INT.ADD               ldg_offset_in_B_pang.s32, ConstI4096.s32, ldg_offset_in_B_ping.s32
[N--D--s] MMA.323216            RegisterC[4:1][2:4].f32, RegisterB_pang2[2:1][1:1].f16.row, RegisterA_pang2[2:1][1:1].f16.col, RegisterC[4:1][2:4].f32

; Loop End  
[NswD---] INT.ADD               slice_k.s32, ConstI1.neg.s32, slice_k.s32
[NwsD---] INT.BYP.TG            p0, slice_k.s32, ConstI0.s32
[N-wD---] CTRL.BR.ALLP0T        HOT_LOOP_START

; Epilogue ====================================================================================
; Add across two registerC
[N-sDw-w] INT.BYP.TE            p0, warp_idx.s32, ConstI0.s32
[Ns-D---] INT.MUL               compute_temp.s32, ConstI8.s32, lane_idx.s32
[N--D---] FOP.ADD               RegisterC[0], RegisterC_2[0], RegisterC[0]
[N--D---] FOP.ADD               RegisterC[1], RegisterC_2[1], RegisterC[1]
[N--D---] FOP.ADD               RegisterC[2], RegisterC_2[2], RegisterC[2]
[N--D---] FOP.ADD               RegisterC[3], RegisterC_2[3], RegisterC[3]
[N--D---] FOP.ADD               RegisterC[4], RegisterC_2[4], RegisterC[4]
[N--D---] FOP.ADD               RegisterC[5], RegisterC_2[5], RegisterC[5]
[N--D---] FOP.ADD               RegisterC[6], RegisterC_2[6], RegisterC[6]
[N-sD---] FOP.ADD               RegisterC[7], RegisterC_2[7], RegisterC[7]

; Reduce Sum Part
[NwwDo--] !P0 LDMA.WR.B128.BST2 RegisterC[0:7], compute_temp, X4, 128, BURST_STRIDE=16
[N--D---] AP.BARRIER            LM0

[N-sD---] P0 LDMA.RD.B128.BST2 reduce_sum[0:7], compute_temp, X4, 128, BURST_STRIDE=16
[N-wD---] P0 FOP.ADD            halfRegisterC[0].l, reduce_sum[0], RegisterC[0]
[N--D---] P0 FOP.ADD            halfRegisterC[0].h, reduce_sum[1], RegisterC[1]
[N--D---] P0 FOP.ADD            halfRegisterC[1].l, reduce_sum[2], RegisterC[2]
[N--D---] P0 FOP.ADD            halfRegisterC[1].h, reduce_sum[3], RegisterC[3]
[N--D---] P0 FOP.ADD            halfRegisterC[2].l, reduce_sum[4], RegisterC[4]
[N--D---] P0 FOP.ADD            halfRegisterC[2].h, reduce_sum[5], RegisterC[5]
[N--D---] P0 FOP.ADD            halfRegisterC[3].l, reduce_sum[6], RegisterC[6]
[N-sD---] P0 FOP.ADD            halfRegisterC[3].h, reduce_sum[7], RegisterC[7]
[N-wD---] AP.BARRIER            LM0

; Interleave shuffle inside warps for memory coalescing
[N-sD---] P0 BIT                shuffle_idx, ConstX00001B03, ConstXE0000003, lane_idx, C0, lut_prog=43176, sh_op=LSL, msk_op=BYP, rot_op=ROL, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2
[N-wD---] P0 SHUFFLE.IDX        halfRegisterC_store[0], halfRegisterC[0], shuffle_idx, 12319
[N--D---] P0 SHUFFLE.IDX        halfRegisterC_store[1], halfRegisterC[1], shuffle_idx, 12319
[N--D---] P0 SHUFFLE.IDX        halfRegisterC_store[2], halfRegisterC[2], shuffle_idx, 12319
[N-sD---] P0 SHUFFLE.IDX        halfRegisterC_store[3], halfRegisterC[3], shuffle_idx, 12319

; Store
; rowC = (lane_idx % 64) / 4
; colC = (lane_idx / 64) * 16 + lane_idx % 4 * 4 + bx * 32
; 0abcdefg -> 000afg00 
; 00000000 0abcdefg
; 11111111 11000000
; 00000abc defg0000
[N-wD---] P0 BIT.AND.LSR        rowC, ConstI2, ConstI63, lane_idx
[N-sD---] P0 BIT                colC, ConstX0402, ConstXFFFFFFC0, lane_idx, C0, lut_prog=58596, sh_op=LSL, msk_op=BYP, rot_op=LSR, msk_mlb_b=S1E0, sh_c=S0E1, sh_e=S2E0, lut_f=S2 
[NswD---] P0 INT.MADD           colC.s32, ConstI32.s32, colC.s32, bidx.s32
[NwsD---] P0 INT.MADD           stg_offset_in_C_1.s32, _N.s32, colC.s32, rowC.s32
[NswD---] P0 INT.MUL            stg_offset_in_C_1.s32, ConstI2.s32, stg_offset_in_C_1.s32
[Nw-Ds--] P0 DMA.ST.B64         halfRegisterC_store[0:1], C[0:1], stg_offset_in_C_1, _, _, _, _, C[0:2]
[NwsD---] P0 FOP.MOV            compute_temp, ConstI32
[NswD---] P0 INT.MADD           stg_offset_in_C_2.s32, _N.s32, stg_offset_in_C_1.s32, compute_temp.s32
[Nw-Do--] P0 DMA.ST.B64         halfRegisterC_store[2:3], C[0:1], stg_offset_in_C_2, _, _, _, _, C[0:2]

; Program end   
[NwwDwww] CTRL.NOP  
[N--D---] AP.BARRIER            LM0
[N--D---] CTRL.END

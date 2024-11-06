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
10000
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
42-43: rowA, colA
44-45: rowB, colB
46   : ldg_offset_in_A_ping
47   : ldg_offset_in_B_ping
48   : ldg_offset_in_A_pang
49   : ldg_offset_in_B_pang
50   : slice_k
51   : lds_share_idx
52   : lds_share_idx2
8-15 : reduce_sum<0-7>
16   : shuffle_idx
</regs>

<CODE>
# Only for QY2 because of LDMA
def debug_store(reg_name, target_bx, target_by, data_type):
  if data_type == "int":
    copy_temp_str = f'[NswD---]    INT.ADD debug_temp.s32, {reg_name}.s32, ConstI0.s32'
  else:
    copy_temp_str = f'[NswD---]    FOP.MUL debug_temp, {reg_name}, ConstF1'

  return f"""
[NwwDwww]    CTRL.NOP
[N--D--o]    LDMA.RD.B32.BST2  bidxy, _, _, 16, BURST_STRIDE=4
[Ns-D---]    INT.BYP.TE    p0, bidy.s32, ConstI{target_by}.s32
[NwsD---]    INT.BYP.TE.AP p0, bidx.s32, ConstI{target_bx}.s32
[N-wD---]    CTRL.BR.ALLP0F SKIP_DEBUG

[N-sD---]    BIT.ROT.LSL debug_offset, ConstI2, threadIdx_x
{copy_temp_str}
[NwwDo--]    DMA.ST.B32 debug_temp, debug_output[0:1], debug_offset

SKIP_DEBUG:
"""
</CODE>
; ; DEBUG INSERTION ============================================================================
; [NwwDwww] CTRL.NOP
; [N--D---] AP.BARRIER LM0
; [N--D--o] FOP.MUL compute_temp, ConstF1, RegisterC[1].h
; <CODE>
; code = debug_store("RegisterC[7]", 0, 0, "float")
; </CODE>
; ; DEBUG INSERTION ============================================================================


; Program start
; Get bidx
[Ns-D---]                   LDMA.RD.B32.BST1 bidx, _, _, 16, BURST_STRIDE=1

; Prologue =====================================================================================
; tx = threadIdx.x
; lane_idx = tx % 128
; warp_idx = tx / 128
; offsetSharePing = (lane_idx / 64) * 512 + (lane_idx % 4) * 128 + (lane_idx % 64) / 4 * 8 + f(lane_idx % 4) + (lane_idx / 64) * 96 + warp_idx * 1216
; offsetSharePang = offsetSharePing + 2432
; lds_share_idx = lane_idx * 2 + lane_idx / 64 * 16 + warp_idx * 1216
; lds_share_idx2 = lane_idx * 2 + lane_idx / 64 * 48 + warp_idx * 1216
[NwsD---] FOP.MOV           tx, threadIdx_x
[NswD---] BIT.LUT.AND       lane_idx, ConstI127, tx
[NwsD---] BIT.ROT.LSR       warp_idx, ConstI7, tx
[NswD---] BIT.ROT.LSR       compute_temp, ConstI6, lane_idx
[NwsD---] INT.MUL           sts_offset_share.s32, ConstI304.s32, compute_temp.s32
[NswD---] BIT.LUT.AND       compute_temp, ConstI3, lane_idx
[NwsD---] INT.MADD          sts_offset_share.s32, ConstI64.s32, sts_offset_share.s32, compute_temp.s32
[NswD---] BIT               compute_temp, ConstI2, ConstI63, lane_idx, ConstI0, lut_prog=41120, sh_op=BYP, msk_op=BYP, rot_op=LSR, msk_mlb_b=S1E0, sh_c=S0E0, sh_e=S0E0, lut_f=S2
[NwsD---] INT.MADD          sts_offset_share.s32, ConstI4.s32, sts_offset_share.s32, compute_temp.s32
[NswD---] BIT.LUT.AND       compute_temp, ConstI3, lane_idx
[NwsD---] INT.ADD           compute_temp2.s32, ConstI1.s32, compute_temp.s32
[NswD---] INT.MUL           compute_temp.s32, compute_temp.s32, compute_temp2.s32
[NwsD---] INT.MADD          sts_offset_share.s32, ConstI4.s32, sts_offset_share.s32, compute_temp.s32  
[NswD---] INT.MADD          sts_offset_share.s32, ConstI608.s32, sts_offset_share.s32, warp_idx.s32
[NwsD---] INT.MADD          compute_temp2.s32, ConstI608.s32, lane_idx.s32, warp_idx.s32
[NswD---] BIT.ROT.LSR       compute_temp, ConstI6, lane_idx
[NwsD---] INT.MADD          lds_share_idx.s32, ConstI8.s32, compute_temp2.s32, compute_temp.s32
[NswD---] INT.MADD          lds_share_idx2.s32, ConstI24.s32, compute_temp2.s32, compute_temp.s32
[Nw-D---] FOP.MOV           RegisterC[0], ConstF0
[N--D---] FOP.MOV           RegisterC[1], ConstF0
[N--D---] FOP.MOV           RegisterC[2], ConstF0
[N--D---] FOP.MOV           RegisterC[3], ConstF0
[N--D---] FOP.MOV           RegisterC[4], ConstF0
[N--D---] FOP.MOV           RegisterC[5], ConstF0
[N--D---] FOP.MOV           RegisterC[6], ConstF0
[N-sD---] FOP.MOV           RegisterC[7], ConstF0

; rowA = lane_idx / 4
; colA = lane_idx % 4 * 8 + warp_idx * 32
; ldg_offset_in_A = rowA * K + colA
; *(float4*)sts_temp_ping = *(float4*)&A[ldg_offset_in_A]; 
; rowB = lane_idx / 4 + warp_idx * 32
; colB = lane_idx % 4 * 8 + bidx * 32
; ldg_offset_in_B = rowB * _N + colB
; *(float4*)&RegisterB[0] = *(float4*)&B[ldg_offset_in_B];
[NswD---] BIT.ROT.LSR       rowA, ConstI2, lane_idx
[NwsD---] INT.MUL           colA.s32, ConstI32.s32, warp_idx.s32
[NswD---] BIT.LUT.AND       compute_temp, ConstI3, lane_idx
[NwsD---] INT.MADD          colA.s32, ConstI8.s32, colA.s32, compute_temp.s32
[NswD---] INT.MADD          ldg_offset_in_A_ping.s32, _K.s32, colA.s32, rowA.s32
[NwsD---] INT.MUL           ldg_offset_in_A_ping.s32, ConstI2.s32, ldg_offset_in_A_ping.s32
[N-wDs--] DMA.LD.B128       sts_temp_ping[0:3], A[0:1], ldg_offset_in_A_ping, _, _, _, _, A[0:2]

[Ns-D---] BIT.ROT.LSR       rowB, ConstI2, lane_idx
[NwsD---] INT.MADD          rowB.s32, ConstI32.s32, rowB.s32, warp_idx.s32
[NswD---] BIT.AND.LSL       colB, ConstI3, ConstI3, lane_idx
[NwsD---] INT.MADD          colB.s32, ConstI32.s32, colB.s32, bidx.s32
[NswD---] INT.MADD          ldg_offset_in_B_ping.s32, _N.s32, colB.s32, rowB.s32
[NwsD---] INT.MUL           ldg_offset_in_B_ping.s32, ConstI2.s32, ldg_offset_in_B_ping.s32
[N-wDs--] DMA.LD.B128       RegisterB_ping[0:3], B[0:1], ldg_offset_in_B_ping, _, _, _, _, B[0:2]

; Loop set
; Hotloop ====================================================================================
[Ns-D---] FOP.MOV           slice_k, ConstI0
[NwsD---] FOP.MOV           compute_temp, _K
[NswD---] INT.ADD           compute_temp.s32, ConstI63.s32, compute_temp.s32
[NwsD---] BIT.ROT.LSR       compute_temp, ConstI6, compute_temp
[NswD---] INT.ADD           compute_temp.s32, ConstI1.s32, compute_temp.s32
[NwsD---] BIT.ROT.LSR       slice_k, ConstI1, compute_temp

HOT_LOOP_START:
; Double Buffer Pang
[NswD---] INT.ADD           ldg_offset_in_A_pang.s32, ConstI128.s32, ldg_offset_in_A_ping.s32
[NwsD---] FOP.MOV           compute_temp, ConstI128
[NswD---] INT.MADD          ldg_offset_in_B_pang.s32, _N.s32, ldg_offset_in_B_ping.s32, compute_temp.s32
[Nw-D-s-] DMA.LD.B128       sts_temp_pang[0:3], A[0:1], ldg_offset_in_A_pang, _, _, _, _, A[0:2]
[N-wDws-] DMA.LD.B128       RegisterB_pang[0:3], B[0:1], ldg_offset_in_B_pang, _, _, _, _, B[0:2]

; Load from share Ping
[Ns-D---] LDMA.WR.B128.BST1 sts_temp_ping[0:3], sts_offset_share, X4, 128, BURST_STRIDE=16
[Nw-Ds--] LDMA.RD.B32.BST2  RegisterA_ping[0:1], lds_share_idx, X4, 128, BURST_STRIDE=1216
[N--Do--] LDMA.RD.B32.BST2  RegisterA_ping[2:3], lds_share_idx2, X4, 736, BURST_STRIDE=1216

; Compute Ping
[Ns-D---] MMA.323216        RegisterC[4:1][2:4].f32, RegisterB_ping0[2:1][1:1].f16.row, RegisterA_ping0[2:1][1:1].f16.col, RegisterC[4:1][2:4].f32
[NwsD---] MMA.323216        RegisterC[4:1][2:4].f32, RegisterB_ping2[2:1][1:1].f16.row, RegisterA_ping2[2:1][1:1].f16.col, RegisterC[4:1][2:4].f32

; Double buffer Ping
[NswD---] INT.ADD           ldg_offset_in_A_ping.s32, ConstI128.s32, ldg_offset_in_A_pang.s32
[NwsD---] INT.MADD          ldg_offset_in_B_ping.s32, _N.s32, ldg_offset_in_B_pang.s32, compute_temp.s32
[N-wDs--] DMA.LD.B128       sts_temp_ping[0:3], A[0:1], ldg_offset_in_A_ping, _, _, _, _, A[0:2]
[N-wDsw-] DMA.LD.B128       RegisterB_ping[0:3], B[0:1], ldg_offset_in_B_ping, _, _, _, _, B[0:2]

; Load from share Pang
[Ns-D---] LDMA.WR.B128.BST1 sts_temp_pang[0:3], sts_offset_share, X4, 4992, BURST_STRIDE=16
[Nw-D-s-] LDMA.RD.B32.BST2  RegisterA_pang[0:1], lds_share_idx, X4, 4992, BURST_STRIDE=1216
[N--D-o-] LDMA.RD.B32.BST2  RegisterA_pang[2:3], lds_share_idx2, X4, 5600, BURST_STRIDE=1216

; Compute Pang
[Ns-D---] MMA.323216        RegisterC[4:1][2:4].f32, RegisterB_pang0[2:1][1:1].f16.row, RegisterA_pang0[2:1][1:1].f16.col, RegisterC[4:1][2:4].f32
[NwsD---] MMA.323216        RegisterC[4:1][2:4].f32, RegisterB_pang2[2:1][1:1].f16.row, RegisterA_pang2[2:1][1:1].f16.col, RegisterC[4:1][2:4].f32

; Loop End
[NswD---] INT.ADD           slice_k.s32, ConstI1.neg.s32, slice_k.s32
[N--D---] AP.BARRIER        LM0
[NwsD---] INT.BYP.TG        p0, slice_k.s32, ConstI0.s32
[N-wD---] CTRL.BR.ALLP0T    HOT_LOOP_START

; Epilogue ====================================================================================
; Reduce Sum Part
[N-sD---] INT.BYP.TE        p0, warp_idx.s32, ConstI0.s32
[Ns-D---] INT.MUL           compute_temp.s32, ConstI8.s32, lane_idx.s32
[NwwDo--] !P0 LDMA.WR.B128.BST2 RegisterC[0:7], compute_temp, X4, 128, BURST_STRIDE=16
[N--D---] AP.BARRIER        LM0

[N-sD---] P0 LDMA.RD.B128.BST2 reduce_sum[0:7], compute_temp, X4, 128, BURST_STRIDE=16
[N-wD---] P0 FOP.ADD        RegisterC[0], reduce_sum[0], RegisterC[0]
[N--D---] P0 FOP.ADD        RegisterC[1], reduce_sum[1], RegisterC[1]
[N--D---] P0 FOP.ADD        RegisterC[2], reduce_sum[2], RegisterC[2]
[N--D---] P0 FOP.ADD        RegisterC[3], reduce_sum[3], RegisterC[3]
[N--D---] P0 FOP.ADD        RegisterC[4], reduce_sum[4], RegisterC[4]
[N--D---] P0 FOP.ADD        RegisterC[5], reduce_sum[5], RegisterC[5]
[N--D---] P0 FOP.ADD        RegisterC[6], reduce_sum[6], RegisterC[6]
[N-sD---] P0 FOP.ADD        RegisterC[7], reduce_sum[7], RegisterC[7]

; Store
[N--D---] BIT               shuffle_idx,                       

; Program end
[NwwDwww] CTRL.NOP
[N--D---] AP.BARRIER LM0
[N--D---] CTRL.END

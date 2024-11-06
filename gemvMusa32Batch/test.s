	.text
	.protected	_Z4gemvP6__halfiiS0_iiS0_iiiiiPfii
	.globl	_Z4gemvP6__halfiiS0_iiS0_iiiiiPfii
	.p2align	8
	.type	_Z4gemvP6__halfiiS0_iiS0_iiiiiPfii,@function
_Z4gemvP6__halfiiS0_iiS0_iiiiiPfii:
		[N-sD---]		FOP.MOV R11, AR0
		[NswD---]		BIT.ROT.ASR_INT32 AR0, C78, AR0
		[NwsD---]		BIT.ROT.LSR AR0, SH51, AR0
		[NswD---]		INT.ADD AR1.s32, R11.s32, AR0.s32
		[NwsD---]		BIT.LUT.AND AR0, AR1, SH54
		[NswD---]		INT.ADD R12.s32, AR0.neg.s32, R11.s32
		[NwsD---]		FOP.MOV R1, R12
		[NswD---]		INT.BYP AR0.s32, R1.s8
		[NwsD---]		BIT.ROT.LSR R2, SH53, AR0
		[NswD---]		INT.ADD R2.s32, AR0.s32, R2.s32
		[NwsD---]		BIT.LUT.AND R3, R2, SH45
		[NswD---]		INT.ADD R3.s8, R3.neg.s8, R1.s8
		[NwsD---]		INT.BYP R3.s32, R3.s8
		[N-wD---]		BIT.ROT.LSL R4, C50, R3
		[Ns-D---]		BIT.ROT.ASR_INT32 R5, C59, AR1
		[N--D---]		BIT.ROT.ASR_INT32 R2, C45, R2
		[NwsD---]		INT.MADD AR1.s32, SH43.s32, R4.s32, R5.s32
		[NswD---]		INT.MADD AR1.s32, SH27.s32, AR1.s32, R2.s32
		[NwsD---]		INT.BYP R[6:7].s64, AR1.s32
		[NswD---]		BIT.ROT.LSR AR1, C78, R6
		[N--D---]		BIT.ROT.LSL R8, C33, R6
		[NwsD---]		BIT.LSL.OR R9, R7, C33, AR1
		[N-wD---]		FOP.MOV R7, SH41
		[N--D---]		FOP.MOV R13, SH18
		[Ns-D---]		FOP.MOV R14, SH19
		[N--D--s]		AP.BARRIER LM4
		[NswD---]		INT.BYP I[0:1].s64, R[8:9].s64
		[NwsD---]		INT.ADD R[8:9].s64, R[13:14].s64, I[0:1].s64
		[N-wD-s-]		DMA.LD.B16 R6.l, R[8:9], R7, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD-w-]		BIT.ROT.LSR AR1, SH52, AR0
		[NswD---]		INT.ADD R7.s32, AR0.s32, AR1.s32
		[NwsD---]		BIT.LUT.AND R10, R7, SH44
		[NswD---]		FOP.MOV R13, R10
		[NwsD---]		INT.ADD R1.s8, R13.neg.s8, R1.s8
		[NswD---]		INT.BYP AR1.s32, R1.s8
		[Nw-D---]		BIT.ROT.LSR R1, SH53, AR1
		[N-sD---]		BIT.ROT.ASR_INT32 R13, C79, R7
		[NswD---]		INT.ADD R1.s32, AR1.s32, R1.s32
		[NswD---]		FOP.MOV R7, SH47
		[N-sD---]		INT.MUL R14.s32, SH59.s32, R5.s32
		[Nw-D---]		BIT.LSL.AND R1, R1, C33, R7
		[NswD---]		INT.MADD R7.s32, SH42.s32, R14.s32, R3.s32
		[NwsD---]		INT.MADD R7.s32, SH58.s32, R7.s32, R13.s32
		[N--D---]		INT.ADD R3.s32, R3.s32, C33.s32
		[NswD---]		INT.ADD R1.s32, R7.s32, R1.s32
		[NwsD---]		INT.MADD R1.s32, R4.s32, R1.s32, R3.s32
		[NswD---]		INT.MADD R3.s32, R1.s32, SH49.s32, C45.s32
		[NwsD---]		BIT.ROT.LSR R3, C33, R3
		[NwsD--w]		FOP.MOV R7, SH40
		[N-wD-o-]		MOV.ST.LM.B16 LM0[R3], R6.l
		[N-wDs--]		DMA.LD.B16 R3.l, R[8:9], R7, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sDw--]		FOP.MOV R6, SH39
		[N--D--o]		MOV.ST.LM.B16 LM8[R1], R3.l
		[N-wD--o]		DMA.LD.B16 R3.l, R[8:9], R6, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD---]		INT.MADD R6.s32, R1.s32, SH53.s32, C45.s32
		[NswD---]		BIT.ROT.LSR R6, C33, R6
		[N-sD---]		FOP.MOV R7, SH38
		[Nw-D--o]		MOV.ST.LM.B16 LM0[R6], R3.l
		[N-wD-s-]		DMA.LD.B16 R3.l, R[8:9], R7, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD-w-]		FOP.MOV R6, SH37
		[N--Do--]		MOV.ST.LM.B16 LM7[R1], R3.l
		[N-wDo--]		DMA.LD.B16 R3.l, R[8:9], R6, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD---]		INT.MADD R6.s32, R1.s32, SH52.s32, C45.s32
		[NswD---]		BIT.ROT.LSR R6, C33, R6
		[N-sD---]		FOP.MOV R7, SH36
		[Nw-Do--]		MOV.ST.LM.B16 LM0[R6], R3.l
		[N-wD--s]		DMA.LD.B16 R3.l, R[8:9], R7, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD--w]		FOP.MOV R6, SH35
		[N--D--o]		MOV.ST.LM.B16 LM6[R1], R3.l
		[N-wD--o]		DMA.LD.B16 R3.l, R[8:9], R6, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N-sD---]		INT.MADD R6.s32, R1.s32, SH48.s32, C45.s32
		[NswD---]		BIT.ROT.LSR R6, C33, R6
		[Nw-D--o]		MOV.ST.LM.B16 LM0[R6], R3.l
		[N--D-s-]		DMA.LD.B16 R3.l, R[8:9], _, 2, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[N--Ds--]		LDMA.RD.B32.BST1 R6, _, _, 0, BURST_STRIDE=4
		[N--Dw--]		INT.MADD R2.s32, SH43.s32, R2.s32, R5.s32
		[Ns-D---]		INT.MADD R4.s32, SH43.s32, R4.s32, R6.s32
		[NwsD---]		INT.MADD R2.s32, SH28.s32, R4.s32, R2.s32
		[NswD---]		INT.BYP R[4:5].s64, R2.s32
		[NwsD---]		BIT.ROT.LSR R2, C78, R4
		[N--D---]		BIT.ROT.LSL R4, C33, R4
		[NswD-w-]		BIT.LSL.OR R5, R5, C33, R2
		[N--Do--]		MOV.ST.LM.B16 LM5[R1], R3.l
		[Nw-D---]		FOP.MOV R1, SH22
		[Ns-D---]		FOP.MOV R2, SH23
		[N--Ds--]		AP.BARRIER LM4
		[NwsD---]		INT.BYP I[0:1].s64, R[4:5].s64
		[NswD---]		INT.ADD R[1:2].s64, R[1:2].s64, I[0:1].s64
		[Nw-D--s]		DMA.LD.B128 R[1:4], R[1:2], _, 16, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[Ns-D---]		INT.MADD R5.s32, R12.s32, R14.s32, C45.s32
		[NwsD---]		INT.MADD R5.s32, R13.s32, R5.s32, C39.s32
		[NswD---]		INT.MADD R6.s32, R5.s32, SH50.s32, C45.s32
		[Nw-D---]		INT.ADD R7.s32, R6.s32, R10.s32
		[N-sD---]		INT.MADD R5.s32, R5.s32, SH46.s32, C45.s32
		[N-wD---]		INT.ADD R8.s32, R7.s32, SH57.s32
		[Ns-D---]		BIT.ROT.ASR_INT32 R5, C45, R5
		[Nw-Dw--]		BIT.ROT.ASR_INT32 R8, C45, R8
		[Nw-D-o-]		LDMA.RD.B32.BST1 R10, R5, X4, 0, BURST_STRIDE=4
		[Ns-D---]		BIT.ROT.ASR_INT32 R5, C45, R6
		[N-sD---]		INT.ADD R6.s32, R7.s32, SH58.s32
		[Nw-D-o-]		LDMA.RD.B32.BST1 R9, R5, X4, 0, BURST_STRIDE=4
		[NswD---]		BIT.ROT.ASR_INT32 R5, C45, R6
		[Nw-Ds--]		LDMA.RD.B32.BST1 R6, R8, X4, 0, BURST_STRIDE=4
		[Nw-Ds-w]		LDMA.RD.B32.BST1 R5, R5, X4, 0, BURST_STRIDE=4
		[Ns-Dw--]		MMA.323216 R14[4:2][2:1].f32, R1[2:1][1:1].f16.row, R9[1:1][2:1].f16.col, 0
		[NwsD---]		MMA.323216 R1[4:2][2:1].f32, R3[2:1][1:1].f16.row, R5[1:1][2:1].f16.col, R14[4:2][2:1].f32
		[Nw-Do--]		LDMA.RD.B32.BST1 R9, _, _, 0, BURST_STRIDE=4
		[N-wD---]		BIT.ROT.LSL R9, C80, R9
		[N--D---]		BIT.ROT.LSL R10, C61, R13
		[N--Ds--]		AP.BARRIER LM4
		[N-sDw--]		INT.ADD R11.s32, R11.s32, SH56.s32
		[N--Ds--]		AP.BARRIER LM4
		[N-wDw--]		INT.BYP.TG p0, R11.u32, SH55.u32
		[N--Ds--]		AP.BARRIER LM4
		[N-sDw--]		BIT.ROT.LSL R12, C50, R12
		[NswD---]	EX	COND.ST.P0 1, 1
		[Nw-D---]		CTRL.BR.ALLPEF BB0_2
		[N-sD---]		INT.MADD R13.s32, R12.s32, C80.s32, C33.s32
		[N-wD--s]		LDMA.WR.B128.BST2 R[1:8], R13, X4, 0, BURST_STRIDE=16
BB0_2:
		[N--D--w]		CTRL.NOP
		[N-sD---]	EX	COND.END.ALWAYS 1
		[NswD---]		INT.BYP.TL.MOVC R13.u32, R11.u32, C34.u32, C36, C0
		[NwsD---]		INT.BYP.TNE p0, R13.s32, C0.s32
		[N-wD--s]		AP.BARRIER LM4
		[N--D--w]		CTRL.NOP
		[NswD---]	EX	COND.ST.P0 1, 0
		[Nw-D---]		CTRL.BR.ALLPEF BB0_4
		[N-sD---]		INT.MADD R12.s32, R12.s32, C80.s32, C33.s32
		[N-wD--o]		LDMA.RD.B128.BST2 R[12:19], R12, X4, 0, BURST_STRIDE=16
		[N--D---]		FOP.ADD R7, R18, R7
		[N--D---]		FOP.ADD R6, R17, R6
		[N--D---]		FOP.ADD R5, R16, R5
		[N--D---]		FOP.ADD R4, R15, R4
		[N--D---]		FOP.ADD R3, R14, R3
		[N--D---]		FOP.ADD R2, R13, R2
		[N-sD---]		FOP.ADD R1, R12, R1
		[Ns-D---]		FOP.ADD R8, R19, R8
BB0_4:
		[NwwD---]		CTRL.NOP
		[N-sD---]	EX	COND.END.ALWAYS 1
		[NswD---]		INT.BYP.TL.MOVC R11.u32, R11.u32, C34.u32, C36, C0
		[N-wD--s]		AP.BARRIER LM4
		[NwsD--w]		INT.BYP.TNE p0, R11.s32, C0.s32
		[N--D--s]		AP.BARRIER LM4
		[N--D--w]		CTRL.NOP
		[NswD---]	EX	COND.ST.P0 1, 0
		[Nw-D---]		CTRL.BR.ALLPEF BB0_6
		[N-sD---]		BIT.ROT.LSR R11, SH60, AR0
		[NswD---]		INT.ADD R11.s32, AR0.s32, R11.s32
		[NwsD---]		BIT.LUT.AND R11, R11, SH63
		[NswD---]		INT.ADD R11.s32, R11.neg.s32, AR0.s32
		[NwsD---]		BIT.ROT.LSR R12, SH61, R11
		[NswD---]		INT.ADD R11.s32, R11.s32, R12.s32
		[N-sD---]		BIT.ROT.LSR R12, SH61, AR0
		[Nw-D---]		BIT.ROT.ASR_INT32 R11, C50, R11
		[N-wD---]		INT.ADD R12.s32, AR0.s32, R12.s32
		[Ns-D---]		BIT.ROT.LSR R13, SH60, AR1
		[Nw-D---]		BIT.LUT.AND R12, R12, SH62
		[N-sD---]		INT.ADD AR1.s32, AR1.s32, R13.s32
		[N-wD---]		INT.ADD AR0.s32, R12.neg.s32, AR0.s32
		[Ns-D---]		BIT.ROT.ASR_INT32 AR1, C80, AR1
		[N-sD---]		INT.ADD R9.s32, R10.s32, R9.s32
		[Nw-D---]		INT.MADD AR0.s32, AR1.s32, AR0.s32, C46.s32
		[NswD---]		INT.MADD AR1.s32, R11.s32, R9.s32, C61.s32
		[NwsD---]		INT.MADD AR0.s32, SH28.s32, AR1.s32, AR0.s32
		[NswD---]		INT.BYP R[9:10].s64, AR0.s32
		[Nw-D---]		BIT.ROT.LSL AR1, C33, R9
		[Ns-D---]		FOP.MUL R4.l, R4, C1
		[N-sD---]		INT.MADD AR0.s32, SH28.s32, AR0.s32, C39.s32
		[Nw-D---]		INT.BYP R4.u32, R4.u16
		[NswD---]		INT.BYP R[11:12].s64, AR0.s32
		[Ns-D---]		FOP.MUL R2.l, R2, C1
		[Nw-D---]		BIT.ROT.LSR AR0, C78, R11
		[N-sD---]		INT.BYP R2.u32, R2.u16
		[N-wD---]		BIT.LSL.OR AR0, R12, C33, AR0
		[Ns-D---]		BIT.LSL.OR R12, R4, C39, R2
		[Nw-D---]		FOP.MOV R2, AR0
		[Ns-D---]		FOP.MUL R4.l, R7, C1
		[N-sD---]		FOP.MUL R7.l, R8, C1
		[Nw-D---]		INT.BYP AR0.u32, R4.u16
		[N-wD---]		INT.BYP R4.u32, R7.u16
		[N-sD---]		FOP.MUL R6.l, R6, C1
		[N-sD---]		BIT.ROT.LSR R7, C78, R9
		[NswD---]		INT.BYP R6.u32, R6.u16
		[N--D---]		BIT.LSL.OR R7, R10, C33, R7
		[NwsD---]		BIT.LSL.OR R13, R4, C39, R6
		[N-wD---]		FOP.MOV R6, AR1
		[N-sD---]		FOP.MUL R4.l, R5, C1
		[N--D---]		FOP.MOV R8, SH14
		[Ns-D---]		FOP.MOV R9, SH15
		[N-wD---]		INT.BYP AR1.u32, R4.u16
		[NwsD---]		INT.BYP I[0:1].s64, R[6:7].s64
		[N-wD---]		INT.ADD R[4:5].s64, R[8:9].s64, I[0:1].s64
		[N--D---]		BIT.LSL.OR AR1, AR0, C39, AR1
		[N-sD---]		FOP.MUL R3.l, R3, C1
		[Ns-D---]		FOP.MUL R1.l, R1, C1
		[N-wD---]		INT.BYP AR0.u32, R3.u16
		[NwsD---]		INT.BYP R1.u32, R1.u16
		[N--D---]		BIT.ROT.LSL R3, C33, R11
		[NswD---]		BIT.LSL.OR AR0, AR0, C39, R1
		[NwsD---]		FOP.MOV R1, R3
		[NwsD---]		DMA.ST.B64 AR[0:1], R[4:5], _, 8, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
		[NswD---]		INT.BYP I[0:1].s64, R[1:2].s64
		[N-wD---]		CTRL.NOP
		[NwsD---]		INT.ADD AR[0:1].s64, R[8:9].s64, I[0:1].s64
		[NswD---]		DMA.ST.B64 R[12:13], AR[0:1], _, 8, 1, 1, chrnt=l2_l3, slc=new, persist=0, stride_add_first=0
BB0_6:
		[Nw-D---]		CTRL.NOP
		[N-sD---]	EX	COND.END.ALWAYS 1
		[N-wD--s]		AP.BARRIER LM4
		[N--D--w]		CTRL.NOP
		[N--D---]		CTRL.END
.Lfunc_end0:
	.size	_Z4gemvP6__halfiiS0_iiS0_iiiiiPfii, .Lfunc_end0-_Z4gemvP6__halfiiS0_iiS0_iiiiiPfii

	.type	llvm.musa.kernel._Z4gemvP6__halfiiS0_iiS0_iiiiiPfii.sm,@object
	.local	llvm.musa.kernel._Z4gemvP6__halfiiS0_iiS0_iiiiiPfii.sm
	.comm	llvm.musa.kernel._Z4gemvP6__halfiiS0_iiS0_iiiiiPfii.sm,4864,1
	.addrsig
	.mtgpu_metadata.start
---
mtgpu.kernels:
  - .arch_id:        220
    .args:
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx.x
          .offset:         0
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
          .name:           blockIdx.y
          .offset:         1
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
          .name:           blockIdx.z
          .offset:         2
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           7
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
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         16
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         17
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
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
        .input_mode:     0
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
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         21
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         22
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
          .offset:         24
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         25
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         26
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         27
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
          .offset:         28
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          8
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         30
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
          .offset:         29
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     2
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           !str ''
          .offset:         32
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
    .barrier_coef:   4
    .clusterX:       0
    .clusterY:       0
    .clusterZ:       0
    .coef_reg_count: 0
    .const_calc_kernel: !str ''
    .constcalc_attr_reg_count: 0
    .constcalc_temp_reg_count: 0
    .global_atomic_count: 0
    .has_barrier:    true
    .has_cluster_barrier: false
    .has_sqmma_conv: false
    .has_texture:    false
    .imms:
      - .immBase:
          .offset:         33
          .value:          65536
      - .immBase:
          .offset:         34
          .value:          262144
      - .immBase:
          .offset:         35
          .value:          2
      - .immBase:
          .offset:         36
          .value:          4
      - .immBase:
          .offset:         37
          .value:          6
      - .immBase:
          .offset:         38
          .value:          8
      - .immBase:
          .offset:         39
          .value:          10
      - .immBase:
          .offset:         40
          .value:          12
      - .immBase:
          .offset:         41
          .value:          14
      - .immBase:
          .offset:         42
          .value:          128
      - .immBase:
          .offset:         43
          .value:          32
      - .immBase:
          .offset:         44
          .value:          4294967232
      - .immBase:
          .offset:         45
          .value:          252
      - .immBase:
          .offset:         46
          .value:          1236
      - .immBase:
          .offset:         47
          .value:          2147483640
      - .immBase:
          .offset:         48
          .value:          22
      - .immBase:
          .offset:         49
          .value:          34
      - .immBase:
          .offset:         50
          .value:          20
      - .immBase:
          .offset:         51
          .value:          25
      - .immBase:
          .offset:         52
          .value:          26
      - .immBase:
          .offset:         53
          .value:          30
      - .immBase:
          .offset:         54
          .value:          4294967168
      - .immBase:
          .offset:         55
          .value:          254
      - .immBase:
          .offset:         56
          .value:          127
      - .immBase:
          .offset:         57
          .value:          1824
      - .immBase:
          .offset:         58
          .value:          608
      - .immBase:
          .offset:         59
          .value:          1216
      - .immBase:
          .offset:         60
          .value:          27
      - .immBase:
          .offset:         61
          .value:          29
      - .immBase:
          .offset:         62
          .value:          4294967288
      - .immBase:
          .offset:         63
          .value:          4294967264
    .indirect_call_flag: false
    .internal_reg_count: 2
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           _Z4gemvP6__halfiiS0_iiS0_iiiiiPfii
    .private_memory_size: 0
    .shared_atomic_lock_idx: 4294967295
    .shared_memory_size: 4884
    .shared_reg_count: 64
    .slot_reg_count: 0
    .temp_reg_count: 22
    .wave_mode:      0
mtgpu.version:
  - 1
  - 0
...

	.mtgpu_metadata.end

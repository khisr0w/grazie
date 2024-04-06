/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/21/2023 10:49:09 PM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "autograd.h"

internal inline bool
IsStackBlocksEmpty(stack_blocks_state *State) { return State->RunningTensorNum == 0; }

internal inline stack_block *
AllocNewStackBlock(size_t MaxNumTensors, stack_block *BelowBlock) {
    stack_block *Result = (stack_block *)Malloc(MaxNumTensors*sizeof(t32 *) + sizeof(stack_block));
    Assert(Result, "failed to allocate stack block");
    Result->TensorPtr = (t32 **)(Result+1);
    Result->MaxNumTen = MaxNumTensors;
    Result->BelowBlock = BelowBlock;

    return Result;
}

internal inline void
StackBlockPush(stack_blocks_state *State, t32 *Tensor) {
    if(!State->CurrentBlock || State->CurrentBlock->MaxNumTen ==
                               State->CurrentBlockTopIdx+1) {
        /* NOTE(Abid): In case, we do not have any blocks, or if the block is full */
        stack_block *NewBlock = NULL;
        /* NOTE(Abid): If we got one in reserve, use that */
        if(State->ReservedBlock) {
            NewBlock = State->ReservedBlock;
            State->ReservedBlock = NULL;
            NewBlock->BelowBlock = State->CurrentBlock;
        } else {
            /* TODO(Abid): Tweak the NewAllocTensorNum growth factor to something reasonable */
            State->NewAllocTensorNum += (i32)(State->NewAllocTensorNum/2);
            NewBlock = AllocNewStackBlock(State->NewAllocTensorNum, State->CurrentBlock);
        }
        State->CurrentBlock = NewBlock;
        State->CurrentBlockTopIdx = (size_t)-1;
    }
    State->CurrentBlock->TensorPtr[++State->CurrentBlockTopIdx] = Tensor;
    /* NOTE(Abid): Find the maximum number of tensors in the stack throughout all the runs of the same computation chain. */
    ++State->RunningTensorNum;
    if(State->RunningTensorNum > State->GlobalMaxTensorNum) State->GlobalMaxTensorNum = State->RunningTensorNum;
}

internal inline void
StackBlockPop(stack_blocks_state *State) {
    Assert(State->RunningTensorNum > 0, "cannot pop empty stack");
    --State->CurrentBlockTopIdx;
    --State->RunningTensorNum;

    /* NOTE(Abid): Reserve if block is empty, while freeing the already reserved block. */
    if(State->CurrentBlockTopIdx == -1) {
        stack_block *BelowBlock = State->CurrentBlock->BelowBlock;

        /* NOTE(Abid): Hold onto one extra block in case we push right after freeing a block. */
        if(State->ReservedBlock) Free(State->ReservedBlock);
        State->ReservedBlock = State->CurrentBlock;

        State->CurrentBlock = BelowBlock;
        if(State->CurrentBlock) State->CurrentBlockTopIdx = State->CurrentBlock->MaxNumTen-1;
    }
}

internal inline t32 *
StackBlockTop(stack_blocks_state *State) {
    Assert(State->RunningTensorNum > 0, "cannot get top of an empty stack");
    return State->CurrentBlock->TensorPtr[State->CurrentBlockTopIdx];
}

/* ============================================
 * NOTE(Abid): Backward Operations
 * ============================================ */

#define __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, OP) \
    f32 TempVal = Value; \
    f32 *StoragePtr = (f32 *)A->Grad.Ptr; \
    for(i32 Idx = 0; Idx < A->Header->StorageNumElements; ++Idx) \
        StoragePtr[Idx] OP TempVal;
internal inline void
__BackwardT32AddToElements(t32 *A, f32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, +=); }

internal inline void
__BackwardT32SubToElements(t32 *A, f32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, -=); }

internal inline void
__BackwardT32MulToElements(t32 *A, f32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, *=); }

internal inline void
__BackwardT32DivToElements(t32 *A, f32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, /=); }

internal inline void
__BackwardT32SetElements(t32 *A, f32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, =); }
#undef __BACKWARD_OP_ELEMENTS_SCALAR

/* NOTE(Abid): The following operation does a reduce sum on the broadcast dims of A and add/sub/mul/div
 *             it to Result tensor. There is a faster way to do it, but that most likely requires a
 *             malloc which I shall never do. */
#define __REDUCE_BROADCAST_DIMS(A, Result, OP) \
    Assert(A->Header->Dim >= Result->Header->Dim, "result tensor cannot be higher than operand"); \
    size_t ANumData = A->Header->StorageNumElements; \
    \
    size_t ResultOffset = 0; \
    size_t AOffset = 0; \
    for(size_t OpNum = 1; OpNum <= ANumData; ++OpNum) { \
        *((f32 *)Result->Grad.Ptr + ResultOffset) OP *((f32 *)A->Grad.Ptr + AOffset); \
                                                        \
        i32 DimMaxNumSoFar = 1; \
        for(i32 DimIdx = 1; DimIdx <= (i32)A->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= A->Header->Sizes[A->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                AOffset -= A->Header->Strides[A->Header->Dim-DimIdx]*(A->Header->Sizes[A->Header->Dim-DimIdx]-1); \
                if((i32)Result->Header->Dim - DimIdx < 0) ResultOffset = 0; \
                else ResultOffset -= Result->Header->Strides[Result->Header->Dim-DimIdx]*(Result->Header->Sizes[Result->Header->Dim-DimIdx]-1); \
                continue; \
            } \
            AOffset += A->Header->Strides[A->Header->Dim-DimIdx]; \
            if((i32)Result->Header->Dim - DimIdx >= 0) ResultOffset += Result->Header->Strides[Result->Header->Dim-DimIdx]; \
            break; \
        } \
    }
internal void 
__BackwardT32ReduceAddBroadcast(t32 *A, t32 *Result) { __REDUCE_BROADCAST_DIMS(A, Result, +=); }
internal void 
__BackwardT32ReduceSubBroadcast(t32 *A, t32 *Result) { __REDUCE_BROADCAST_DIMS(A, Result, -=); }
#undef __REDUCE_BROADCAST_DIMS

#define __BACKWARD_BINARY_MUL(OtherOperand, Parent, ResultOperand, OTHER_OPERAND_DTYPE) \
    Assert(Parent->Header->Dim >= ResultOperand->Header->Dim, "operand tensor dim cannot be higher than parent"); \
    Assert(Parent->Header->Dim >= OtherOperand->Header->Dim, "operand tensor dim cannot be higher than parent"); \
    \
    size_t ResultOffset = 0; \
    size_t ParentOffset = 0; \
    size_t OtherOperandOffset = 0; \
    size_t ParentNumData = Parent->Header->StorageNumElements; \
    for(size_t OpNum = 1; OpNum <= ParentNumData; ++OpNum) { \
        *((f32 *)ResultOperand->Grad.Ptr + ResultOffset) += *((OTHER_OPERAND_DTYPE *)OtherOperand->Data.Ptr + OtherOperandOffset) * \
                                                                *((f32 *)Parent->Grad.Ptr + ParentOffset); \
        i32 DimMaxNumSoFar = 1; \
        /* NOTE(Abid): If we have reached the end of the current dim in ResultOperand */ \
        for(i32 DimIdx = 1; DimIdx <= (i32)Parent->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= Parent->Header->Sizes[Parent->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                ParentOffset -= Parent->Header->Strides[Parent->Header->Dim-DimIdx] * \
                                (Parent->Header->Sizes[Parent->Header->Dim-DimIdx]-1); \
                if((i32)ResultOperand->Header->Dim - DimIdx < 0) ResultOffset = 0; \
                else ResultOffset -= ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]* \
                                     (ResultOperand->Header->Sizes[ResultOperand->Header->Dim-DimIdx]-1); \
                if((i32)OtherOperand->Header->Dim - DimIdx < 0) OtherOperandOffset = 0; \
                else OtherOperandOffset -= OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]* \
                                           (OtherOperand->Header->Sizes[OtherOperand->Header->Dim-DimIdx]-1); \
                continue; \
            } \
            ParentOffset += Parent->Header->Strides[Parent->Header->Dim-DimIdx]; \
            if((i32)ResultOperand->Header->Dim - DimIdx >= 0) ResultOffset += ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]; \
            if((i32)OtherOperand->Header->Dim - DimIdx >= 0) OtherOperandOffset += OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]; \
            break; \
        } \
    }
internal void 
__BackwardT32Mul(t32 *OtherOperand, t32 *Parent, t32 *ResultOperand) {
    switch (OtherOperand->Data.DType) {
        case dtype_f32: { __BACKWARD_BINARY_MUL(OtherOperand, Parent, ResultOperand, f32); } break;
        case dtype_i32: { __BACKWARD_BINARY_MUL(OtherOperand, Parent, ResultOperand, i32); } break;
    }
}
#undef __BACKWARD_BINARY_MUL

#define __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, OTHER_OPERAND_DTYPE) \
    Assert(Parent->Header->Dim >= ResultOperand->Header->Dim, "operand tensor dim cannot be higher than parent"); \
    Assert(Parent->Header->Dim >= OtherOperand->Header->Dim, "operand tensor dim cannot be higher than parent"); \
    \
    size_t ResultOffset = 0; \
    size_t ParentOffset = 0; \
    size_t OtherOperandOffset = 0; \
    size_t ParentNumData = Parent->Header->StorageNumElements; \
    for(size_t OpNum = 1; OpNum <= ParentNumData; ++OpNum) { \
        f32 ParentGrad = *((f32 *)Parent->Grad.Ptr + ParentOffset); \
        OTHER_OPERAND_DTYPE OtherOperandData = *((OTHER_OPERAND_DTYPE *)OtherOperand->Data.Ptr + OtherOperandOffset); \
        __BACKWARD_BINARY_DIV_OPERAND_OP; \
        i32 DimMaxNumSoFar = 1; \
        for(i32 DimIdx = 1; DimIdx <= (i32)Parent->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= Parent->Header->Sizes[Parent->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                ParentOffset -= Parent->Header->Strides[Parent->Header->Dim-DimIdx]*(Parent->Header->Sizes[Parent->Header->Dim-DimIdx]-1); \
                if((i32)ResultOperand->Header->Dim - DimIdx < 0) ResultOffset = 0; \
                else ResultOffset -= ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]* \
                                     (ResultOperand->Header->Sizes[ResultOperand->Header->Dim-DimIdx]-1); \
                if((i32)OtherOperand->Header->Dim - DimIdx < 0) OtherOperandOffset = 0; \
                else OtherOperandOffset -= OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]* \
                                           (OtherOperand->Header->Sizes[OtherOperand->Header->Dim-DimIdx]-1); \
                continue; \
            } \
            ParentOffset += Parent->Header->Strides[Parent->Header->Dim-DimIdx]; \
            if((i32)ResultOperand->Header->Dim - DimIdx >= 0) ResultOffset += ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]; \
            if((i32)OtherOperand->Header->Dim - DimIdx >= 0) OtherOperandOffset += OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]; \
            break; \
        } \
    }
internal void
__BackwardT32Div(t32 *OtherOperand, t32 *Parent, t32 *ResultOperand, u32 OperandIdx) {

    if(OperandIdx == 0) { /* First Operand */
        #define __BACKWARD_BINARY_DIV_OPERAND_OP \
            *((f32 *)ResultOperand->Grad.Ptr + ResultOffset) += (1.f / OtherOperandData) * ParentGrad

        switch (OtherOperand->Data.DType) {
            case dtype_f32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, f32); } break;
            case dtype_i32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, i32); } break;
        }
        #undef __BACKWARD_BINARY_DIV_OPERAND_OP
    } else { /* Second Operand */
        #define __BACKWARD_BINARY_DIV_OPERAND_OP \
            f32 SelfData = *((f32 *)ResultOperand->Data.Ptr + ResultOffset);\
            *((f32 *)ResultOperand->Grad.Ptr + ResultOffset) += (-OtherOperandData / (SelfData*SelfData)) * ParentGrad

        switch (OtherOperand->Data.DType) {
            case dtype_f32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, f32); } break;
            case dtype_i32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, i32); } break;
        }
        #undef __BACKWARD_BINARY_DIV_OPERAND_OP
    }
}
#undef __BACKWARD_BINARY_DIV

internal inline void
__ExpandVectorDim(t32 *A, u32 Pos) {
    Assert(A->Header->Dim == 1, "non-vector tensors given");
    ++A->Header->Dim;
    A->Header->Sizes[1-Pos] = A->Header->Sizes[0];
    A->Header->Strides[1-Pos] = A->Header->Strides[0];
    A->Header->Sizes[Pos] = 1;
    A->Header->Strides[Pos] = 0;
}

internal inline void
__SqueezeMatrixToVectorDim(t32 *A) {
    Assert(A->Header->Dim == 2, "must be matrix (dim==2)");
    Assert((A->Header->Sizes[0] == 1) || (A->Header->Sizes[1] == 1), "at least one dimension size must be 1");

    if(A->Header->Sizes[0] == 1) {
        A->Header->Sizes[0] = A->Header->Sizes[1];
        A->Header->Strides[0] = A->Header->Strides[1];
    }
    A->Header->Dim = 1;
}

internal void
__BackwardT32MatMul(t32 **Operands, t32 *Parent, u32 OperandIdx) {
    t32 *OtherOperand = Operands[1-OperandIdx];
    t32 *ResOper = Operands[OperandIdx];

    /* NOTE(Abid): If dim=1, expand by a dim to make it a matrix.
     *             Ergo, we shall never have a vector (dim != 1) */
    u32 OrigOperand0Dim = Operands[0]->Header->Dim;
    u32 OrigOperand1Dim = Operands[1]->Header->Dim;
    u32 OrigParentDim = Parent->Header->Dim;
    if(Operands[0]->Header->Dim == 1) {
        __ExpandVectorDim(Operands[0], 0);
        if(Parent->Header->Dim == 1) __ExpandVectorDim(Parent, 0);
    }
    if(Operands[1]->Header->Dim == 1) {
        __ExpandVectorDim(Operands[1], 1);
        if(Parent->Header->Dim == 1) __ExpandVectorDim(Parent, 1);
    }

    /* NOTE(Abid): Temporarily transpose the OtherOperand */
    __T32TransposeInPlaceNoGrad(OtherOperand, OtherOperand->Header->Dim-1, OtherOperand->Header->Dim-2);

    /* NOTE(Abid): Defining the first and second operand, as well as their values based on OperandIdx. */
    t32 *FirstOper = NULL;
    void *FirstOperValPtr = NULL;
    t32 *SecOper = NULL;
    void *SecOperValPtr = NULL;
    if(OperandIdx == 1) {
        FirstOper = OtherOperand;
        FirstOperValPtr = OtherOperand->Data.Ptr;
        SecOper = Parent;
        SecOperValPtr = Parent->Grad.Ptr;

    } else {
        FirstOper = Parent;
        FirstOperValPtr = Parent->Grad.Ptr;
        SecOper = OtherOperand;
        SecOperValPtr = OtherOperand->Data.Ptr;
    }

    /* NOTE(Abid): Total number of broadcasts carried out during the forward process. */
    u32 NumOfBroadcastOps = 1;
    for(u32 Idx = 2; Idx < Parent->Header->Dim; ++Idx) NumOfBroadcastOps *= GetSizeR(Parent, Idx);

    size_t ResultOffset = 0;
    size_t FirstOffset = 0;
    size_t SecondOffset = 0;
    u32 ReducedDimSize = GetSizeR(FirstOper, 0);

    u32 TotalMatMulOps = NumOfBroadcastOps * GetSizeR(ResOper, 0) * GetSizeR(ResOper, 1);
    for(size_t OpNum = 1; OpNum <= TotalMatMulOps; ++OpNum) {
        for(u32 ReduceDimIdx = 0; ReduceDimIdx < ReducedDimSize; ++ReduceDimIdx) {
            *((f32 *)ResOper->Grad.Ptr + ResultOffset) += *((f32 *)FirstOperValPtr + FirstOffset) *
                                                                    *((f32 *)SecOperValPtr + SecondOffset);
            FirstOffset += GetStrideR(FirstOper, 0);
            SecondOffset += GetStrideR(SecOper, 1);
        }
        FirstOffset -= GetStrideR(FirstOper, 0) * GetSizeR(FirstOper, 0);
        SecondOffset -= GetStrideR(SecOper, 1) * GetSizeR(SecOper, 1);

        u32 DimMaxNumSoFar = GetSizeR(ResOper, 0);

        /* NOTE(Abid): In case we've not reached end of dim=1 */
        if(OpNum % DimMaxNumSoFar != 0) {
            SecondOffset += GetStrideR(SecOper, 0);
            ResultOffset += GetStrideR(ResOper, 0);
            continue;
        } else {
            SecondOffset -= GetStrideR(SecOper, 0) * (GetSizeR(SecOper, 0)-1);
            ResultOffset -= GetStrideR(ResOper, 0) * (GetSizeR(ResOper, 0)-1);
        }
        DimMaxNumSoFar *= GetSizeR(ResOper, 1);

        /* NOTE(Abid): In case we've not reached end of dim=2 */
        if(OpNum % DimMaxNumSoFar != 0) {
            FirstOffset += GetStrideR(FirstOper, 1);
            ResultOffset += GetStrideR(ResOper, 1);
            continue;
        } else {
            FirstOffset -= GetStrideR(FirstOper, 1) * (GetSizeR(FirstOper, 1)-1);
            ResultOffset -= GetStrideR(ResOper, 1) * (GetSizeR(ResOper, 1)-1);
        }

        /* NOTE(Abid): We are in the broadcast dimensions now. Parent tensor is the one with
         *             the greatest possible dim, therefore, we loop through that */
        for(u32 DimIdx = 2; DimIdx < Parent->Header->Dim; ++DimIdx) {
            DimMaxNumSoFar *= GetSizeR(Parent, DimIdx);

            /* NOTE(Abid): Check if we've reached the end of this dim */
            if(OpNum % DimMaxNumSoFar == 0) {
                if(ResOper->Header->Dim-DimIdx > 0) {
                    ResultOffset -= GetStrideR(ResOper, DimIdx) * (GetSizeR(ResOper, DimIdx)-1);
                } else ResultOffset = 0;

                if(FirstOper->Header->Dim-DimIdx > 0) {
                    FirstOffset -= GetStrideR(FirstOper, DimIdx) * (GetSizeR(FirstOper, DimIdx)-1);
                } else FirstOffset = 0;

                if(SecOper->Header->Dim-DimIdx > 0) {
                    SecondOffset -= GetStrideR(SecOper, DimIdx) * (GetSizeR(SecOper, DimIdx)-1);
                } else SecondOffset = 0;
            } else {
                /* NOTE(Abid): In case we've not reached the end of this dim */
                if(ResOper->Header->Dim-DimIdx > 0) ResultOffset += GetStrideR(ResOper, DimIdx);
                else ResultOffset = 0;

                if(FirstOper->Header->Dim-DimIdx > 0) FirstOffset += GetStrideR(FirstOper, DimIdx);
                else FirstOffset = 0;

                if(SecOper->Header->Dim-DimIdx > 0) SecondOffset += GetStrideR(SecOper, DimIdx);
                else SecondOffset = 0;
                break;
            }
        }
    }

    /* NOTE(Abid): Reverse the temporary transpose */
    __T32TransposeInPlaceNoGrad(OtherOperand, OtherOperand->Header->Dim-1, OtherOperand->Header->Dim-2);

    /* NOTE(Abid): Reverse the vector dim expansions */
    if(Operands[0]->Header->Dim != OrigOperand0Dim) __SqueezeMatrixToVectorDim(Operands[0]);
    if(Operands[1]->Header->Dim != OrigOperand1Dim) __SqueezeMatrixToVectorDim(Operands[1]);
    if(Parent->Header->Dim != OrigParentDim) __SqueezeMatrixToVectorDim(Parent);
}

internal void
__T32BackwardSigmoid(t32 *Operand, t32 *Parent) {
    size_t OperandNumData = Operand->Header->StorageNumElements;
    size_t ParentOffset = 0;
    size_t OperOffset = 0;
    for(size_t OpNum = 1; OpNum <= OperandNumData; ++OpNum) {
        f32 SigValue = ((f32 *)Parent->Data.Ptr)[OperOffset];
        ((f32 *)Operand->Grad.Ptr)[OperOffset] += (SigValue * (1-SigValue))*
                                                     ((f32 *)Parent->Grad.Ptr)[ParentOffset];
        i32 DimMaxNumSoFar = 1;
        for(i32 DimIdx = 1; DimIdx <= (i32)Operand->Header->Dim; ++DimIdx) {
            DimMaxNumSoFar *= Operand->Header->Sizes[Operand->Header->Dim-DimIdx];
            if(OpNum % DimMaxNumSoFar == 0) {
                OperOffset -= Operand->Header->Strides[Operand->Header->Dim-DimIdx]*
                              (Operand->Header->Sizes[Operand->Header->Dim-DimIdx]-1);
                ParentOffset -= Parent->Header->Strides[Parent->Header->Dim-DimIdx]*
                                (Parent->Header->Sizes[Parent->Header->Dim-DimIdx]-1);
                continue;
            }
            OperOffset += Operand->Header->Strides[Operand->Header->Dim-DimIdx];
            ParentOffset += Parent->Header->Strides[Parent->Header->Dim-DimIdx];
            break;
        }
    }
}

#if 0
internal inline void
__SqueezeEmptyDims(t32 *A) {
    u32 DimsToPreserve = 0;
    for(i32 Idx = 0; Idx < (i32)A->Header->Dim; ++Idx) {
        if(A->Header->Size[Idx] == 1) continue;
        for(i32 Jdx = Idx-1; Jdx < Idx; --Jdx) {
            if(A->Header->Size[Jdx] != 1) break;
            A->Header->Size[Jdx] = A->Header->Size[Idx];
        }
        ++DimsToPreserve;
    }
    if(DimsToPreserve == 0) ++DimsToPreserve;
    A->Header->Dim = DimsToPreserve;
}
#endif

/* TODO(Abid): At the current moment, the gradients for the non-leaf tensors are also stored in
 *             within the tensor, which is not good. Ideally, one would want the gradient of
 *             non-leaf tensors to be transient and only used for computing leaf tensor grads.
 *             As a optimization, one could keep track of the memory footprint used to achieve
 *             transient gradient calculation for non-leaf tensors. Then, on the second run, we
 *             can completely forgo any allocation at all. */
/* TODO(Abid): Have to see if the current routine does adhere to the rules of `topological sort` */
/* TODO(Abid): In the end, add an Assert so that RootTensor should only have shape=(1). */

/* NOTE(Abid): If for a differentiable operation, one of the operands is also the result tensor,
 *             then we have nasty infinite loop on our hands.
 *             TODO: This can be fixed if we check that result is never the same as operand(s) */
/* TODO(Abid): Instead of having the root tensor saved here, we must create a computational graph
 *             explicitely and then pass around that variable to the T32Backprop function. This
 *             way, we can have multiple graphs at the same time for time when we need them. 
 *             The nice thing is, once you build it then you don't have to save operands when 
 *             you do operations on tensor, since we don't have to worry about that anymore. */
internal void
T32Backprop(t32 *RootTensor) {
    /* NOTE(Abid): Here's how `T32Backprop` works:
     * - The function will loop through the computation tree and calculate
     *   the gradient. That means it will not be recursive (at least for now).
     *
     * - In order to keep track of the tensors we want to traverse, we use a stack.
     *
     * - The first time Backward is called on a tensor, it allocates a default stack
     *   size and then appends new stack block as they become necessary. Sort of
     *   like a linked list of stack blocks: stack1 -> stack2 -> ... -> stackN(top)
     *
     * - At the end of the first run, it will calculate the amount of memory used
     *   for the entire run (GlobalStackSize), then deallocates the stacks and
     *   allocates a single stack block with the size of GlobalStackSize. This will
     *   ensure that the next Backward run will not require any memory allocation.
     *
     * - The function also stores the address of the previous root tensor for which
     *   it was called on. In the next call, if the address matches, then know we
     *   are in some nth run of a backward computation; however, if the address is
     *   different then the whole set of local_persist variables are set to their
     *   defaults, as we expect to be in the 1st run of a new chain of backward
     *   computations. */

    local_persist t32 *PrevRootTensor = {0};
    local_persist stack_blocks_state StackState = {0};

    Assert(RootTensor->Header && RootTensor->Grad.Ptr, "root tensor grad is not tracked");

    /* NOTE(Abid): If the below condition doesn't hit, then we are in the nth step of the same computation chain. */
    if((PrevRootTensor != RootTensor) || (PrevRootTensor->Grad.Ptr != RootTensor->Grad.Ptr)) {
        /* NOTE(Abid): Its the first run of a new computation chain. */
        if(PrevRootTensor && PrevRootTensor->Grad.Ptr) {
            while(StackState.CurrentBlock) {
                stack_block *NextBlock = StackState.CurrentBlock->BelowBlock;
                Free(StackState.CurrentBlock); 
                StackState.CurrentBlock = NextBlock;
            }

            /* NOTE(Abid): Free the reserved block from previous runs */
            Free(StackState.ReservedBlock);
            StackState.ReservedBlock = NULL;

        }

        /* NOTE(Abid): This will be the same whether this is the first call of the routine
         *             or we have a new computation chain. */
        StackState.CurrentBlockTopIdx = (size_t)-1;
        StackState.GlobalMaxTensorNum = 0;
        StackState.RunningTensorNum = 0;
        StackState.NewAllocTensorNum = 5;

        StackState.CurrentBlock = AllocNewStackBlock(StackState.NewAllocTensorNum, NULL);
    }

    __BackwardT32AddToElements(RootTensor, 1.f);

    /* NOTE(Abid): Backpropagation logic starts here */
    StackBlockPush(&StackState, RootTensor);

    while(!IsStackBlocksEmpty(&StackState)) {
        t32 *CurrentTensor = StackBlockTop(&StackState);
        tensor_op CurrentOp = CurrentTensor->Header->DerivedOp.TensorOp;
        StackBlockPop(&StackState);
        if(!CurrentTensor->Header->ShouldGrad || CurrentOp == op_None) continue;

        Assert(CurrentTensor->Header->DerivedOp.Operands, "tensor op set without operand(s)");
        Assert(CurrentTensor->Header->DerivedOp.Operands+1, "tensor op set without operand(s)");
        t32 *Operands[2];
        Operands[0] = CurrentTensor->Header->DerivedOp.Operands[0];
        Operands[1] = CurrentTensor->Header->DerivedOp.Operands[1];

        Assert(Operands[0]->Grad.Ptr, "grad storage not found");
        Assert((CurrentOp > op_UnaryEnd  && Operands[1]->Grad.Ptr) ||
               CurrentOp < op_UnaryEnd, "grad storage not found");
        Assert(CurrentTensor->Data.DType == dtype_f32, "cannot backpropagate through a non-float tensor")
        Assert(Operands[0]->Data.DType == dtype_f32, "cannot backpropagate through a non-float tensor")
        Assert((CurrentOp > op_UnaryEnd  && Operands[1]->Data.DType == dtype_f32) ||
               CurrentOp < op_UnaryEnd, "cannot backpropagate through a non-float tensor")

        switch (CurrentOp) {
            case op_UnaryNegate: {
            } break;
            case op_UnaryBroadcast: {
            } break;
            case op_UnaryTranpose: {
            } break;
            case op_UnaryTranposeAll: {
            } break;
            case op_UnaryReduceSumAll: {
                t32 *Operand = CurrentTensor->Header->DerivedOp.Operands[0];
                StackBlockPush(&StackState, Operand);

                __BackwardT32AddToElements(Operand, *(f32 *)CurrentTensor->Grad.Ptr);
            } break;
            case op_UnarySigmoid: {
                t32 *Operand = CurrentTensor->Header->DerivedOp.Operands[0];
                StackBlockPush(&StackState, Operand);

                __T32BackwardSigmoid(Operand, CurrentTensor);
            } break;
            case op_BinaryAdd: {
                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);

                __BackwardT32ReduceAddBroadcast(CurrentTensor, Operands[0]);
                __BackwardT32ReduceAddBroadcast(CurrentTensor, Operands[1]);
            } break;
            case op_BinarySub: {
                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);

                __BackwardT32ReduceAddBroadcast(CurrentTensor, Operands[0]);
                __BackwardT32ReduceSubBroadcast(CurrentTensor, Operands[1]);
            } break;
            case op_BinaryMul: {
                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);

                __BackwardT32Mul(Operands[1], CurrentTensor, Operands[0]);
                __BackwardT32Mul(Operands[0], CurrentTensor, Operands[1]);
            } break;
            case op_BinaryDiv: {
                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);

                __BackwardT32Div(Operands[1], CurrentTensor, Operands[0], 0);
                __BackwardT32Div(Operands[0], CurrentTensor, Operands[1], 1);
            } break;
            case op_BinaryMatmul: {
                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);

                __BackwardT32MatMul(Operands, CurrentTensor, 0);
                __BackwardT32MatMul(Operands, CurrentTensor, 1);
            } break;
            default: Assert(0, "invalid code path");
        }
    }

    Assert(!StackState.CurrentBlock, "oops, we should not have current block at the end");

    /* NOTE(Abid): Check if our maximum required stack size exceeded the reserved blocks's size, which means we haven't
     *             caliberated the optimal size. Should only happen if new chain is introduced or on the first run. */
    Assert(StackState.ReservedBlock, "reserved block should've been here! Gary, who took the block, man?");
    if(StackState.GlobalMaxTensorNum > StackState.ReservedBlock->MaxNumTen) {
        Free(StackState.ReservedBlock); 
        StackState.ReservedBlock = AllocNewStackBlock(StackState.GlobalMaxTensorNum, NULL);
    }

    PrevRootTensor = RootTensor;

    StackState.RunningTensorNum = 0;
}

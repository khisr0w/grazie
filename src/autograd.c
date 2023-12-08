/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/21/2023 10:49:09 PM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "autograd.h"

internal inline boolean
IsStackBlocksEmpty(stack_blocks_state *State) { return State->RunningTensorNum == 0; }

internal inline stack_block *
AllocNewStackBlock(size_t MaxNumTensors, stack_block *BelowBlock) {
    stack_block *Result = (stack_block *)Malloc(MaxNumTensors*sizeof(tensor32 *) + sizeof(stack_block));
    Assert(Result, "failed to allocate stack block");
    Result->TensorPtr = (tensor32 **)(Result+1);
    Result->MaxNumTen = MaxNumTensors;
    Result->BelowBlock = BelowBlock;

    return Result;
}

internal inline void
StackBlockPush(stack_blocks_state *State, tensor32 *Tensor) {
    if(!State->CurrentBlock || State->CurrentBlock->MaxNumTen ==
                               State->CurrentBlockTopIdx+1) {
        // In case, we do not have any blocks, or if the block is full
        stack_block *NewBlock = NULL;
        // NOTE(Abid): If we got one in reserve, use that
        if(State->ReservedBlock) {
            NewBlock = State->ReservedBlock;
            State->ReservedBlock = NULL;
            NewBlock->BelowBlock = State->CurrentBlock;
        } else {
            // TODO(Abid): Tweak the NewAllocTensorNum growth factor to something reasonable
            State->NewAllocTensorNum += (int32)(State->NewAllocTensorNum/2);
            NewBlock = AllocNewStackBlock(State->NewAllocTensorNum, State->CurrentBlock);
        }
        State->CurrentBlock = NewBlock;
        State->CurrentBlockTopIdx = (size_t)-1;
    }
    State->CurrentBlock->TensorPtr[++State->CurrentBlockTopIdx] = Tensor;
    // NOTE(Abid): Find the maximum number of tensors in the stack throughout all the runs of the same computation chain.
    ++State->RunningTensorNum;
    if(State->RunningTensorNum > State->GlobalMaxTensorNum) State->GlobalMaxTensorNum = State->RunningTensorNum;
}

internal inline void
StackBlockPop(stack_blocks_state *State) {
    Assert(State->RunningTensorNum > 0, "cannot pop empty stack");
    --State->CurrentBlockTopIdx;
    --State->RunningTensorNum;

    // NOTE(Abid): Reserve if block is empty, while freeing the already reserved block.
    if(State->CurrentBlockTopIdx == -1) {
        stack_block *BelowBlock = State->CurrentBlock->BelowBlock;

        // NOTE(Abid): Hold onto one extra block in case we push right after freeing a block.
        if(State->ReservedBlock) Free(State->ReservedBlock);
        State->ReservedBlock = State->CurrentBlock;

        State->CurrentBlock = BelowBlock;
        if(State->CurrentBlock) State->CurrentBlockTopIdx = State->CurrentBlock->MaxNumTen-1;
    }
}

internal inline tensor32 *
StackBlockTop(stack_blocks_state *State) {
    Assert(State->RunningTensorNum > 0, "cannot get top of an empty stack");
    return State->CurrentBlock->TensorPtr[State->CurrentBlockTopIdx];
}

/* ============================================
 * NOTE(Abid): Backward Operations
 * ============================================ */

#define __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, OP) \
    float32 TempVal = Value; \
    float32 *StoragePtr = (float32 *)A->Grad.Ptr; \
    for(int32 Idx = 0; Idx < A->Header->StorageNumElements; ++Idx) \
        StoragePtr[Idx] OP TempVal;
internal inline void
__BackwardT32AddToElements(tensor32 *A, float32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, +=); }

internal inline void
__BackwardT32SubToElements(tensor32 *A, float32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, -=); }

internal inline void
__BackwardT32MulToElements(tensor32 *A, float32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, *=); }

internal inline void
__BackwardT32DivToElements(tensor32 *A, float32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, /=); }

internal inline void
__BackwardT32SetElements(tensor32 *A, float32 Value)
{ __BACKWARD_OP_ELEMENTS_SCALAR(A, Value, =); }
#undef __BACKWARD_OP_ELEMENTS_SCALAR

// NOTE(Abid): The following operation does a reduce sum on the broadcast dims of A and add/sub/mul/div it
//             to Result tensor. There is a faster way to do it, but that most likely requires a malloc
//             which I shall never do.
#define __REDUCE_BROADCAST_DIMS(A, Result, OP) \
    Assert(A->Header->Dim >= Result->Header->Dim, "result tensor cannot be higher than operand"); \
    size_t ANumData = A->Header->StorageNumElements; \
    \
    size_t ResultOffset = 0; \
    size_t AOffset = 0; \
    for(size_t OpNum = 1; OpNum <= ANumData; ++OpNum) { \
        *((float32 *)Result->Grad.Ptr + ResultOffset) OP *((float32 *)A->Grad.Ptr + AOffset); \
                                                        \
        int32 DimMaxNumSoFar = 1; \
        for(int32 DimIdx = 1; DimIdx <= (int32)A->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= A->Header->Sizes[A->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                AOffset -= A->Header->Strides[A->Header->Dim-DimIdx]*(A->Header->Sizes[A->Header->Dim-DimIdx]-1); \
                if((int32)Result->Header->Dim - DimIdx < 0) ResultOffset = 0; \
                else ResultOffset -= Result->Header->Strides[Result->Header->Dim-DimIdx]*(Result->Header->Sizes[Result->Header->Dim-DimIdx]-1); \
                continue; \
            } \
            AOffset += A->Header->Strides[A->Header->Dim-DimIdx]; \
            if((int32)Result->Header->Dim - DimIdx >= 0) ResultOffset += Result->Header->Strides[Result->Header->Dim-DimIdx]; \
            break; \
        } \
    }
internal void __BackwardT32ReduceAddBroadcast(tensor32 *A, tensor32 *Result) { __REDUCE_BROADCAST_DIMS(A, Result, +=); }
internal void __BackwardT32ReduceSubBroadcast(tensor32 *A, tensor32 *Result) { __REDUCE_BROADCAST_DIMS(A, Result, -=); }
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
        *((float32 *)ResultOperand->Grad.Ptr + ResultOffset) += *((OTHER_OPERAND_DTYPE *)OtherOperand->Data.Ptr + OtherOperandOffset) * \
                                                                *((float32 *)Parent->Grad.Ptr + ParentOffset); \
        int32 DimMaxNumSoFar = 1; \
        for(int32 DimIdx = 1; DimIdx <= (int32)Parent->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= Parent->Header->Sizes[Parent->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                ParentOffset -= Parent->Header->Strides[Parent->Header->Dim-DimIdx]*(Parent->Header->Sizes[Parent->Header->Dim-DimIdx]-1); \
                if((int32)ResultOperand->Header->Dim - DimIdx < 0) ResultOffset = 0; \
                else ResultOffset -= ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]* \
                                     (ResultOperand->Header->Sizes[ResultOperand->Header->Dim-DimIdx]-1); \
                if((int32)OtherOperand->Header->Dim - DimIdx < 0) OtherOperandOffset = 0; \
                else OtherOperandOffset -= OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]* \
                                           (OtherOperand->Header->Sizes[OtherOperand->Header->Dim-DimIdx]-1); \
                continue; \
            } \
            ParentOffset += Parent->Header->Strides[Parent->Header->Dim-DimIdx]; \
            if((int32)ResultOperand->Header->Dim - DimIdx >= 0) ResultOffset += ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]; \
            if((int32)OtherOperand->Header->Dim - DimIdx >= 0) OtherOperandOffset += OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]; \
            break; \
        } \
    }
internal void 
__BackwardT32BinaryMul(tensor32 *OtherOperand, tensor32 *Parent, tensor32 *ResultOperand) {
    switch (OtherOperand->Data.DType) {
        case dtype_float32: { __BACKWARD_BINARY_MUL(OtherOperand, Parent, ResultOperand, float32); } break;
        case dtype_int32: { __BACKWARD_BINARY_MUL(OtherOperand, Parent, ResultOperand, int32); } break;
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
        float32 ParentGrad = *((float32 *)Parent->Grad.Ptr + ParentOffset); \
        OTHER_OPERAND_DTYPE OtherOperandData = *((OTHER_OPERAND_DTYPE *)OtherOperand->Data.Ptr + OtherOperandOffset); \
        __BACKWARD_BINARY_DIV_OPERAND_OP; \
        int32 DimMaxNumSoFar = 1; \
        for(int32 DimIdx = 1; DimIdx <= (int32)Parent->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= Parent->Header->Sizes[Parent->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                ParentOffset -= Parent->Header->Strides[Parent->Header->Dim-DimIdx]*(Parent->Header->Sizes[Parent->Header->Dim-DimIdx]-1); \
                if((int32)ResultOperand->Header->Dim - DimIdx < 0) ResultOffset = 0; \
                else ResultOffset -= ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]* \
                                     (ResultOperand->Header->Sizes[ResultOperand->Header->Dim-DimIdx]-1); \
                if((int32)OtherOperand->Header->Dim - DimIdx < 0) OtherOperandOffset = 0; \
                else OtherOperandOffset -= OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]* \
                                           (OtherOperand->Header->Sizes[OtherOperand->Header->Dim-DimIdx]-1); \
                continue; \
            } \
            ParentOffset += Parent->Header->Strides[Parent->Header->Dim-DimIdx]; \
            if((int32)ResultOperand->Header->Dim - DimIdx >= 0) ResultOffset += ResultOperand->Header->Strides[ResultOperand->Header->Dim-DimIdx]; \
            if((int32)OtherOperand->Header->Dim - DimIdx >= 0) OtherOperandOffset += OtherOperand->Header->Strides[OtherOperand->Header->Dim-DimIdx]; \
            break; \
        } \
    }
internal void
__BackwardT32BinaryDiv(tensor32 *OtherOperand, tensor32 *Parent, tensor32 *ResultOperand, uint32 OperandIdx) {

    if(OperandIdx == 0) { // First Operand
        #define __BACKWARD_BINARY_DIV_OPERAND_OP \
            *((float32 *)ResultOperand->Grad.Ptr + ResultOffset) += (1.f / OtherOperandData) * ParentGrad

        switch (OtherOperand->Data.DType) {
            case dtype_float32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, float32); } break;
            case dtype_int32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, int32); } break;
        }
        #undef __BACKWARD_BINARY_DIV_OPERAND_OP
    } else {
        #define __BACKWARD_BINARY_DIV_OPERAND_OP \
            float32 SelfData = *((float32 *)ResultOperand->Data.Ptr + ResultOffset);\
            *((float32 *)ResultOperand->Grad.Ptr + ResultOffset) += (-OtherOperandData / (SelfData*SelfData)) * ParentGrad

        switch (OtherOperand->Data.DType) {
            case dtype_float32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, float32); } break;
            case dtype_int32: { __BACKWARD_BINARY_DIV(OtherOperand, Parent, ResultOperand, int32); } break;
        }
        #undef __BACKWARD_BINARY_DIV_OPERAND_OP
    }
}
#undef __BACKWARD_BINARY_DIV

#if 0
internal void
__BackwardT32BinaryMatMul(tensor32 *OtherOperand, tensor32 *Parent, tensor32 *ResultOperand, uint32 OperandIdx) {

    // TODO(Abid) Start here...
    if(OperandIdx == 0) {
        switch (OtherOperand->Data.DType) {
            case dtype_float32: { } break;
            case dtype_int32: { } break;
        }
    } else {
        switch (OtherOperand->Data.DType) {
            case dtype_float32: { } break;
            case dtype_int32: { } break;
        }
    }
}
#endif

/* TODO(Abid): At the current moment, the gradients for the non-leaf tensors are also stored in
 *             within the tensor, which is not good. Ideally, one would want the gradient of
 *             non-leaf tensors to be transient and only used for computing leaf tensor grads.
 *             As a optimization, one could keep track of the memory footprint used to achieve
 *             transient gradient calculation for non-leaf tensors. Then, on the second run, we
 *             can completely forgo any allocation at all.
 * TODO(Abid): Have to see if the current routine does adhere to the rules of `topological sort`
 * TODO(Abid): In the end, add an Assert so that RootTensor should only have shape=(1).
 */
internal void
Backward(tensor32 *RootTensor, boolean SetInitGradZero) {
    /*
       NOTE(Abid): Here's how `Backward` works:

       - The function will loop through the computation tree and calculate
         the gradient. That means it will not be recursive (at least for now).

       - In order to keep track of the tensors we want to traverse, we use a stack.

       - The first time Backward is called on a tensor, it allocates a default stack
         size and then appends new stack block as they become necessary. Sort of
         like a linked list of stack blocks: stack1 -> stack2 -> ... -> stackN(top)

       - At the end of the first run, it will calculate the amount of memory used
         for the entire run (GlobalStackSize), then deallocates the stacks and
         allocates a single stack block with the size of GlobalStackSize. This will
         ensure that the next Backward run will not require any memory allocation.

       - The function also stores the address of the previous root tensor for which
         it was called on. In the next call, if the address matches, then know we
         are in some nth run of a backward computation; however, if the address is
         different then the whole set of local_persist variables are set to their
         defaults, as we expect to be in the 1st run of a new chain of backward
         computations.
    */

    local_persist tensor32 *PrevRootTensor = {0};
    local_persist stack_blocks_state StackState = {0};

    Assert(RootTensor->Header && RootTensor->Grad.Ptr, "root tensor grad is not tracked");

    // NOTE(Abid): If the below condition doesn't hit, then we are in the nth step of the same computation chain.
    if((PrevRootTensor != RootTensor) || (PrevRootTensor->Grad.Ptr != RootTensor->Grad.Ptr)) {
        // NOTE(Abid): Its the first run of a new computation chain.
        if(PrevRootTensor && PrevRootTensor->Grad.Ptr) {
            while(StackState.CurrentBlock) {
                stack_block *NextBlock = StackState.CurrentBlock->BelowBlock;
                Free(StackState.CurrentBlock); 
                StackState.CurrentBlock = NextBlock;
            }

            // NOTE(Abid): Free the reserved block from previous runs
            Free(StackState.ReservedBlock);
            StackState.ReservedBlock = NULL;

        }

        // NOTE(Abid): This will be the same whether this is the first call of the routine
        //             or we have a new computation chain.
        StackState.CurrentBlockTopIdx = (size_t)-1;
        StackState.GlobalMaxTensorNum = 0;
        StackState.RunningTensorNum = 0;
        StackState.NewAllocTensorNum = 5;

        StackState.CurrentBlock = AllocNewStackBlock(StackState.NewAllocTensorNum, NULL);
    }

    if(SetInitGradZero) __BackwardT32SetElements(RootTensor, 0.f);
    __BackwardT32AddToElements(RootTensor, 1.f);

    // NOTE(Abid): Backpropagation logic starts here
    StackBlockPush(&StackState, RootTensor);

    while(!IsStackBlocksEmpty(&StackState)) {
        tensor32 *CurrentTensor = StackBlockTop(&StackState);
        StackBlockPop(&StackState);
        if(!CurrentTensor->Header->ShouldGrad) continue;

        switch (CurrentTensor->Header->DerivedOp.TensorOp) {
            case op_UnaryNegate: {
            } break;
            case op_UnaryBroadcast: {
            } break;
            case op_UnaryTranpose: {
            } break;
            case op_UnaryTranposeAll: {
            } break;
            case op_BinaryAdd: {
                Assert(CurrentTensor->Header->DerivedOp.Operands, "tensor op set without operand(s)");
                Assert(CurrentTensor->Header->DerivedOp.Operands+1, "tensor op set without operand(s)");
                tensor32 *Operands[2];
                Operands[0] = CurrentTensor->Header->DerivedOp.Operands[0];
                Operands[1] = CurrentTensor->Header->DerivedOp.Operands[1];
                Assert(Operands[0]->Grad.Ptr, "grad storage not found");
                Assert(Operands[1]->Grad.Ptr, "grad storage not found");

                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);
                if(SetInitGradZero) { __BackwardT32SetElements(Operands[0], 0.f); __BackwardT32SetElements(Operands[1], 0.f); }

                __BackwardT32ReduceAddBroadcast(CurrentTensor, Operands[0]);
                __BackwardT32ReduceAddBroadcast(CurrentTensor, Operands[1]);
            } break;
            case op_BinarySub: {
                // NOTE(Abid): Copy pasta here
                Assert(CurrentTensor->Header->DerivedOp.Operands, "tensor op set without operand(s)");
                Assert(CurrentTensor->Header->DerivedOp.Operands+1, "tensor op set without operand(s)");
                tensor32 *Operands[2];
                Operands[0] = CurrentTensor->Header->DerivedOp.Operands[0];
                Operands[1] = CurrentTensor->Header->DerivedOp.Operands[1];
                Assert(Operands[0]->Grad.Ptr, "grad storage not found");
                Assert(Operands[1]->Grad.Ptr, "grad storage not found");

                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);
                if(SetInitGradZero) { __BackwardT32SetElements(Operands[0], 0.f); __BackwardT32SetElements(Operands[1], 0.f); }

                __BackwardT32ReduceAddBroadcast(CurrentTensor, Operands[0]);
                __BackwardT32ReduceSubBroadcast(CurrentTensor, Operands[1]);
            } break;
            case op_BinaryMul: {
                Assert(CurrentTensor->Header->DerivedOp.Operands, "tensor op set without operand(s)");
                Assert(CurrentTensor->Header->DerivedOp.Operands+1, "tensor op set without operand(s)");
                tensor32 *Operands[2];
                Operands[0] = CurrentTensor->Header->DerivedOp.Operands[0];
                Operands[1] = CurrentTensor->Header->DerivedOp.Operands[1];
                Assert(Operands[0]->Grad.Ptr, "grad storage not found");
                Assert(Operands[1]->Grad.Ptr, "grad storage not found");

                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);
                if(SetInitGradZero) { __BackwardT32SetElements(Operands[0], 0.f); __BackwardT32SetElements(Operands[1], 0.f); }

                __BackwardT32BinaryMul(Operands[1], CurrentTensor, Operands[0]);
                __BackwardT32BinaryMul(Operands[0], CurrentTensor, Operands[1]);
            } break;
            case op_BinaryDiv: {
                Assert(CurrentTensor->Header->DerivedOp.Operands, "tensor op set without operand(s)");
                Assert(CurrentTensor->Header->DerivedOp.Operands+1, "tensor op set without operand(s)");
                tensor32 *Operands[2];
                Operands[0] = CurrentTensor->Header->DerivedOp.Operands[0];
                Operands[1] = CurrentTensor->Header->DerivedOp.Operands[1];
                Assert(Operands[0]->Grad.Ptr, "grad storage not found");
                Assert(Operands[1]->Grad.Ptr, "grad storage not found");

                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);
                if(SetInitGradZero) { __BackwardT32SetElements(Operands[0], 0.f); __BackwardT32SetElements(Operands[1], 0.f); }

                __BackwardT32BinaryDiv(Operands[1], CurrentTensor, Operands[0], 0);
                __BackwardT32BinaryDiv(Operands[0], CurrentTensor, Operands[1], 1);
            } break;
            case op_BinaryMatmul: {
                Assert(CurrentTensor->Header->DerivedOp.Operands, "tensor op set without operand(s)");
                Assert(CurrentTensor->Header->DerivedOp.Operands+1, "tensor op set without operand(s)");
                tensor32 *Operands[2];
                Operands[0] = CurrentTensor->Header->DerivedOp.Operands[0];
                Operands[1] = CurrentTensor->Header->DerivedOp.Operands[1];
                Assert(Operands[0]->Grad.Ptr, "grad storage not found");
                Assert(Operands[1]->Grad.Ptr, "grad storage not found");

                StackBlockPush(&StackState, Operands[1]);
                StackBlockPush(&StackState, Operands[0]);
                if(SetInitGradZero) { __BackwardT32SetElements(Operands[0], 0.f); __BackwardT32SetElements(Operands[1], 0.f); }

                // __BackwardT32BinaryMatMul(Operands[1], CurrentTensor, Operands[0], 0);
                // __BackwardT32BinaryMatMul(Operands[0], CurrentTensor, Operands[1], 1);
            } break;
            case op_ReduceSumAll: {
                tensor32 *Operand = CurrentTensor->Header->DerivedOp.Operands[0];
                StackBlockPush(&StackState, Operand);
                if(SetInitGradZero) __BackwardT32SetElements(Operand, 0.f);

                __BackwardT32AddToElements(Operand, *(float32 *)CurrentTensor->Grad.Ptr);
            } break;
            case op_None: {
                continue;
            } break;
            /* case op_Ignore: {
                continue;
            } break; */
            default: Assert(0, "invalid code path");
        }
    }

    Assert(!StackState.CurrentBlock, "oops, we should not have current block at the end");

    // NOTE(Abid): Check if our maximum required stack size exceeded the reserved blocks's size, which means we haven't
    //             caliberated the optimal size. Should only happen if new chain is introduced or on the first run.
    Assert(StackState.ReservedBlock, "reserved block should've been here! Gary, who took the block, man?");
    if(StackState.GlobalMaxTensorNum > StackState.ReservedBlock->MaxNumTen) {
        Free(StackState.ReservedBlock); 
        StackState.ReservedBlock = AllocNewStackBlock(StackState.GlobalMaxTensorNum, NULL);
    }

    PrevRootTensor = RootTensor;

    StackState.RunningTensorNum = 0;
}

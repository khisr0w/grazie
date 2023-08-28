/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/21/2023 10:49:09 PM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

typedef struct stack_block stack_block;
struct stack_block
{
    tensor32 *TensorPtr;
    size_t NumTensors;

    stack_block *BelowBlock;
};

typedef struct
{
    stack_block *CurrentStackBlock;
    size_t CurrentStackTopIdx;
    size_t GlobalMaxTensorNum;
    size_t RunningTensorNum;
    size_t NewAllocTensorNum;

    stack_block *ReservedBlock;
} stack_blocks_state;

internal inline boolean
IsStackBlocksEmpty(stack_blocks_state *State)
{
    return ((State->CurrentStackTopIdx == -1) && (!State->CurrentStackBlock));
}

internal inline stack_block *
AllocNewStackBlock(size_t NumTensors, stack_block *BelowBlock)
{
    stack_block *Result = (stack_block *)Malloc(NumTensors*sizeof(tensor32) + sizeof(stack_block));
    Assert(Result, "failed to allocate stack block");
    Result->TensorPtr = (tensor32 *)(Result+1);
    Result->NumTensors = NumTensors;
    Result->BelowBlock = BelowBlock;

    return Result;
}

internal inline void
StackBlockPush(stack_blocks_state *State, tensor32 Tensor)
{
    // NOTE(Abid): If full, create new stack block
    if(State->CurrentStackTopIdx >= State->CurrentStackBlock->NumTensors)
    {
        stack_block *NewBlock = NULL;
        // NOTE(Abid): If we got one in reserve, use that
        if(State->ReservedBlock) 
        {
            NewBlock = State->ReservedBlock;
            State->ReservedBlock = NULL;
            NewBlock->BelowBlock = State->CurrentStackBlock;
        }
        else
        {
            // TODO(Abid): Tweak the NewAllocTensorNum growth factor to something reasonable
            State->NewAllocTensorNum += (int32)(State->NewAllocTensorNum/2);
            NewBlock = AllocNewStackBlock(State->NewAllocTensorNum, State->CurrentStackBlock);
        }

        State->CurrentStackBlock = NewBlock;
        State->CurrentStackTopIdx = (size_t)-1;
    }

    State->CurrentStackBlock->TensorPtr[++State->CurrentStackTopIdx] = Tensor;
    // NOTE(Abid): Find the maximum number of tensors in the stack throughout all the runs of the same computation chain.
    ++State->RunningTensorNum;
    if(State->RunningTensorNum > State->GlobalMaxTensorNum) State->GlobalMaxTensorNum = State->RunningTensorNum;
}

internal inline void
StackBlockPop(stack_blocks_state *State)
{
    Assert((State->CurrentStackTopIdx != -1) || (!State->CurrentStackBlock), "cannot pop empty stack");
    State->CurrentStackTopIdx--;
    --State->RunningTensorNum;

    // NOTE(Abid): Reserve if block is empty, while freeing the already reserved block.
    if(State->CurrentStackTopIdx == -1)
    {
        stack_block *Temp = State->CurrentStackBlock->BelowBlock;

        // NOTE(Abid): Hold onto one extra block in case we push right after freeing a block.
        if (State->ReservedBlock)
        {
            Free(State->ReservedBlock);
            State->ReservedBlock = State->CurrentStackBlock;
        }
        else State->ReservedBlock = State->CurrentStackBlock;

        State->CurrentStackBlock = Temp;
        if(State->CurrentStackBlock) State->CurrentStackTopIdx = State->CurrentStackBlock->NumTensors-1;
    }
}

internal inline tensor32
StackBlockTop(stack_blocks_state *State) {
    Assert(State->CurrentStackTopIdx != -1, "cannot get top of an empty stack");
    return State->CurrentStackBlock->TensorPtr[State->CurrentStackTopIdx];
}

internal void
Backward(tensor32 RootTensor)
{
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

    local_persist tensor32 PrevRootTensor = {0};
    local_persist stack_blocks_state State = {NULL, (size_t)-1, 0, 0, 2, NULL};

    Assert(RootTensor.Grad, "root tensor grad is not tracked");

    // NOTE(Abid): If the below condition doesn't hit, then we are in the nth step of the same computation chain.
    if((PrevRootTensor.Header != RootTensor.Header) ||
       (PrevRootTensor.Data != RootTensor.Data))
    {
        // NOTE(Abid): Its the first run of a new computation chain.
        if(PrevRootTensor.Header && PrevRootTensor.Data)
        {
            while(State.CurrentStackBlock)
            {
                stack_block *NextBlock = State.CurrentStackBlock->BelowBlock;
                Free(State.CurrentStackBlock); 
                State.CurrentStackBlock = NextBlock;
            }

            // NOTE(Abid): Free the reserved block from previous runs
            Free(State.ReservedBlock);
            State.ReservedBlock = NULL;

        }

        // NOTE(Abid): This will be the same whether this is the first call of the routine
        //             or we have a new computation chain.
        State.CurrentStackTopIdx = (size_t)-1;
        State.GlobalMaxTensorNum = 0;
        State.RunningTensorNum = 0;
        State.NewAllocTensorNum = 2;

        State.CurrentStackBlock = AllocNewStackBlock(State.NewAllocTensorNum, NULL);
    }

    // NOTE(Abid): Backpropagation logic starts here
    StackBlockPush(&State, RootTensor);
    CallOnGradStorage(RootTensor, T32SetElementsInPlace(RootTensor, 1.f));

    while(!IsStackBlocksEmpty(&State))
    {
        tensor32 CurrentTensor = StackBlockTop(&State);
        StackBlockPop(&State);

        switch (CurrentTensor.Header->DerivedOp.TensorOp)
        {
            case op_UnaryNegate:
            {
            } break;
            case op_UnaryBroadcast:
            {
            } break;
            case op_UnaryTranpose:
            {
            } break;
            case op_UnaryTranposeAll:
            {
            } break;
            case op_BinaryAdd:
            {
                tensor32 FirstOperand = CurrentTensor.Header->DerivedOp.Operands[0];
                tensor32 SecondOperand = CurrentTensor.Header->DerivedOp.Operands[1];
                if(SecondOperand.Header->ShouldGrad)
                {
                    Assert(SecondOperand.Header->GradStorageInit, "no grad storage for trackable tensor");
                    StackBlockPush(&State, SecondOperand);
                }
                if(FirstOperand.Header->ShouldGrad)
                {
                    Assert(FirstOperand.Header->GradStorageInit, "no grad storage for trackable tensor");
                    StackBlockPush(&State, FirstOperand);
                }

                StackBlockTop(&State);
            } break;
            case op_BinarySub:
            {
            } break;
            case op_BinaryMul:
            {
            } break;
            case op_BinaryDiv:
            {
            } break;
            case op_BinaryMatmul:
            {
            } break;
            case op_None:
            {
                continue;
            } break;
            default: Assert(0, "invalid code path");
        }
    }

    Assert(!State.CurrentStackBlock, "oops, we should not have current block at the end");

    // NOTE(Abid): Check if our maximum required stack size exceeded the reserved blocks's size, which means we haven't
    //             caliberated the optimal size. Should only happen if new chain is introduced or on the first run.
    Assert(State.ReservedBlock, "reserved block should've been here! Gary, who took the block, man?");
    if(State.GlobalMaxTensorNum > State.ReservedBlock->NumTensors)
    {
        Free(State.ReservedBlock); 
        State.ReservedBlock = AllocNewStackBlock(State.GlobalMaxTensorNum, NULL);
    }

    PrevRootTensor = RootTensor;

    State.RunningTensorNum = 0;
}

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
    int32 GlobalMaxTensorNum;
    int32 RunningTensorNum;

    stack_block *ReservedBlock;
} stack_blocks_state;

internal inline boolean
IsStackBlocksEmpty(stack_blocks_state *State)
{
    return ((State->CurrentStackTopIdx == -1) && (!State->CurrentStackBlock));
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
        }
        else
        {
            // TODO(Abid): Change the number of allocated memory here to get greater sizes of stack block.
            // WARNING(Abid): Do not allocate based on the NumTensors of previous block size it might be a ReservedBlock.
            //                Maybe have a variable in the State that tracks the growth rate of new blocks.
            size_t NewNumTensor = State->CurrentStackBlock->NumTensors;
            NewBlock = (stack_block *)Malloc(NewNumTensor*sizeof(tensor32) + sizeof(stack_block));
            NewBlock->TensorPtr = (tensor32 *)(NewBlock+1);
            NewBlock->NumTensors = NewNumTensor;
        }

        NewBlock->BelowBlock = State->CurrentStackBlock;
        State->CurrentStackBlock = NewBlock;
        State->CurrentStackTopIdx = (size_t)-1;
    }

    State->CurrentStackBlock->TensorPtr[++State->CurrentStackTopIdx] = Tensor;
    ++State->RunningTensorNum;
    if(State->RunningTensorNum > State->GlobalMaxTensorNum) State->GlobalMaxTensorNum = State->RunningTensorNum;
}

internal inline void
StackBlockPop(stack_blocks_state *State)
{
    Assert(State->CurrentStackTopIdx != -1, "cannot pop empty stack");
    State->CurrentStackTopIdx--;
    ++State->RunningTensorNum;

    // NOTE(Abid): Free if block is empty.
    if(State->CurrentStackTopIdx == -1)
    {
        stack_block *Temp = State->CurrentStackBlock->BelowBlock;

        // NOTE(Abid): Hold onto one extra block in case we push right after freeing a block.
        if(State->ReservedBlock) Free(State->CurrentStackBlock);
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

    local_persist stack_blocks_state State = {0, (size_t)-1, 0, 0};

    // NOTE(Abid): If the below condition doesn't hit, then we are in the nth step of the same computation chain.
    if((PrevRootTensor.Header != RootTensor.Header) ||
       (PrevRootTensor.Data != RootTensor.Data))
    {
        // NOTE(Abid): Its the first run of a new computation chain.
        if(PrevRootTensor.Header && PrevRootTensor.Data)
        {
            Assert(State.CurrentStackBlock, "CurrentStackBlock given NULL");
            while(State.CurrentStackBlock)
            {
                stack_block *NextBlock = State.CurrentStackBlock->BelowBlock;
                Free(State.CurrentStackBlock); 
                State.CurrentStackBlock = NextBlock;
            }

            State.CurrentStackTopIdx = (size_t)-1;
            State.GlobalMaxTensorNum = 0;
        }

        // NOTE(Abid): This will be the same whether this is the first call of the routine
        //             or we have a new computation chain.
        size_t InitialStackTensorNum = 2;
        State.CurrentStackBlock = (stack_block *)Malloc(InitialStackTensorNum*sizeof(tensor32)+
                                                                   sizeof(stack_block));
        State.CurrentStackBlock->TensorPtr = (tensor32 *)(State.CurrentStackBlock+1);
                                                                  
        Assert(State.CurrentStackBlock->TensorPtr, "failed to allocate stack");
        State.CurrentStackBlock->NumTensors = InitialStackTensorNum;
    }

    // NOTE(Abid): Backpropagation logic starts here

    StackBlockPush(&State, RootTensor);
    tensor32 CurrentTensor = StackBlockTop(&State);
    // tensor32 *FirstOperand = (tensor32 *)(Ten.Header->DerivedOp.Operands);
    // tensor32 *SecondOperand = FirstOperand+1;
    tensor_op Operation = CurrentTensor.Header->DerivedOp.TensorOp;
    // Operation = op_BinarySub;

    while(IsStackBlocksEmpty(&State))
    {
        switch (Operation)
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
            default: Assert(0, "invalid code path");
        }
    }

    // NOTE(Abid): If multiple blocks have been allocated, free all and create a block with maximum needed stack size.
    if(State.CurrentStackBlock->BelowBlock)
    {
        while(State.CurrentStackBlock)
        {
            stack_block *NextBlock = State.CurrentStackBlock->BelowBlock;
            Free(State.CurrentStackBlock); 
            State.CurrentStackBlock = NextBlock;
        }
        State.CurrentStackBlock = (stack_block *)Malloc(State.GlobalMaxTensorNum*sizeof(tensor32) + sizeof(stack_block));
    }

    PrevRootTensor = RootTensor;
}

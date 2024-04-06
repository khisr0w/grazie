/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  12/6/2023 1:14:14 AM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(AUTOGRAD_H)

typedef struct stack_block stack_block;
struct stack_block {
    t32 **TensorPtr;
    size_t MaxNumTen;

    stack_block *BelowBlock;
};

typedef struct {
    stack_block *CurrentBlock;
    size_t CurrentBlockTopIdx;
    size_t GlobalMaxTensorNum;
    size_t RunningTensorNum;
    size_t NewAllocTensorNum;

    stack_block *ReservedBlock;
} stack_blocks_state;

#define AUTOGRAD_H
#endif

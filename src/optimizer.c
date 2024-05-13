/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/14/2024 3:23:20 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#include "optimizer.h"

typedef struct {
    t32 **Array;
    usize MaxSize;
    usize Used;
} tensor_list;

#if 0
internal tensor_list *
__GLOBALCurrentOptimizerList(tensor_list *NewList, bool ShouldChange) {
    local_persist tensor_list CurrentList;
    if(NewList != NULL) {
        Assert(ShouldChange, "optimizer list cannot be changed. Forgot `ShouldChange`?")
        CurrentList.Array = NewList->Array;
        CurrentList.Size = NewList->Size;
        CurrentList.Ptr = NewList->Ptr;
    }

    return &CurrentList;
}
#endif

internal inline void
__T32AddToTensorList(tensor_list *TensorList, t32 *Tensor) {
    Assert(TensorList->Array != NULL, "cannot add tensor to an NULL tensor_list");

    Assert(TensorList->Used+1 != TensorList->MaxSize, "");
    /* TODO(Abid): Must remove this entire if clause, we are using arenas now. */
    if(TensorList->Used+1 == TensorList->MaxSize) {
        TensorList->MaxSize += TensorList->MaxSize; /* Factorial increase TODO(Abid): A better solution */
        // TensorList->Array = (t32 **)Realloc(TensorList->Array, TensorList->MaxSize);
        Assert(TensorList->Array, "Realloc failed to move memory");
    }

    TensorList->Array[TensorList->Used++] = Tensor;
}

internal inline tensor_list
T32AllocateTensorList(usize Size, mem_arena *Arena) {
    tensor_list TensorList = {0};
    TensorList.MaxSize = Size;
    // TensorList.Array = (t32 **)Malloc(TensorList.Size*sizeof(t32 *));
    TensorList.Array = PushArray(Arena, t32 *, TensorList.MaxSize);

    return TensorList;
}

#if 0
internal tensor_list
T32ToggleOptimWatch(bool Start) {
    local_persist bool IsWatching = false;
    tensor_list RegisteredList = {0};

    if(Start) {
        Assert(!IsWatching, "must end an optimizer watch to begin another.");
        IsWatching = true;
        tensor_list TensorList = {0};
        TensorList.Size = 1024;
        TensorList.Ptr = 0;
        TensorList.Array = (t32 **)Malloc(TensorList.Size*sizeof(t32 *));
        RegisteredList = *__GLOBALCurrentOptimizerList(&TensorList, true);
    } else {
        Assert(IsWatching, "no optimizer watch enabled for it to end.");
        IsWatching = false;
        RegisteredList = *__GLOBALCurrentOptimizerList(NULL, false);

        /* NOTE(Abid): Reset the list */
        tensor_list DummyEmptyList = {0};
        __GLOBALCurrentOptimizerList(&DummyEmptyList, true);
    } 

    return RegisteredList;
}
#endif

internal inline void
T32ZeroGrad(tensor_list TensorList) {
    for(u32 TensorIdx = 0; TensorIdx < TensorList.Used; ++TensorIdx) {
        t32 *Tensor = TensorList.Array[TensorIdx];
        for(usize DataIdx = 0; DataIdx < Tensor->Header->StorageNumElements; ++DataIdx) {
            ((f32 *)Tensor->Grad.Ptr)[DataIdx] = 0.f;
        }
    }
}

internal void
T32SGDOptim(tensor_list TensorList, f32 LearningRate) {
    /* TODO(Abid): Add nesterov momentum as well as weight decay. */

    for(u32 TensorIdx = 0; TensorIdx < TensorList.Used; ++TensorIdx) {
        t32 *Tensor = TensorList.Array[TensorIdx];
        size_t NumData = Tensor->Header->StorageNumElements;
        size_t Offset = 0;
        for(size_t OpNum = 1; OpNum <= NumData; ++OpNum) {
            *((f32 *)Tensor->Data.Ptr + Offset) -= LearningRate * (*((f32 *)Tensor->Grad.Ptr + Offset));
            i32 DimMaxNumSoFar = 1;
            for(i32 DimIdx = 1; DimIdx <= (i32)Tensor->Header->Dim; ++DimIdx) {
                DimMaxNumSoFar *= Tensor->Header->Sizes[Tensor->Header->Dim-DimIdx];
                if(OpNum % DimMaxNumSoFar == 0) {
                    Offset -= Tensor->Header->Strides[Tensor->Header->Dim-DimIdx] *
                              (Tensor->Header->Sizes[Tensor->Header->Dim-DimIdx]-1);
                    continue;
                }
                Offset += Tensor->Header->Strides[Tensor->Header->Dim-DimIdx];
                break;
            }
        }

    }
}

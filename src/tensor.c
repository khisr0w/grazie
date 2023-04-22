/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 10:19:09 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

/*
   TODO(Abid):

   - Convert all the code for operations (except for tensor creation) to work for both
     int32 and float32 tensors.


   NOTE(Abid): Here how the memory mapping is supposedly going to work for tensors.

   - If `no_grad` is turned on, we will go through the same math operation as if its off;
     however, in this case we will set the variable `ShouldGrad` to false. When going
     through the autograd, we will skip computing the gradient for said tensors.
     TODO(Abid): It must be checked that the user do not compute the grad of a tensor
                 based on a tensor that's before a 'no-grad' computation. So, always
                 check the `ShouldGrad` of operands.

   - It's also important for the operation memory to be flushed after each
     run through the loop, so that we can re-use the same memory again.
     Perhaps a function called `BackwardAndFlush()` be used for this.
     TODO(Abid): We could have a check at the start of each loop to see if its free.

   - The `PrintTensor` will be a macro where it will reset its memory footprint after the
     function __UnsafePrintTensor is done being called. Therefore, if there was an adhoc
     computation at the argument of PrintTensor, then we will print and free the memory.

   - To remove a non-presistent tensor, we will go through all the operands who is not
     persistent and free those as well.

   - When computing a tensor inside a for loop, if we've already got non-persistent
     operands from the computation of previous steps AND the the shape and memory
     requirements are the same as our new computation, then simply override.
     Otherwise, we free the memory and allocate new one that fits the requirements. (MAYBE!)
 */

#include "tensor.h"

#if 0
#define F32Tensor(Shape, Data) F32Tensor_(Shape, ArrayLength(Shape), Data, ArrayLength(Data))
internal tensor_f32
F32Tensor_(int32 *Shape, int32 ShapeLength, float32 *Data = NULL, size_t DataLength = 0, boolean Intialize = true)
{
    tensor_f32 Result = {};

    size_t StorageSize = 1;
    for (int32 i = 0; i < ShapeLength; ++i) { StorageSize *= Shape[i]; }
    Assert(StorageSize != 0, "Wrong shape given, cannot be zero");

    size_t FinalSize = StorageSize*sizeof(float32);

    Result.Header.Offset = 0;

    Assert(ShapeLength <= MAX_SHAPE_LENGTH, "Max shape length excceded");
    Result.Header.Dim = ShapeLength;
    memcpy(Result.Header.Sizes, Shape, Result.Header.Dim*sizeof(int32));

    // NOTE(Abid): Calculate the strides given the tensor shape
    for (int32 i = 0; i < Result.Header.Dim; ++i) Result.Header.Strides[i] = 1;
    if (Result.Header.Dim > 1)
    {
        for (int32 i = 0; i < (Result.Header.Dim-1); ++i)
            for (int32 j = i+1; j < Result.Header.Dim; ++j) { Result.Header.Strides[i] *= Result.Header.Sizes[j]; }
    }

    Result.Storage = (float32 *)Malloc(FinalSize);
    Assert(Result.Storage, "Storage memory cannot be allocated");

    // NOTE(Abid): Check if the DataLength makes sense with the shape
    if(Intialize)
    {
        Assert(DataLength == StorageSize, "Mismatch of data and tensor shape");
        memcpy(Result.Storage, Data, StorageSize*sizeof(float32));
    }
    
    return Result;
}
#endif

#if 0
internal inline tensor_i32
I32TenAssign(tensor_i32 Tensor)
{
    // Tensor.Header->IsPersist = true;
    return Tensor;
}
#endif

#define I32Tensor(Shape, Data) _I32Tensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data), true)
internal inline tensor_i32
_I32Tensor(uint32 *Shape, uint32 ShapeLength, int32 *Data, size_t DataLength, boolean Intialize)
{
    tensor_i32 Result = {0};

    size_t StorageSize = 1;
    for (uint32 i = 0; i < ShapeLength; ++i) { StorageSize *= Shape[i]; }
    Assert(StorageSize != 0, "Wrong shape given, cannot be zero");

    size_t FinalSize = sizeof(tensor_header) +
                       2*ShapeLength*sizeof(uint32) + // NOTE(Abid): Stride and Shape
                       StorageSize*sizeof(int32);

    // NOTE(Abid): Memory mapping
    Result.Header = (tensor_header *)Malloc(FinalSize);
    Assert(Result.Header, "Storage memory cannot be allocated");
    Result.Header->Sizes = (uint32 *)(Result.Header+1);
    Result.Header->Strides = (uint32 *)(Result.Header->Sizes + ShapeLength);
    // Result.Header->IsPersist = false;
    // NOTE(Abid): Setting whether to compute the backward pass or not
    Result.Header->ShouldGrad = IS_GRAD_PRESERVE() ? true : false;
    Result.Header->DerivedOp.TensorOp = op_None;
    Result.Header->DerivedOp.OpContext = NULL;

    Result.Storage = (int32 *)(Result.Header->Strides + ShapeLength);
    
    // NOTE(Abid): Setting Default Values
    Result.Header->Offset = 0;
    Result.Header->Dim = ShapeLength;
    memcpy(Result.Header->Sizes, Shape, ShapeLength*sizeof(uint32));

    // NOTE(Abid): Calculate the strides given the tensor shape
    for (uint32 i = 0; i < Result.Header->Dim; ++i) Result.Header->Strides[i] = 1;
    if (Result.Header->Dim > 1)
    {
        for (uint32 i = 0; i < (Result.Header->Dim-1); ++i)
            for (uint32 j = i+1; j < Result.Header->Dim; ++j) { Result.Header->Strides[i] *=
                                                               Result.Header->Sizes[j]; }
    }

    if(Intialize)
    {
        // NOTE(Abid): Check if the DataLength makes sense with the shape
        Assert(DataLength == StorageSize, "Mismatch of data and tensor shape");
        memcpy(Result.Storage, Data, StorageSize*sizeof(int32));
    }
    
    return Result;
}

internal inline size_t
GetStorageSize(int32 *Sizes, int32 Dim)
{
    size_t Result = 1;

    for (int32 Idx = 0; Idx < Dim; ++Idx)
    {
        Assert(Sizes[Idx], "tensor shape cannot be zero")
        Result *= Sizes[Idx];
    }

    return Result;
}

internal inline boolean
IsShapeEqual(tensor_header *A, tensor_header *B)
{
    if(A->Dim != B->Dim) return false;

    for (uint32 Idx = 0; Idx < A->Dim; ++Idx)
        if (A->Sizes[Idx] != B->Sizes[Idx]) return false;

    return true;
}

internal inline void
_PrintTensorHeader(char *Name, tensor_header *TenHeader)
{
    printf("%s", Name); printf(" -> shape (");

    for (uint32 i = 0; i < (TenHeader->Dim-1); ++i) { printf("%d,", TenHeader->Sizes[i]); }
    printf("%d) :=\n",TenHeader->Sizes[TenHeader->Dim-1]);
}

internal inline boolean
_PrintIsOuterAndUpdateDims(int32 *AccessDimIdx, uint32 *AccessDims, tensor_header *TenHeader, int32 *PrintCount)
{
    boolean IsOuterDim = *AccessDimIdx != (int32)(TenHeader->Dim-1);
    if (IsOuterDim)
    {
        // NOTE(Abid): Start of the dimension
        if (AccessDims[*AccessDimIdx] == 0) { printf("["); ++(*PrintCount); }

        // NOTE(Abid): In case we have reached the end of this dimension
        if (AccessDims[*AccessDimIdx] == TenHeader->Sizes[*AccessDimIdx])
        {
            printf("]"); ++(*PrintCount);
            if((*AccessDimIdx-1) >= 0 &&
                    AccessDims[*AccessDimIdx-1] < (TenHeader->Sizes[*AccessDimIdx-1]-1))
            { printf(", "); ++(*PrintCount); }

            AccessDims[(*AccessDimIdx)--] = 0;
            if(*AccessDimIdx == -1) return IsOuterDim;
            ++AccessDims[(*AccessDimIdx)];
        } else ++(*AccessDimIdx);
    }

    return IsOuterDim;
}

// TODO(Abid): The print currently is leaking memory since its not freeing the tensors that are
//             the result of in-argument computations. Needs to be macro'ed out once memory
//             module is written.

internal void
PrintF32Tensor(tensor_f32 Tensor)
{
    int32 MaxPrintWidth = 20;
    int32 PrintCount = 0;

    tensor_header *TenHeader = Tensor.Header;
    _PrintTensorHeader("tensor f32", TenHeader);

    uint32 *AccessDims = (uint32 *)Calloc(TenHeader->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        if(PrintCount >= MaxPrintWidth) { printf("\n"); PrintCount = 0; }
        if(_PrintIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader, &PrintCount)) continue;

        // NOTE(Abid): Main printing of the last dimension
        printf("["); ++PrintCount;
        ++PrintCount;
        for (uint32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;
            size_t Pos = 0;
            for (uint32 PosIdx = 0; PosIdx < Tensor.Header->Dim; ++PosIdx) Pos += AccessDims[PosIdx] * Tensor.Header->Strides[PosIdx];

            float32 Value = Tensor.Storage[Pos];
            printf("%.3f", Value); ++PrintCount;
            if(AccessDims[AccessDimIdx] < (TenHeader->Sizes[AccessDimIdx]-1)) { printf(", "); ++PrintCount; }
        }
        printf("]"); ++PrintCount;
        if(AccessDims[AccessDimIdx-1] < (TenHeader->Sizes[AccessDimIdx-1]-1)) { printf(", "); ++PrintCount; }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }
    printf("\n\n");

    Free(AccessDims);
}

internal void
PrintI32Tensor(tensor_i32 Tensor)
{
    int32 MaxPrintWidth = 20;
    int32 PrintCount = 0;

    tensor_header *TenHeader = Tensor.Header;
    _PrintTensorHeader("tensor i32", TenHeader);

    uint32 *AccessDims = (uint32 *)Calloc(TenHeader->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        if(PrintCount >= MaxPrintWidth) { printf("\n"); PrintCount = 0; }
        if(_PrintIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader, &PrintCount)) continue;

        // NOTE(Abid): Main printing of the last dimension
        printf("["); ++PrintCount;
        ++PrintCount;
        for (uint32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;

            size_t Pos = 0;
            for (uint32 PosIdx = 0; PosIdx < Tensor.Header->Dim; ++PosIdx) Pos += AccessDims[PosIdx] * Tensor.Header->Strides[PosIdx];

            int32 Value = Tensor.Storage[Pos];
            printf("%d", Value); ++PrintCount;
            if(AccessDims[AccessDimIdx] < (TenHeader->Sizes[AccessDimIdx]-1)) { printf(", "); ++PrintCount; }
        }
        printf("]"); ++PrintCount;
        if(AccessDims[AccessDimIdx-1] < (TenHeader->Sizes[AccessDimIdx-1]-1)) { printf(", "); ++PrintCount; }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }
    printf("\n\n");

    Free(AccessDims);
}

// =======================================
// NOTE(Abid): Math Operations tensor_i32
// =======================================

internal tensor_i32
I32TenAdd(tensor_i32 A, tensor_i32 B)
{
    Assert(IsShapeEqual(A.Header, B.Header), "shape mismatch for + operation");
    tensor_i32 Result = _I32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false);

    // NOTE(Abid): We should not compute the grad in the backward case.
    if(!IS_GRAD_PRESERVE()) Result.Header->ShouldGrad = false;

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            size_t ATenOuterDimOffset = 0;
            size_t BTenOuterDimOffset = 0;
            size_t ResultOuterDimOffset = 0;

            // NOTE(Abid): Offset until the start of the inner dimension
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx];
                BTenOuterDimOffset += AccessDims[Idx] * B.Header->Strides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * Result.Header->Strides[Idx];
            }
            for (uint32 Idx = 0; Idx < A.Header->Sizes[AccessDimIdx]; ++Idx)
            {
                Result.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset] + B.Storage[BTenOuterDimOffset];

                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx];
                BTenOuterDimOffset += B.Header->Strides[AccessDimIdx];
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx];
            }

            // NOTE(Abid): Go to the previous dimension while setting this one to zero
            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    // NOTE(Abid): Add information for backpropagation
    Result.Header->DerivedOp.TensorOp = op_BinaryAdd;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32)*2);
    Operands[0] = A;
    Operands[1] = B;
    Result.Header->DerivedOp.Operands = Operands;

    // NOTE(Abid): Free the superfluous allocations
    Free(AccessDims);
    return Result;
}

internal tensor_i32
I32TenSub(tensor_i32 A, tensor_i32 B)
{
    Assert(IsShapeEqual(A.Header, B.Header), "shape mismatch for - operation");
    tensor_i32 Result = _I32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false);

    // NOTE(Abid): We should not compute the grad in the backward case.
    if(!IS_GRAD_PRESERVE()) Result.Header->ShouldGrad = false;

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            size_t ATenOuterDimOffset = 0;
            size_t BTenOuterDimOffset = 0;
            size_t ResultOuterDimOffset = 0;

            // NOTE(Abid): Offset until the start of the inner dimension
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx];
                BTenOuterDimOffset += AccessDims[Idx] * B.Header->Strides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * Result.Header->Strides[Idx];
            }
            for (uint32 Idx = 0; Idx < A.Header->Sizes[AccessDimIdx]; ++Idx)
            {
                Result.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset] -
                                                       B.Storage[BTenOuterDimOffset];

                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx];
                BTenOuterDimOffset += B.Header->Strides[AccessDimIdx];
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx];
            }

            // NOTE(Abid): Go to the previous dimension while setting this one to zero
            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    // NOTE(Abid): Add information for backpropagation
    Result.Header->DerivedOp.TensorOp = op_BinarySub;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32)*2);
    Operands[0] = A;
    Operands[1] = B;
    Result.Header->DerivedOp.Operands = Operands;

    // NOTE(Abid): Free the superfluous allocations
    Free(AccessDims);
    return Result;
}

internal tensor_i32
I32TenMul(tensor_i32 A, tensor_i32 B)
{
    Assert(IsShapeEqual(A.Header, B.Header), "shape mismatch for * operation");
    tensor_i32 Result = _I32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false);

    // NOTE(Abid): We should not compute the grad in the backward case.
    if(!IS_GRAD_PRESERVE()) Result.Header->ShouldGrad = false;

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            size_t ATenOuterDimOffset = 0;
            size_t BTenOuterDimOffset = 0;
            size_t ResultOuterDimOffset = 0;

            // NOTE(Abid): Offset until the start of the inner dimension
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx];
                BTenOuterDimOffset += AccessDims[Idx] * B.Header->Strides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * Result.Header->Strides[Idx];
            }
            for (uint32 Idx = 0; Idx < A.Header->Sizes[AccessDimIdx]; ++Idx)
            {
                Result.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset] *
                                                       B.Storage[BTenOuterDimOffset];

                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx];
                BTenOuterDimOffset += B.Header->Strides[AccessDimIdx];
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx];
            }

            // NOTE(Abid): Go to the previous dimension while setting this one to zero
            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    // NOTE(Abid): Add information for backpropagation
    Result.Header->DerivedOp.TensorOp = op_BinaryMult;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32)*2);
    Operands[0] = A;
    Operands[1] = B;
    Result.Header->DerivedOp.Operands = Operands;

    // NOTE(Abid): Free the superfluous allocations
    Free(AccessDims);
    return Result;
}

internal tensor_i32
I32TenDiv(tensor_i32 A, tensor_i32 B)
{
    Assert(IsShapeEqual(A.Header, B.Header), "shape mismatch for division operation");
    tensor_i32 Result = _I32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false);

    // NOTE(Abid): We should not compute the grad in the backward case.
    if(!IS_GRAD_PRESERVE()) Result.Header->ShouldGrad = false;

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            size_t ATenOuterDimOffset = 0;
            size_t BTenOuterDimOffset = 0;
            size_t ResultOuterDimOffset = 0;

            // NOTE(Abid): Offset until the start of the inner dimension
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx];
                BTenOuterDimOffset += AccessDims[Idx] * B.Header->Strides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * Result.Header->Strides[Idx];
            }
            for (uint32 Idx = 0; Idx < A.Header->Sizes[AccessDimIdx]; ++Idx)
            {
                Result.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset] /
                                                       B.Storage[BTenOuterDimOffset];

                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx];
                BTenOuterDimOffset += B.Header->Strides[AccessDimIdx];
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx];
            }

            // NOTE(Abid): Go to the previous dimension while setting this one to zero
            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    // NOTE(Abid): Add information for backpropagation
    Result.Header->DerivedOp.TensorOp = op_BinaryMult;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32)*2);
    Operands[0] = A;
    Operands[1] = B;
    Result.Header->DerivedOp.Operands = Operands;

    // NOTE(Abid): Free the superfluous allocations
    Free(AccessDims);
    return Result;
}


internal tensor_i32
I32TenTranspose(tensor_i32 A, int32 Dim1, int32 Dim2)
{
    Dim1 = (A.Header->Dim + Dim1) % A.Header->Dim;
    Dim2 = (A.Header->Dim + Dim2) % A.Header->Dim;

    Assert(Dim1 != Dim2, "incorrect dimensions for transpose; select two different dimensions");
    Assert(((int32)A.Header->Dim >= Dim1) && ((int32)A.Header->Dim >= Dim2),
           "dimension index out of bounds");

    uint32 *NewSizes = (uint32 *)Malloc(2*A.Header->Dim*sizeof(uint32));
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(uint32));
    uint32 *NewStrides = NewSizes + A.Header->Dim;
    memcpy(NewStrides, A.Header->Strides, A.Header->Dim*sizeof(uint32));


    // NOTE(Abid): Swamp the sizes and strides
    NewSizes[Dim1] = A.Header->Sizes[Dim2];
    NewSizes[Dim2] = A.Header->Sizes[Dim1];

    NewStrides[Dim1] = A.Header->Strides[Dim2];
    NewStrides[Dim2] = A.Header->Strides[Dim1];

    tensor_i32 ResultTen = _I32Tensor(NewSizes, A.Header->Dim, 0, 0, false);

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == NewSizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            uint32 ATenOuterDimOffset = 0;
            uint32 ResultOuterDimOffset = 0;
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * NewStrides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * ResultTen.Header->Strides[Idx];
            }
            for(uint32 Idx = 0; Idx < NewSizes[AccessDimIdx]; ++Idx)
            {
                ResultTen.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset];
                ATenOuterDimOffset += NewStrides[AccessDimIdx];
                ResultOuterDimOffset += ResultTen.Header->Strides[AccessDimIdx];
            }

            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    // NOTE(Abid): Add information for backpropagation
    ResultTen.Header->DerivedOp.TensorOp = op_UnaryTranpose;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32) + sizeof(int32)*2);
    Operands[0] = A;
    int32 *TranposedDims = (int32 *)(Operands+1);
    TranposedDims[0] = Dim1;
    TranposedDims[1] = Dim2;
    ResultTen.Header->DerivedOp.Operands = Operands;
    ResultTen.Header->DerivedOp.OpContext = TranposedDims;

    Free(NewSizes);
    Free(AccessDims);
    return ResultTen;
}

internal tensor_i32
I32TenTransposeAll(tensor_i32 A)
{
    uint32 *NewSizes = (uint32 *)Malloc(2*A.Header->Dim*sizeof(uint32));
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(uint32));
    uint32 *NewStrides = NewSizes + A.Header->Dim;
    memcpy(NewStrides, A.Header->Strides, A.Header->Dim*sizeof(uint32));

    // NOTE(Abid): Swamp the sizes and strides
    for(uint32 Idx = 0; Idx < A.Header->Dim; ++Idx)
    {
        NewSizes[Idx] = A.Header->Sizes[A.Header->Dim-Idx-1];
        NewStrides[Idx] = A.Header->Strides[A.Header->Dim-Idx-1];
    }

    tensor_i32 ResultTen = _I32Tensor(NewSizes, A.Header->Dim, 0, 0, false);

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == NewSizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            uint32 ATenOuterDimOffset = 0;
            uint32 ResultOuterDimOffset = 0;
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * NewStrides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * ResultTen.Header->Strides[Idx];
            }
            for(uint32 Idx = 0; Idx < NewSizes[AccessDimIdx]; ++Idx)
            {
                ResultTen.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset];
                ATenOuterDimOffset += NewStrides[AccessDimIdx];
                ResultOuterDimOffset += ResultTen.Header->Strides[AccessDimIdx];
            }

            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }


    // NOTE(Abid): Add information for backpropagation
    ResultTen.Header->DerivedOp.TensorOp = op_UnaryTranposeAll;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32));
    Operands[0] = A;
    ResultTen.Header->DerivedOp.Operands = Operands;

    Free(NewSizes);
    Free(AccessDims);
    return ResultTen;
}

internal tensor_i32
I32TenMatMul(tensor_i32 A, tensor_i32 B)
{
    Assert(A.Header->Dim == B.Header->Dim, "dimension mismatch for MatMul operation");
    Assert((A.Header->Dim != 0) && (B.Header->Dim != 0), "dimension must be non-zero")
    // NOTE(Abid): check if shape is appropriate
    if(A.Header->Dim >= 2)
    {
        uint32 MatMulOutDimLastIdx = A.Header->Dim-2;
        //NOTE(Abid): Outer dim check
        for(uint32 Idx = 0; Idx < MatMulOutDimLastIdx; ++Idx)
            Assert(A.Header->Sizes[Idx] == B.Header->Sizes[Idx], "outer shape mismatch for MatMul operation");

        //NOTE(Abid): MatMul dim check
        Assert(A.Header->Sizes[MatMulOutDimLastIdx+1] == B.Header->Sizes[A.Header->Dim-2],
               "inner shape mismatch for MatMul operation");
    }
    else
    {
        // NOTE(Abid): In case of 1-dimensional tensors
        Assert((A.Header->Dim == 1) && (A.Header->Sizes[0] == B.Header->Sizes[0]),
               "inner shape mismatch for MatMul operation");
    }


    // WARNING(Abid): Don't do memory allocation during this call, it ain't nice
    uint32 *NewSizes = (uint32 *)Malloc(A.Header->Dim*sizeof(uint32));
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(uint32));
    NewSizes[A.Header->Dim-1] = B.Header->Sizes[B.Header->Dim-1];

    tensor_i32 ResultTen = _I32Tensor(NewSizes, A.Header->Dim, 0, 0, false);

    // NOTE(Abid): Should we compute the grad in the backward case, or not.
    if(!IS_GRAD_PRESERVE()) ResultTen.Header->ShouldGrad = false;

#if 0
    uint32 *ResultAccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    for(uint32 Idx = 0; Idx < A.Header->Dim; ++Idx) ResultAccessDims[Idx] = (uint32)-1;
    uint32 ResultAccessDimIdx = 0;
#endif

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Check if we are in outer dim and then update
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-2);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            size_t ATenOuterDimOffset = 0;
            size_t BTenOuterDimOffset = 0;
            size_t ResultOuterDimOffset = 0;

            // NOTE(Abid): Offset until the start of the inner dimension
            for(uint32 Idx = 0; Idx < A.Header->Dim-2; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx];
                BTenOuterDimOffset += AccessDims[Idx] * B.Header->Strides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * ResultTen.Header->Strides[Idx];
            }

            // TODO(Abid): Implement the case where its a 1-dimensional matrix
            uint32 CollapsedInnerDim = A.Header->Sizes[A.Header->Dim-1];
            for(uint32 ColIdx = 0; ColIdx < B.Header->Sizes[B.Header->Dim-1]; ++ColIdx)
            {
                size_t BTenOffset = BTenOuterDimOffset + ColIdx*B.Header->Strides[A.Header->Dim-1];
                
                for(uint32 RowIdx = 0; RowIdx < A.Header->Sizes[A.Header->Dim-2]; ++RowIdx)
                {
                    size_t ATenOffset = ATenOuterDimOffset + RowIdx*A.Header->Strides[A.Header->Dim-2];
                    size_t ResultOffset = ColIdx * ResultTen.Header->Strides[ResultTen.Header->Dim-1] +
                                          RowIdx * ResultTen.Header->Strides[ResultTen.Header->Dim-2];
                    int32 Result = 0;
                    for(uint32 Idx = 0; Idx < CollapsedInnerDim; ++Idx)
                    {
                        int32 AVal = A.Storage[ATenOffset + Idx*A.Header->Strides[A.Header->Dim-1]];
                        int32 BVal = B.Storage[BTenOffset + Idx*B.Header->Strides[B.Header->Dim-2]];
                        Result += AVal*BVal;
                    }
                    ResultTen.Storage[ResultOuterDimOffset + ResultOffset] = Result;
                }
            }

            // NOTE(Abid): Go to the previous dimension while setting this one to zero
            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    // NOTE(Abid): Add information for backpropagation
    ResultTen.Header->DerivedOp.TensorOp = op_BinaryMatmul;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32)*2);
    Operands[0] = A;
    Operands[1] = B;
    ResultTen.Header->DerivedOp.Operands = Operands;

    Free(NewSizes);
    Free(AccessDims);
    return ResultTen;
}

internal tensor_i32
I32TenNeg(tensor_i32 A)
{
    tensor_i32 Result = _I32Tensor(A.Header->Sizes, A.Header->Dim, 0, 0, false);

    // NOTE(Abid): We should not compute the grad in the backward case.
    if(!IS_GRAD_PRESERVE()) Result.Header->ShouldGrad = false;

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1);
        if(IsOuterDim)
        {
            // NOTE(Abid): In case we have reached the end of this dimension
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx])
            {
                AccessDims[AccessDimIdx--] = 0;
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
            }
            else ++AccessDimIdx;
        }
        else
        {
            size_t ATenOuterDimOffset = 0;
            size_t ResultOuterDimOffset = 0;

            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx)
            {
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx];
                ResultOuterDimOffset += AccessDims[Idx] * Result.Header->Strides[Idx];
            }

            for(uint32 Idx = 0; Idx < A.Header->Sizes[AccessDimIdx]; ++Idx)
            {
                Result.Storage[ResultOuterDimOffset] = A.Storage[ATenOuterDimOffset]*-1;

                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx];
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx];
            }
            // NOTE(Abid): Go to the previous dimension while setting this one to zero
            AccessDims[AccessDimIdx--] = 0;
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx];
        }
    }

    Result.Header->DerivedOp.TensorOp = op_UnaryNegate;
    tensor_i32 *Operands = (tensor_i32 *)Malloc(sizeof(tensor_i32));
    Operands[0] = A;
    Result.Header->DerivedOp.Operands = Operands;

    Free(AccessDims);
    return Result;
}

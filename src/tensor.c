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

   - Indexing Schemes
   - Implement Broadcast
   - Softmax
   - Sigmoid
   - View
   - Convolution

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

#define _ALLOC_TENSOR_DTYPE(TYPE, DTYPE) \
    tensor32 Result = {0}; \
    \
    size_t StorageSize = 1; \
    for (uint32 i = 0; i < ShapeLength; ++i) { StorageSize *= Shape[i]; } \
    Assert(StorageSize != 0, "Wrong shape given, cannot be zero"); \
    \
    size_t FinalSize = sizeof(tensor_header) + \
                       2*ShapeLength*sizeof(uint32) + /* NOTE(Abid): Stride and Shape */ \
                       2*sizeof(tensor32) + /* NOTE(Abid): For tensor operands */ \
                       StorageSize*sizeof(TYPE); \
    \
    /* NOTE(Abid): Memory mapping */ \
    Result.Header = (tensor_header *)Malloc(FinalSize); \
    Assert(Result.Header, "Storage memory cannot be allocated"); \
    Result.Header->Sizes = (uint32 *)(Result.Header+1); \
    Result.Header->Strides = (uint32 *)(Result.Header->Sizes + ShapeLength); \
    Result.Header->DType = DTYPE; \
    /* Result.Header->IsPersist = false; */ \
    /* NOTE(Abid): Setting whether to compute the backward pass or not */ \
    Result.Header->ShouldGrad = IS_GRAD_PRESERVE() ? true : false; \
    Result.Header->DerivedOp.TensorOp = op_None; \
    Result.Header->DerivedOp.OpContext = NULL; \
    \
    Result.Header->DerivedOp.Operands = (tensor32 *)(Result.Header->Strides + ShapeLength); \
    Result.Storage = (void *)((tensor32 *)Result.Header->DerivedOp.Operands + 2); /* NOTE(Abid): 2 operands by default */ \
    \
    /* NOTE(Abid): Setting Default Values */ \
    Result.Header->Offset = 0; \
    Result.Header->Dim = ShapeLength; \
    memcpy(Result.Header->Sizes, Shape, ShapeLength*sizeof(uint32)); \
    \
    /* NOTE(Abid): Calculate the strides given the tensor shape */ \
    for (uint32 i = 0; i < Result.Header->Dim; ++i) Result.Header->Strides[i] = 1; \
    if (Result.Header->Dim > 1) \
    { \
        for (uint32 i = 0; i < (Result.Header->Dim-1); ++i) \
            for (uint32 j = i+1; j < Result.Header->Dim; ++j) { Result.Header->Strides[i] *= \
                                                                Result.Header->Sizes[j]; } \
    } \
    \
    if(Intialize) \
    { \
        /* NOTE(Abid): Check if the DataLength makes sense with the shape */ \
        Assert(DataLength == StorageSize, "Mismatch of data and tensor shape"); \
        memcpy(Result.Storage, Data, StorageSize*sizeof(TYPE)); \
    } \
    \
    return Result; \

#define F32Tensor(Shape, Data) _F32Tensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data), true)
internal inline tensor32
_F32Tensor(uint32 *Shape, uint32 ShapeLength, float32 *Data, size_t DataLength, boolean Intialize)
{ _ALLOC_TENSOR_DTYPE(float32, dtype_float32); }


#define I32Tensor(Shape, Data) _I32Tensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data), true)
internal inline tensor32
_I32Tensor(uint32 *Shape, uint32 ShapeLength, int32 *Data, size_t DataLength, boolean Intialize)
{ _ALLOC_TENSOR_DTYPE(int32, dtype_int32); }

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

#define _PRINT_DTYPE(TEN_NAME, TYPE, PRINT_FORMAT) \
    printf(TEN_NAME); \
    printf(" -> shape ("); \
    for (uint32 i = 0; i < (TenHeader->Dim-1); ++i) { printf("%d,", TenHeader->Sizes[i]); } \
    printf("%d) :=\n",TenHeader->Sizes[TenHeader->Dim-1]); \
    \
    uint32 *AccessDims = (uint32 *)Calloc(TenHeader->Dim, sizeof(uint32)); \
    int32 AccessDimIdx = 0; \
    \
    while(AccessDimIdx >= 0) \
    { \
        if(PrintCount >= MaxPrintWidth) { printf("\n"); PrintCount = 0; } \
        if(_PrintIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader, &PrintCount)) continue; \
        \
        /* NOTE(Abid): Main printing of the last dimension */ \
        printf("["); ++PrintCount; \
        ++PrintCount; \
        \
        for (uint32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx) \
        { \
            AccessDims[AccessDimIdx] = Idx; \
            \
            size_t Pos = 0; \
            for (uint32 PosIdx = 0; PosIdx < Tensor.Header->Dim; ++PosIdx) Pos += AccessDims[PosIdx] * Tensor.Header->Strides[PosIdx]; \
            \
            printf(PRINT_FORMAT, ((TYPE *)Tensor.Storage)[Pos]); \
            ++PrintCount; \
            if(AccessDims[AccessDimIdx] < (TenHeader->Sizes[AccessDimIdx]-1)) { printf(", "); ++PrintCount; } \
        } \
        printf("]"); ++PrintCount; \
        if(AccessDims[AccessDimIdx-1] < (TenHeader->Sizes[AccessDimIdx-1]-1)) { printf(", "); ++PrintCount; } \
        \
        /* NOTE(Abid): Go to the previous dimension while setting this one to zero */ \
        AccessDims[AccessDimIdx--] = 0; \
        ++AccessDims[AccessDimIdx]; \
    } \
    \
    printf("\n\n"); \
    \
    Free(AccessDims); \

internal void
PrintTensor32(tensor32 Tensor)
{
    int32 MaxPrintWidth = 20;
    int32 PrintCount = 0;

    tensor_header *TenHeader = Tensor.Header;

    // NOTE(Abid): Print the header and data
    tensor_dtype DType = TenHeader->DType;
    switch(DType)
    {
        case dtype_int32: { _PRINT_DTYPE("tensor i32", int32, "%d"); } break;
        case dtype_float32: { _PRINT_DTYPE("tensor f32", float, "%f"); } break;
        default: Assert(0, "invalid code path");
    }
}

// =======================================
// NOTE(Abid): Math Operations tensor_i32
// =======================================

#define _BINARY_ELEMENTWISE_OP(OPERATION, OP_TYPE, A_TYPE, B_TYPE, RES_TYPE) \
    if(!IS_GRAD_PRESERVE()) Result.Header->ShouldGrad = false; \
    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32)); \
    int32 AccessDimIdx = 0; \
    \
    while(AccessDimIdx >= 0) \
    { \
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1); \
        if(IsOuterDim) \
        { \
            if(AccessDims[AccessDimIdx] == A.Header->Sizes[AccessDimIdx]) \
            { \
                AccessDims[AccessDimIdx--] = 0; \
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx]; \
            } \
            else ++AccessDimIdx; \
        } \
        else \
        { \
            size_t ATenOuterDimOffset = 0; \
            size_t BTenOuterDimOffset = 0; \
            size_t ResultOuterDimOffset = 0; \
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx) \
            { \
                ATenOuterDimOffset += AccessDims[Idx] * A.Header->Strides[Idx]; \
                BTenOuterDimOffset += AccessDims[Idx] * B.Header->Strides[Idx]; \
                ResultOuterDimOffset += AccessDims[Idx] * Result.Header->Strides[Idx]; \
            } \
            for (uint32 Idx = 0; Idx < A.Header->Sizes[AccessDimIdx]; ++Idx) \
            { \
                ((RES_TYPE *)Result.Storage)[ResultOuterDimOffset] = (RES_TYPE)(((A_TYPE *)A.Storage)[ATenOuterDimOffset] OPERATION \
                                                                                ((B_TYPE *)B.Storage)[BTenOuterDimOffset]); \
                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx]; \
                BTenOuterDimOffset += B.Header->Strides[AccessDimIdx]; \
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx]; \
            } \
            AccessDims[AccessDimIdx--] = 0; \
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx]; \
        } \
    } \
    \
    Result.Header->DerivedOp.TensorOp = OP_TYPE; \
    tensor32 *Operands = Result.Header->DerivedOp.Operands; \
    Operands[0] = A; \
    Operands[1] = B; \
    \
    Free(AccessDims); \
    return Result; \

#define _BINARY_ELEMENTWISE_OP_DTYPE(A, B, OPERATION, OP_TYPE) \
    Assert(IsShapeEqual(A.Header, B.Header), "shape mismatch for" #OPERATION "operation"); \
    tensor32 Result = {0}; \
    if(A.Header->DType == B.Header->DType) \
    { \
        if(A.Header->DType == dtype_float32) \
        { \
            Result = _F32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false); \
            _BINARY_ELEMENTWISE_OP(OPERATION, OP_TYPE, float32, float32, float32); \
        } \
        else if(A.Header->DType == dtype_int32) \
        { \
            if(OP_TYPE == op_BinaryDiv) \
            { \
                Result = _F32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false); \
                _BINARY_ELEMENTWISE_OP(OPERATION, OP_TYPE, int32, int32, float32); \
            } \
            else \
            { \
                Result = _I32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false); \
                _BINARY_ELEMENTWISE_OP(OPERATION, OP_TYPE, int32, int32, int32); \
            } \
        } \
        else Assert(0, "invalid code path"); \
    } \
    if(B.Header->DType == dtype_float32) \
    { \
        Result = _F32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false); \
        _BINARY_ELEMENTWISE_OP(OPERATION, OP_TYPE, int32, float32, float32); \
    } \
    else if(A.Header->DType == dtype_float32) \
    { \
        Result = _F32Tensor(A.Header->Sizes, B.Header->Dim, 0, 0, false); \
        _BINARY_ELEMENTWISE_OP(OPERATION, OP_TYPE, float32, int32, float32); \
    } \
    else Assert(0, "invalid code path"); \
    \
    return Result; \


// NOTE(Abid): Elementwise Operations
internal tensor32
T32Add(tensor32 A, tensor32 B) { _BINARY_ELEMENTWISE_OP_DTYPE(A, B, +, op_BinaryAdd); }

internal tensor32
T32Sub(tensor32 A, tensor32 B) { _BINARY_ELEMENTWISE_OP_DTYPE(A, B, -, op_BinarySub); }

internal tensor32
T32Mul(tensor32 A, tensor32 B) { _BINARY_ELEMENTWISE_OP_DTYPE(A, B, *, op_BinaryMul); }

internal tensor32
T32Div(tensor32 A, tensor32 B) { _BINARY_ELEMENTWISE_OP_DTYPE(A, B, /, op_BinaryDiv); }

#if 0
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
    tensor_i32 *Operands = ResultTen.Header->DerivedOp.Operands;
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
    tensor_i32 *Operands = ResultTen.Header->DerivedOp.Operands;
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
    tensor_i32 *Operands = ResultTen.Header->DerivedOp.Operands;
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
    tensor_i32 *Operands = Result.Header->DerivedOp.Operands;
    Operands[0] = A;
    Result.Header->DerivedOp.Operands = Operands;

    Free(AccessDims);
    return Result;
}
#endif

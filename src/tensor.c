/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 10:19:09 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

/*
   TODO(Abid):

   - Add broadcast information for `Backward` computation.
   - Here is an important question: if we do a transpose and change the layout of the data
     then should the grad also be changed?
   - The stride calculation is trivial and makes operation on inplace tranposed tensor slow
   - Write a more efficient math ops routines for when `IsContiguous = true` in the tensor

   - Implement Broadcast
   - Indexing Schemes
   - Softmax
   - Sigmoid
   - View
   - Convolution (https://github.com/vdumoulin/conv_arithmetic)

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

#define __ALLOC_TENSOR_DTYPE(TYPE, StoreGrad) \
    tensor32 Result = {0}; \
    \
    size_t DataSize = 1; \
    for (uint32 i = 0; i < ShapeLength; ++i) { DataSize *= Shape[i]; } \
    Assert(DataSize != 0, "wrong shape given, cannot be zero"); \
    \
    size_t FinalSize = sizeof(tensor_header) + \
                       2*ShapeLength*sizeof(uint32) + /* NOTE(Abid): For Stride, Shape */ \
                       ShapeLength*sizeof(uint32) + /* NOTE(Abid): For SizesAccessPtr */ \
                       2*sizeof(tensor32) + /* NOTE(Abid): For tensor operands */ \
                       ((int32)StoreGrad + 1)*DataSize*sizeof(TYPE); /* StoreGrad for backprop */ \
    \
    /* NOTE(Abid): Memory mapping */ \
    Result.Header = (tensor_header *)Malloc(FinalSize); \
    Assert(Result.Header, "storage memory cannot be allocated"); \
    Result.Header->Sizes = (uint32 *)(Result.Header+1); \
    Result.Header->Strides = (uint32 *)(Result.Header->Sizes + ShapeLength); \
    Result.Header->AccessSizes = (uint32 *)(Result.Header->Strides + ShapeLength); \
    Result.Header->DType = dtype_##TYPE; \
    Result.Header->IsContiguous = true; \
    Result.Header->StorageNumElements = DataSize; \
    /* NOTE(Abid): Setting whether to compute the backward pass or not */ \
    Result.Header->ShouldGrad = IS_GRAD_PRESERVE(); \
    Result.Header->GradStorageInit = IS_GRAD_PRESERVE(); \
    Result.Header->DerivedOp.TensorOp = op_None; \
    Result.Header->DerivedOp.OpContext = NULL; /* TODO(Abid): This memory gets created during math ops, which is not nice
                                                              Figure out a max ceiling and always allocate that at init */ \
    \
    Result.Header->DerivedOp.Operands = (tensor32 *)(Result.Header->AccessSizes + ShapeLength); \
    Result.Data = (void *)((tensor32 *)Result.Header->DerivedOp.Operands + 2); /* 2 operands by default */ \
    if(StoreGrad) { \
        Result.Grad = ((TYPE *)Result.Data) + DataSize; \
        memset(Result.Grad, 0, DataSize*sizeof(TYPE)); \
    } \
    else Result.Grad = NULL; \
    \
    /* NOTE(Abid): Setting Default Values */ \
    Result.Header->Offset = 0; \
    Result.Header->Dim = ShapeLength; \
    memcpy(Result.Header->Sizes, Shape, ShapeLength*sizeof(uint32)); \
    \
    /* NOTE(Abid): Calculate the strides given the tensor shape */ \
    for(uint32 i = 0; i < Result.Header->Dim; ++i) { \
        if(Result.Header->Sizes[i] == 1) Result.Header->Strides[i] = 0; \
        else Result.Header->Strides[i] = 1; \
    } \
    if(Result.Header->Dim > 1) { \
        /* NOTE(Abid): If the size in a dim is 1, then the stride will be zero */ \
        for(uint32 i = 0; i < (Result.Header->Dim-1); ++i) \
                for(uint32 j = i+1; j < Result.Header->Dim; ++j) { Result.Header->Strides[i] *= \
                                                                   Result.Header->Sizes[j]; } \
    } \
    \
    if(Data) { \
        /* NOTE(Abid): Check if the DataLength makes sense with the shape */ \
        Assert(DataLength == DataSize, "data and tensor shape mismatch"); \
        memcpy(Result.Data, Data, DataSize*sizeof(TYPE)); \
    } \
    \
    return Result; \

#define T32Data(Shape, Data, TYPE) _##TYPE##AllocTensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data))
#define T32Empty(Shape, TYPE) _##TYPE##AllocTensor(Shape, ArrayLength(Shape), 0, 0)

internal inline tensor32
_float32AllocTensor(uint32 *Shape, uint32 ShapeLength, float32 *Data, size_t DataLength)
{ __ALLOC_TENSOR_DTYPE(float32, false); }

internal inline tensor32
_int32AllocTensor(uint32 *Shape, uint32 ShapeLength, int32 *Data, size_t DataLength)
{ __ALLOC_TENSOR_DTYPE(int32, false); }

#define SHAPE(...) __VA_ARGS__
#define ARRAY(...) __VA_ARGS__
#define TensorFromArrayLiteral(NAME, DTYPE, Shape, Values) \
    {0}; \
    do \
    { \
        uint32 Shape_Arr[] = { Shape }; \
        DTYPE Values_Arr[] = { Values }; \
        NAME = _##DTYPE##AllocTensor(Shape_Arr, ArrayLength(Shape_Arr), \
                                     Values_Arr, ArrayLength(Values_Arr)); \
    } while(0);

internal inline size_t
GetStorageSize(uint32 *Sizes, uint32 Dim)
{
    size_t Result = 1;

    for (uint32 Idx = 0; Idx < Dim; ++Idx)
    {
        Assert(Sizes[Idx], "tensor shape cannot be zero")
        Result *= Sizes[Idx];
    }

    return Result;
}

internal inline boolean
IsArrayEqual(uint32 *Array1, uint32 *Array2, uint32 Array1Length, uint32 Array2Length)
{
    if(Array1Length != Array2Length) return false;

    for (uint32 Idx = 0; Idx < Array1Length; ++Idx)
        if (Array1[Idx] != Array2[Idx]) return false;

    return true;
}

internal inline boolean
IsShapeEqual(tensor_header *A, tensor_header *B) { return IsArrayEqual(A->Sizes, B->Sizes, A->Dim, B->Dim); }

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

#define _PRINT_DTYPE(TEN_NAME, STORAGE, TYPE, PRINT_FORMAT) \
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
            printf(PRINT_FORMAT, ((TYPE *)Tensor.STORAGE)[Pos]); \
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
        case dtype_int32: { _PRINT_DTYPE("tensor i32", Data, int32, "%d"); } break;
        case dtype_float32: { _PRINT_DTYPE("tensor f32", Data, float, "%.3f"); } break;
        default: Assert(0, "invalid code path");
    }
}

internal inline void
T32SetElementsInPlace(tensor32 A, float32 Value)
{
    switch (A.Header->DType)
    {
        case dtype_float32:
        {
            float32 TempVal = Value;
            float32 *StoragePtr = (float32 *)A.Data;
            for(int32 Idx = 0; Idx < A.Header->StorageNumElements; ++Idx)
                StoragePtr[Idx] = TempVal;
        } break;
        case dtype_int32:
        {
            int32 TempVal = (int32)Value;
            int32 *StoragePtr = (int32 *)A.Data;
            for(int32 Idx = 0; Idx < A.Header->StorageNumElements; ++Idx)
                StoragePtr[Idx] = TempVal;
        } break;
        default: Assert(0, "invalid code path");
    }
}

// TODO(Abid): Not sure about this, could be bad for perf. if chained
#define CallOnGradStorage(Tensor, Function) \
    do \
    { \
        void *Grad = Tensor.Grad; \
        void *Data = Tensor.Data; \
        tensor_dtype DType = Tensor.Header->DType; \
        Tensor.Header->DType = dtype_float32; \
        Tensor.Data = Grad; \
        Tensor.Grad = Data; \
        Function; \
        Tensor.Data = Data; \
        Tensor.Grad = Grad; \
        Tensor.Header->DType = DType; \
    } while(0)


// =======================================
// NOTE(Abid): Math Operations
// =======================================

// NOTE(Abid): 1. Main routines for elementwise binary operations.
internal inline void
_CheckEndofDimAndUpdate(tensor_header *AHeader, tensor_header *BHeader, tensor_header *ResultHeader,
                        size_t *AOffset, size_t *BOffset, size_t *ResultOffset) {
    uint32 DimIdx = 1;
    // NOTE(Abid): If we've reached the end of the this dim in result tensor
    while(ResultHeader->AccessSizes[ResultHeader->Dim- DimIdx] == ResultHeader->Sizes[ResultHeader->Dim- DimIdx]) {
        // NOTE(Abid): If there is nothing left to calculate
        if(ResultHeader->Dim - DimIdx == 0) break;

        int32 ACurrentDim = (int32)AHeader->Dim - DimIdx;
        int32 BCurrentDim = (int32)BHeader->Dim - DimIdx;
        int32 ResultCurrentDim = (int32)ResultHeader->Dim - DimIdx;

        if(ACurrentDim <= 0) { 
            *AOffset = 0;
            AHeader->AccessSizes[0] = 0;
            // NOTE(Abid): Then B tensor is the one with greater dim
            BHeader->AccessSizes[BCurrentDim] = 0;
            ++BHeader->AccessSizes[BCurrentDim-1];
            *BOffset -= BHeader->Strides[BCurrentDim]*BHeader->Sizes[BCurrentDim];
            *BOffset += BHeader->Strides[BCurrentDim-1];
        }
        else if(BCurrentDim <= 0) {
            *BOffset = 0;
            BHeader->AccessSizes[0] = 0;
            // NOTE(Abid): Then A tensor is the one with greater dim
            AHeader->AccessSizes[ACurrentDim] = 0;
            ++AHeader->AccessSizes[ACurrentDim-1];
            *AOffset -= AHeader->Strides[ACurrentDim]*AHeader->Sizes[ACurrentDim];
            *AOffset += AHeader->Strides[ACurrentDim-1];
        }
        else {
            // NOTE(Abid): Both tensors have dim in this case
            AHeader->AccessSizes[ACurrentDim] = 0;
            ++AHeader->AccessSizes[ACurrentDim-1];
            *AOffset -= AHeader->Strides[ACurrentDim]*AHeader->Sizes[ACurrentDim];
            *AOffset += AHeader->Strides[ACurrentDim-1];

            BHeader->AccessSizes[BCurrentDim] = 0;
            ++BHeader->AccessSizes[BCurrentDim-1];
            *BOffset -= BHeader->Strides[BCurrentDim]*BHeader->Sizes[BCurrentDim];
            *BOffset += BHeader->Strides[BCurrentDim-1];
        }

        ResultHeader->AccessSizes[ResultCurrentDim] = 0;
        ++ResultHeader->AccessSizes[ResultCurrentDim-1];
        *ResultOffset -= ResultHeader->Strides[ResultCurrentDim]*ResultHeader->Sizes[ResultCurrentDim];
        *ResultOffset += ResultHeader->Strides[ResultCurrentDim-1];
        ++(DimIdx);
    }
}

/* TODO(Abid): For optimization, if the tensor is contiguous, then the next element is just +1, thefore, we can
 *             run a for loop instead of calculating the next element each time, this will work since for shape=1,
 *             the stride will be zero. */
#define __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, A_DTYPE, B_DTYPE, R_DTYPE, OP) \
    while(ResDataLeft >= 0) { \
        *((R_DTYPE *)Result.Data + ResultOffset) = (R_DTYPE)(*((A_DTYPE *)A.Data + AOffset) OP *((B_DTYPE *)B.Data + BOffset)); \
        int32 ACurrentDim = (int32)A.Header->Dim - DimIdx; \
        int32 BCurrentDim = (int32)B.Header->Dim - DimIdx; \
        int32 ResultCurrentDim = (int32)Result.Header->Dim - DimIdx; \
        AOffset += A.Header->Strides[ACurrentDim]; ++A.Header->AccessSizes[ACurrentDim]; \
        BOffset += B.Header->Strides[BCurrentDim]; ++B.Header->AccessSizes[BCurrentDim]; \
        ResultOffset += Result.Header->Strides[ResultCurrentDim]; ++Result.Header->AccessSizes[ResultCurrentDim]; \
        \
        _CheckEndofDimAndUpdate(A.Header, B.Header, Result.Header, &AOffset, &BOffset, &ResultOffset); \
        --ResDataLeft; \
    }

#define __BIN_ELEMENTWISE_OP(A, B, Result, OP) \
    /* NOTE(Abid): Assert here that result does match the highest dim and sizes (broadcast size as well) */ \
    uint32 GreaterDim = 0; \
    if (A.Header->Dim > B.Header->Dim) GreaterDim = A.Header->Dim; \
    else GreaterDim = B.Header->Dim; \
    Assert(Result.Header->Dim == GreaterDim, "result-operand(s) dimension mismatch"); \
    for(uint32 Idx = 1; Idx <= GreaterDim; Idx++) { \
        uint32 ASize = 0; \
        uint32 BSize = 0; \
        if((int32)(A.Header->Dim - Idx) >= 0) ASize = A.Header->Sizes[A.Header->Dim - Idx]; \
        if((int32)(B.Header->Dim - Idx) >= 0) BSize = B.Header->Sizes[B.Header->Dim - Idx]; \
        /* NOTE(Abid): Check for shape alignment for the operands */ \
        Assert(((ASize > BSize) && ((BSize == 1) || (BSize == 0))) || \
               ((BSize > ASize) && ((ASize == 1) || (ASize == 0))) || \
               (BSize == ASize), "operand(s) shape mismatch"); \
        uint32 GreaterSize = ASize > BSize ? ASize : BSize; \
        Assert(Result.Header->Sizes[Result.Header->Dim - Idx] == GreaterSize, "result-operand(s) shape mismatch"); \
    } \
    \
    /* NOTE(Abid): Initialize variables */ \
    int64 ResDataLeft = Result.Header->StorageNumElements; \
    uintptr AOffset = 0; \
    uintptr BOffset = 0; \
    uintptr ResultOffset = 0; \
    uint32 DimIdx = 1; \
    \
    memset(Result.Header->AccessSizes, 0, Result.Header->Dim*sizeof(uint32)); \
    memset(A.Header->AccessSizes, 0, A.Header->Dim*sizeof(uint32)); \
    memset(B.Header->AccessSizes, 0, B.Header->Dim*sizeof(uint32)); \
    \
    bin_op_dtypes OpDTypes = bin_op_dtypes_all_float; /* Assuming all float32 types initially. */ \
    if(A.Header->DType == B.Header->DType) { if(A.Header->DType == dtype_int32) OpDTypes = 2; } \
    else if(A.Header->DType == dtype_int32) { OpDTypes = 6; } /* A is int32, B is float32 */ \
    else OpDTypes = 4; /* A is float32, B is int32 */ \
    \
    if(Result.Header->DType == dtype_int32) OpDTypes += 1; /* Determining the result data type */ \
    \
    switch(OpDTypes) { \
        case bin_op_dtypes_all_float:       { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, float32, float32, float32, OP); } break; \
        case bin_op_dtypes_float_float_int: { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, float32, float32, int32, OP);   } break; \
        case bin_op_dtypes_int_int_float:   { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, int32, int32, float32, OP);     } break; \
        case bin_op_dtypes_all_int:         { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, int32, int32, int32, OP);       } break; \
        case bin_op_dtypes_float_int_float: { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, float32, int32, float32, OP);   } break; \
        case bin_op_dtypes_float_int_int:   { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, float32, int32, int32, OP);     } break; \
        case bin_op_dtypes_int_float_float: { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, int32, float32, float32, OP);   } break; \
        case bin_op_dtypes_int_float_int:   { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, int32, float32, int32, OP);     } break; \
        default:                            InvalidCodePath; \
    }

typedef enum {
    bin_op_dtypes_all_float = 0,
    bin_op_dtypes_float_float_int = 1,
    bin_op_dtypes_int_int_float = 2,
    bin_op_dtypes_all_int = 3,

    bin_op_dtypes_float_int_float = 4,
    bin_op_dtypes_float_int_int = 5,
    bin_op_dtypes_int_float_float = 6,
    bin_op_dtypes_int_float_int = 7,
} bin_op_dtypes;
internal void T32Add(tensor32 A, tensor32 B, tensor32 Result) { __BIN_ELEMENTWISE_OP(A, B, Result, +); }
internal void T32Sub(tensor32 A, tensor32 B, tensor32 Result) { __BIN_ELEMENTWISE_OP(A, B, Result, -); }
internal void T32Mul(tensor32 A, tensor32 B, tensor32 Result) { __BIN_ELEMENTWISE_OP(A, B, Result, *); }
internal void T32Div(tensor32 A, tensor32 B, tensor32 Result) { __BIN_ELEMENTWISE_OP(A, B, Result, /); }
#undef __BIN_ELEMENTWISE_OP_DTYPE
#undef __BIN_ELEMENTWISE_OP

internal inline uint32
GetIndex(uint32 Dim, int32 Index)
{
    int32 Result = (int32)Dim + Index;
    if(Result < 0) Result = 0;
    Assert(Result < (int32)(2*Dim), "index out of bounds");
    return Result % Dim;
}

// NOTE(Abid): 2. Main routines for MatMul operation

// TODO(Abid): Improve this routine by reshaping all operand tensors to be the same dim (expanding all with shape 1) and stride 0.
//             For now, let's just do a simple hack with a few conditionals.
internal void 
T32MatMul(tensor32 A, tensor32 B, tensor32 Result) {
    // NOTE(Abid): Assert greater dim is the same as the result tensors' dim
    uint32 GreaterDim = 0;
    uint32 LesserDim = 0;
    uint32 ResLastBroadDim = 0;

    if (A.Header->Dim > B.Header->Dim) { GreaterDim = A.Header->Dim; LesserDim = B.Header->Dim; }
    else { GreaterDim = B.Header->Dim; LesserDim = A.Header->Dim; }
    if(LesserDim > 1) ResLastBroadDim = 3;
    else ResLastBroadDim = 2;
    Assert(Result.Header->Dim == GreaterDim, "result-operand(s) dimension mismatch");

    uint32 ALastIdx = GetIndex(A.Header->Dim, -1);
    uint32 ASecondLastIdx = GetIndex(A.Header->Dim, -2);

    uint32 BLastIdx = GetIndex(B.Header->Dim, -1);
    uint32 BSecondLastIdx = GetIndex(B.Header->Dim, -2);
    /* NOTE(Abid): Assert here the tensor operand(s) can be matrix-multiplied */
    Assert(A.Header->Sizes[ALastIdx] == B.Header->Sizes[BSecondLastIdx],
           "operand(s) shape mismatch for MatMul operation");

    // NOTE(Abid): Assert the shapes of the result tensor match with the MatMul operation's required shape.
    //             1. In case, the lesserDim is 1, then we expect the last dimension of result to be 1 as well
    //             2. Otherwise, the dimensions must match with the result of the MatMul operation.
    Assert(((LesserDim > 1) && (
                                    (A.Header->Sizes[GetIndex(A.Header->Dim, -2)] ==
                                     Result.Header->Sizes[GetIndex(Result.Header->Dim, -2)]) &&
                                    (B.Header->Sizes[GetIndex(B.Header->Dim, -1)] ==
                                     Result.Header->Sizes[GetIndex(Result.Header->Dim, -1)])
                               )
           ) || ((LesserDim == 1) && (B.Header->Dim == 1 ? Result.Header->Sizes[GetIndex(Result.Header->Dim, - 1)] == 1 :
                                                           Result.Header->Sizes[GetIndex(Result.Header->Dim, - 2)] == 1)),
           "result-operand(s) shape mismatch for MatMul operation");

    // NOTE(Abid): Assert broadcast shape match
    for(uint32 Idx = 3; Idx <= GreaterDim; ++Idx) {
        uint32 ASize = 0; uint32 BSize = 0;
        if((int32)(A.Header->Dim - Idx) >= 0) ASize = A.Header->Sizes[A.Header->Dim - Idx];
        if((int32)(B.Header->Dim - Idx) >= 0) BSize = B.Header->Sizes[B.Header->Dim - Idx];
        /* NOTE(Abid): Check for shape alignment for the operands */
        boolean IsAGreater = ASize > BSize;
        Assert((IsAGreater && ((BSize == 1) || (BSize == 0))) ||
               ((BSize > ASize) && ((ASize == 1) || (ASize == 0))) ||
               (BSize == ASize), "operand(s) shape mismatch");
        uint32 GreaterSize = IsAGreater ? ASize : BSize;
        Assert(Result.Header->Sizes[Result.Header->Dim - Idx] == GreaterSize, "result-operand(s) shape mismatch");
    }

    /* NOTE(Abid): Initialize variables */
    int64 ResDataLeft = Result.Header->StorageNumElements;
    uintptr AOffset = 0;
    uintptr BOffset = 0;
    uintptr ResultOffset = 0;

    memset(Result.Header->AccessSizes, 0, Result.Header->Dim*sizeof(uint32));
    memset(A.Header->AccessSizes, 0, A.Header->Dim*sizeof(uint32));
    memset(B.Header->AccessSizes, 0, B.Header->Dim*sizeof(uint32));

    int32 IsBroadcastDim = false; // Last if we count from the right
    while(ResDataLeft) {
        // NOTE(Abid): We are progressing row-wise on the result tensor
        float32 DotResult = 0;
        for(uint32 Idx = 0; Idx < A.Header->Sizes[ALastIdx]; ++Idx) {
            DotResult += (float32)(*((float32 *)A.Data + AOffset) * *((float32 *)B.Data + BOffset));

            AOffset += A.Header->Strides[ALastIdx];
            BOffset += B.Header->Strides[BSecondLastIdx];
        }
        *((float32 *)Result.Data + ResultOffset) = DotResult;
        ++Result.Header->AccessSizes[Result.Header->Dim-1];

        AOffset -= A.Header->Strides[ALastIdx]*A.Header->Sizes[ALastIdx];
        BOffset -= B.Header->Strides[BSecondLastIdx]*B.Header->Sizes[BSecondLastIdx];
        if(--ResDataLeft == 0) break;

        // NOTE(Abid): If Result dim = 1, then it won't go beyond this point.
        // NOTE(Abid): In case we have reached the end of the result tensor row
        if(Result.Header->AccessSizes[Result.Header->Dim-1] == Result.Header->Sizes[Result.Header->Dim-1]) {
            ResultOffset -= Result.Header->Strides[Result.Header->Dim-1]*(Result.Header->Sizes[Result.Header->Dim-1]-1);
            Result.Header->AccessSizes[Result.Header->Dim-1] = 0;

            if(ResLastBroadDim > 2) {
                ++Result.Header->AccessSizes[Result.Header->Dim-2];
                if(Result.Header->AccessSizes[Result.Header->Dim-2] == Result.Header->Sizes[Result.Header->Dim-2]) {
                    // NOTE(Abid): We've reached the end, begin broadcast semantics for matrix...
                    IsBroadcastDim = true;

                    // NOTE(Abid): Zero out matrix multiplication dimensions, and take offsets to the start.
                    AOffset -= A.Header->Strides[ASecondLastIdx]*(A.Header->Sizes[ASecondLastIdx]-1);
                    A.Header->AccessSizes[ALastIdx] = 0; A.Header->AccessSizes[ASecondLastIdx] = 0;

                    BOffset -= B.Header->Strides[BLastIdx]*(B.Header->Sizes[BLastIdx]-1);
                    B.Header->AccessSizes[BLastIdx] = 0; B.Header->AccessSizes[BSecondLastIdx] = 0;

                    Result.Header->AccessSizes[Result.Header->Dim - (ResLastBroadDim-1)] = 0;
                    ResultOffset -= Result.Header->Strides[Result.Header->Dim-2]*(Result.Header->Sizes[Result.Header->Dim-2]-1);
                } else {
                    AOffset += A.Header->Strides[ASecondLastIdx]; // Going to the next column element (start of next row).
                    BOffset -= B.Header->Strides[BLastIdx]*(B.Header->Sizes[BLastIdx]-1);
                    ResultOffset += Result.Header->Strides[Result.Header->Dim-2];
                }
            }
            else {
                // NOTE(Abid): Result has only 1 matmul dimension, the rest is for broadcast.
                IsBroadcastDim = true;

                if(A.Header->Dim > B.Header->Dim) {
                    // NOTE(Abid): BOffset needs to be zero'd out
                    AOffset -= A.Header->Strides[ASecondLastIdx]*(A.Header->Sizes[ASecondLastIdx]-2);
                    A.Header->AccessSizes[ALastIdx] = 0; A.Header->AccessSizes[ASecondLastIdx] = 0;
                    BOffset = 0;
                } else {
                    // NOTE(Abid): AOffset needs to be zero'd out
                    BOffset -= B.Header->Strides[BLastIdx]*(B.Header->Sizes[BLastIdx]-2);
                    B.Header->AccessSizes[BLastIdx] = 0; B.Header->AccessSizes[BSecondLastIdx] = 0;
                    AOffset = 0;
                }


                Result.Header->AccessSizes[Result.Header->Dim - (ResLastBroadDim-1)] = 0;
            }
        } else {
            BOffset += B.Header->Strides[B.Header->Dim-1]; // Going to the next row element (start of next column).
            ResultOffset += Result.Header->Strides[Result.Header->Dim-1];
        }

        if(IsBroadcastDim) {
            for(uint32 CurrentBroadIter = 0; CurrentBroadIter < ResLastBroadDim; ++CurrentBroadIter) {
                int32 CurrenntDimIdx = ResLastBroadDim + CurrentBroadIter;
                int32 CurResBroadIdx = (int32)Result.Header->Dim - CurrenntDimIdx;
                int32 CurABroadIdx = (int32)A.Header->Dim - CurrenntDimIdx;
                int32 CurBBroadIdx = (int32)B.Header->Dim - CurrenntDimIdx;

                if(Result.Header->AccessSizes[CurResBroadIdx]+1 ==
                   Result.Header->Sizes[CurResBroadIdx]) {
                    // NOTE(Abid): We are at the end of this broadcast dimension
                    Result.Header->AccessSizes[CurResBroadIdx] = 0;
                    ResultOffset -= Result.Header->Strides[CurResBroadIdx]*Result.Header->Sizes[CurResBroadIdx];
                    if(CurABroadIdx >= 0) {
                        A.Header->AccessSizes[CurABroadIdx] = 0;
                        AOffset -= A.Header->Strides[CurABroadIdx]*A.Header->Sizes[CurABroadIdx];
                    } else {
                        memset(A.Header->AccessSizes, 0, A.Header->Dim*sizeof(uint32));
                        AOffset = 0;
                    }
                    if(CurBBroadIdx >= 0) {
                        B.Header->AccessSizes[CurBBroadIdx] = 0;
                        BOffset -= B.Header->Strides[CurBBroadIdx]*B.Header->Sizes[CurBBroadIdx];
                    } else {
                        memset(A.Header->AccessSizes, 0, A.Header->Dim*sizeof(uint32));
                        BOffset = 0;
                    }
                } else {
                    ++Result.Header->AccessSizes[CurResBroadIdx];
                    ResultOffset += Result.Header->Strides[CurResBroadIdx];
                    if(CurABroadIdx >= 0) {
                        ++A.Header->AccessSizes[CurABroadIdx];
                        AOffset += A.Header->Strides[CurABroadIdx];
                    }
                    if(CurBBroadIdx >= 0) {
                        ++B.Header->AccessSizes[CurBBroadIdx];
                        BOffset += B.Header->Strides[CurBBroadIdx];
                    }
                    break;
                }
            }
            IsBroadcastDim = false;
        }
    }
}

// NOTE(Abid): 3. Main routines for Unary operations
// TODO(Abid): Implement T32ElementSet() and T32ElementOp here
#if 0
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
                Result.Data[ResultOuterDimOffset] = A.Data[ATenOuterDimOffset]*-1;

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

// =======================================
// NOTE(Abid): Reshaping Operations
// =======================================

// TODO(Abid): Must take care of non-contiguous tensors, especially when flattening
#define _TRANSPOSED_COPY_TENSOR_DTYPE(DTYPE) \
    while(AccessDimIdx >= 0) \
    { \
        /* NOTE(Abid): Check if we are in outer dim and then update */ \
        boolean IsOuterDim = AccessDimIdx < (int32)(A.Header->Dim-1); \
        if(IsOuterDim) \
        { \
            /* NOTE(Abid): In case we have reached the end of this dimension */ \
            if(AccessDims[AccessDimIdx] == NewSizes[AccessDimIdx]) \
            { \
                AccessDims[AccessDimIdx--] = 0; \
                if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx]; \
            } \
            else ++AccessDimIdx; \
        } \
        else \
        { \
            uint32 ATenOuterDimOffset = 0; \
            uint32 ResultOuterDimOffset = 0; \
            for(uint32 Idx = 0; Idx < A.Header->Dim-1; ++Idx) \
            { \
                ATenOuterDimOffset += AccessDims[Idx] * NewStrides[Idx]; \
                ResultOuterDimOffset += AccessDims[Idx] * ResultTen.Header->Strides[Idx]; \
            } \
            for(uint32 Idx = 0; Idx < NewSizes[AccessDimIdx]; ++Idx) \
            { \
                ((DTYPE *)ResultTen.Data)[ResultOuterDimOffset] = ((DTYPE *)A.Data)[ATenOuterDimOffset]; \
                ATenOuterDimOffset += NewStrides[AccessDimIdx]; \
                ResultOuterDimOffset += ResultTen.Header->Strides[AccessDimIdx]; \
            } \
            \
            AccessDims[AccessDimIdx--] = 0; \
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx]; \
        } \
    }

internal void
T32TransposeInPlace(tensor32 A, int32 Dim1, int32 Dim2)
{
    // TODO(Abid): Properly check if the transpose could make the tensor contiguous again.
    Assert(A.Header->IsContiguous, "cannot transpose non-contiguous tensor");

    // NOTE(Abid): Support for negative indexing.
    Dim1 = (A.Header->Dim + Dim1) % A.Header->Dim;
    Dim2 = (A.Header->Dim + Dim2) % A.Header->Dim;

    Assert(((int32)A.Header->Dim >= Dim1) && ((int32)A.Header->Dim >= Dim2), "dimension index out of bounds");

    // NOTE(Abid): Swap the sizes and strides
    uint32 Temp = A.Header->Sizes[Dim1];
    A.Header->Sizes[Dim1] = A.Header->Sizes[Dim2];
    A.Header->Sizes[Dim2] = Temp;
    Temp = A.Header->Strides[Dim1];
    A.Header->Strides[Dim1] = A.Header->Strides[Dim2];
    A.Header->Strides[Dim2] = Temp;

    A.Header->IsContiguous = false;
}

#if 0
internal tensor32
T32Transpose(tensor32 A, int32 Dim1, int32 Dim2, tensor32 ResultTen)
{
    // NOTE(Abid): Support for negative indexing.
    Dim1 = (A.Header->Dim + Dim1) % A.Header->Dim;
    Dim2 = (A.Header->Dim + Dim2) % A.Header->Dim;

    Assert(ResultTen.Header->DType == A.Header->DType, "tensor type mismatch");
    Assert(ResultTen.Header->StorageNumElements == A.Header->StorageNumElements, "tensor storage size mismatch");
    Assert(Dim1 != Dim2, "incorrect dimensions for transpose; select two different dimensions");
    Assert(((int32)A.Header->Dim >= Dim1) && ((int32)A.Header->Dim >= Dim2),
           "dimension index out of bounds");

    uint32 *NewSizes = ResultTen.Header->Sizes;
    uint32 *NewStrides = ResultTen.Header->Strides;
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(uint32));
    memcpy(NewStrides, A.Header->Strides, A.Header->Dim*sizeof(uint32));

    // NOTE(Abid): Swap the sizes and strides
    NewSizes[Dim1] = A.Header->Sizes[Dim2];
    NewSizes[Dim2] = A.Header->Sizes[Dim1];
    NewStrides[Dim1] = A.Header->Strides[Dim2];
    NewStrides[Dim2] = A.Header->Strides[Dim1];

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    // Copy the tensor to the new one based on DType
    if (A.Header->DType == dtype_float32) { _TRANSPOSED_COPY_TENSOR_DTYPE(float32) }
    else if (A.Header->DType == dtype_int32) { _TRANSPOSED_COPY_TENSOR_DTYPE(int32) }
    else Assert(0, "invalid code path");

    // NOTE(Abid): Add information for backpropagation
    ResultTen.Header->DerivedOp.TensorOp = op_UnaryTranpose;
    tensor32 *Operands = ResultTen.Header->DerivedOp.Operands;
    Operands[0] = A;
    int32 *TranposedDims = (int32 *)(Operands+1);
    TranposedDims[0] = Dim1;
    TranposedDims[1] = Dim2;
    ResultTen.Header->DerivedOp.Operands = Operands;
    ResultTen.Header->DerivedOp.OpContext = TranposedDims;

    Free(AccessDims);
    return ResultTen;
}

internal tensor32
T32TransposeAll(tensor32 A, tensor32 ResultTen)
{
    Assert(ResultTen.Header->DType == A.Header->DType, "tensor type mismatch");
    Assert(GetStorageSize(ResultTen.Header->Sizes, ResultTen.Header->Dim) == GetStorageSize(A.Header->Sizes, A.Header->Dim),
           "tensor storage size mismatch");

    uint32 *NewSizes = ResultTen.Header->Sizes;
    uint32 *NewStrides = ResultTen.Header->Strides;
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(uint32));
    memcpy(NewStrides, A.Header->Strides, A.Header->Dim*sizeof(uint32));

    // NOTE(Abid): Swap the sizes and strides
    for(uint32 Idx = 0; Idx < A.Header->Dim; ++Idx)
    {
        NewSizes[Idx] = A.Header->Sizes[A.Header->Dim-Idx-1];
        NewStrides[Idx] = A.Header->Strides[A.Header->Dim-Idx-1];
    }

    uint32 *AccessDims = (uint32 *)Calloc(A.Header->Dim, sizeof(uint32));
    int32 AccessDimIdx = 0;

    if (A.Header->DType == dtype_float32) { _TRANSPOSED_COPY_TENSOR_DTYPE(float32) }
    else if (A.Header->DType == dtype_int32) { _TRANSPOSED_COPY_TENSOR_DTYPE(int32) }
    else Assert(0, "invalid code path");

    // NOTE(Abid): Add information for backpropagation
    ResultTen.Header->DerivedOp.TensorOp = op_UnaryTranposeAll;
    tensor32 *Operands = ResultTen.Header->DerivedOp.Operands;
    Operands[0] = A;
    ResultTen.Header->DerivedOp.Operands = Operands;

    Free(AccessDims);
    return ResultTen;
}
#endif

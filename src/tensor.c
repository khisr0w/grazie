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
     then the grad should also be changed?
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

#define _ALLOC_TENSOR_DTYPE(TYPE, DTYPE) \
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
                       DataSize*sizeof(TYPE); \
    if(IS_GRAD_PRESERVE()) FinalSize += DataSize*sizeof(float32); /* NOTE(Abid): For Grad storage */ \
    \
    /* NOTE(Abid): Memory mapping */ \
    Result.Header = (tensor_header *)Malloc(FinalSize); \
    Assert(Result.Header, "storage memory cannot be allocated"); \
    Result.Header->Sizes = (uint32 *)(Result.Header+1); \
    Result.Header->Strides = (uint32 *)(Result.Header->Sizes + ShapeLength); \
    Result.Header->AccessSizes = (uint32 *)(Result.Header->Strides + ShapeLength); \
    Result.Header->DType = DTYPE; \
    Result.Header->IsContiguous = true; \
    Result.Header->DataStorageSize = DataSize; \
    /* NOTE(Abid): Setting whether to compute the backward pass or not */ \
    Result.Header->ShouldGrad = IS_GRAD_PRESERVE(); \
    Result.Header->GradStorageInit = IS_GRAD_PRESERVE(); \
    Result.Header->DerivedOp.TensorOp = op_None; \
    Result.Header->DerivedOp.OpContext = NULL; /* TODO(Abid): This memory gets created during math ops, which is not nice
                                                              Figure out a max ceiling and always allocate that at init */ \
    \
    Result.Header->DerivedOp.Operands = (tensor32 *)(Result.Header->AccessSizes + ShapeLength); \
    Result.Data = (void *)((tensor32 *)Result.Header->DerivedOp.Operands + 2); /* NOTE(Abid): 2 operands by default */ \
    if(Result.Header->GradStorageInit) Result.Grad = ((TYPE *)Result.Data) + DataSize; \
    else Result.Grad = NULL; \
    memset(Result.Grad, 0, DataSize*sizeof(float32)); \
    \
    /* NOTE(Abid): Setting Default Values */ \
    Result.Header->Offset = 0; \
    Result.Header->Dim = ShapeLength; \
    memcpy(Result.Header->Sizes, Shape, ShapeLength*sizeof(uint32)); \
    \
    /* NOTE(Abid): Calculate the strides given the tensor shape */ \
    for(uint32 i = 0; i < Result.Header->Dim; ++i) \
    { \
        if(Result.Header->Sizes[i] == 1) Result.Header->Strides[i] = 0; \
        else Result.Header->Strides[i] = 1; \
    } \
    if(Result.Header->Dim > 1) \
    { /* NOTE(Abid): If the size in a dim is 1, then the stride will be zero */ \
        for(uint32 i = 0; i < (Result.Header->Dim-1); ++i) \
                for(uint32 j = i+1; j < Result.Header->Dim; ++j) { Result.Header->Strides[i] *= \
                                                                   Result.Header->Sizes[j]; } \
    } \
    \
    if(Intialize) \
    { \
        /* NOTE(Abid): Check if the DataLength makes sense with the shape */ \
        Assert(DataLength == DataSize, "Mismatch of data and tensor shape"); \
        memcpy(Result.Data, Data, DataSize*sizeof(TYPE)); \
    } \
    \
    return Result; \

#define F32Tensor(Shape, Data) _float32AllocTensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data), true)
internal inline tensor32
_float32AllocTensor(uint32 *Shape, uint32 ShapeLength, float32 *Data, size_t DataLength, boolean Intialize)
{ _ALLOC_TENSOR_DTYPE(float32, dtype_float32); }


#define I32Tensor(Shape, Data) _int32AllocTensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data), true)
internal inline tensor32
_int32AllocTensor(uint32 *Shape, uint32 ShapeLength, int32 *Data, size_t DataLength, boolean Intialize)
{ _ALLOC_TENSOR_DTYPE(int32, dtype_int32); }

#define SHAPE(...) __VA_ARGS__
#define ARRAY(...) __VA_ARGS__
#define TensorFromArrayLiteral(NAME, DTYPE, Shape, Values) \
    {0}; \
    do \
    { \
        uint32 Shape_Arr[] = { Shape }; \
        DTYPE Values_Arr[] = { Values }; \
        NAME = _##DTYPE##AllocTensor(Shape_Arr, ArrayLength(Shape_Arr), \
                                     Values_Arr, ArrayLength(Values_Arr), true); \
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

// TODO(Abid): The print currently is leaking memory since its not freeing the tensors that are
//             the result of in-argument computations. Needs to be macro'ed out once memory
//             module is written.

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
        case dtype_float32: { _PRINT_DTYPE("tensor f32", Data, float, "%f"); } break;
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
            for(int32 Idx = 0; Idx < A.Header->DataStorageSize; ++Idx)
                StoragePtr[Idx] = TempVal;
        } break;
        case dtype_int32:
        {
            int32 TempVal = (int32)Value;
            int32 *StoragePtr = (int32 *)A.Data;
            for(int32 Idx = 0; Idx < A.Header->DataStorageSize; ++Idx)
                StoragePtr[Idx] = TempVal;
        } break;
        default: Assert(0, "invalid code path");
    }
}

// =======================================
// NOTE(Abid): Math Operations tensor_i32
// =======================================
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

#if 0
// NOTE(Abid): Elementwise Operations
#define _BINARY_ELEMENTWISE_OP(A, B, Result, OPERATION, OP_TYPE, A_TYPE, B_TYPE, RES_TYPE) \
    Result.Header->ShouldGrad = IS_GRAD_PRESERVE(); \
    tensor32 GTETensor = A; \
    tensor32 LTETensor = B; \
    if(A.Header->Dim < B.Header->Dim) \
    { \
        GTETensor = B; \
        LTETensor = A; \
    } \
    \
    uint32 MaxDim = GTETensor.Header->Dim; \
    size_t *ResultAccessDims = (uint32 *)Calloc(MaxDim*(1+1+1 + 1 + 1), sizeof(uint32)); /* TODO(Abid): Do not allocate here */ \
    size_t *GTETensorAccessDims = ResultAccessDims + MaxDim; \
    size_t *LTETensorAccessDims = GTETensorAccessDims + MaxDim; \
    uint32 *MaxResultSizes = ResultAccessDims + MaxDim; \
    broadcast_rules *BroadcastRules = (broadcast_rules *)(MaxResultSizes + MaxDim);\
    int32 AccessDimIdx = 0; \
    \
    uint32 DimDiff = GTETensor.Header->Dim - LTETensor.Header->Dim; \
    for (uint32 Idx = 1; Idx <= MaxDim: ++Idx) \
    { \
        if(LTETensor.Header->Dim - Idx >= 0) \
        { \
            uint32 GTESize = GTETensor.Header->Sizes[GTETensor.Header->Dim-Idx] \
            uint32 LTESize = LTETensor.Header->Sizes[LTETensor.Header->Dim-Idx] \
            uint32 MaxSize = max(GTESize, LTESize); \
            boolean IsMinOne = MaxSize + 1 == GTESize + LTESize; \
            boolean SameDim = 2*MaxSize == GTESize + LTESize; \
            boolean IsBothOne = IsMinOne && SameDim; \
            Assert(IsMinOne || SameDim, "tensors are not broadcastable"); \
            if(IsMinOne) \
            { \
                if (IsBothOne) BroadcastRules[MaxDim-Idx] = brule_SameSize; \
                else if(LTESize == 1) BroadcastRules[MaxDim-Idx] = brule_LTERepeat; \
                else BroadcastRules[MaxDim-Idx] = brule_GTERepeat; \
            } \
            else BroadcastRules[MaxDim-Idx] = brule_GTERepeat; \
            MaxResultSizes[MaxDim-Idx] = MaxSize; \
        } \
        MaxResultSizes[MaxDim-Idx] = GTETensor.Header->Sizes[MaxDim-Idx]; \
    } \
    Assert(IsArrayEqual(Result.Header->Sizes, AssumedResultSizes, Result.Header->Dim, MaxDim)) \
    \
    /* NOTE(Abid): Binary Op logic here */ \
    \

    Result.Header->DerivedOp.TensorOp = OP_TYPE; \
    tensor32 *Operands = Result.Header->DerivedOp.Operands; \
    Operands[0] = A; \
    Operands[1] = B; \
    \
    Free(AccessDims);
#endif

#if 0
    while(AccessDimIdx >= 0) \
    { \
        boolean IsOuterDim = ResultAccessDims < (int32)(MaxDim-1); \
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
                ((RES_TYPE *)Result.Data)[ResultOuterDimOffset] = (RES_TYPE)(((A_TYPE *)A.Data)[ATenOuterDimOffset] OPERATION \
                                                                                ((B_TYPE *)B.Data)[BTenOuterDimOffset]); \
                ATenOuterDimOffset += A.Header->Strides[AccessDimIdx]; \
                BTenOuterDimOffset += B.Header->Strides[AccessDimIdx]; \
                ResultOuterDimOffset += Result.Header->Strides[AccessDimIdx]; \
            } \
            AccessDims[AccessDimIdx--] = 0; \
            if(AccessDimIdx != -1) ++AccessDims[AccessDimIdx]; \
        } \
    } \

#define _BINARY_ELEMENTWISE_OP_DTYPE(A, B, Result, OPERATION, OP_TYPE) \
    /* Assert(IsShapeEqual(A.Header, B.Header), "shape mismatch for " #OPERATION " operation"); */ \
    Assert (A.Header->DType == dtype_int32 || A.Header->DType == dtype_float32, #OPERATION " can only be carried over int/float tensors"); \
    Assert (B.Header->DType == dtype_int32 || B.Header->DType == dtype_float32, #OPERATION " can only be carried over int/float tensors"); \
    /* Assert(IsShapeEqual(A.Header, Result.Header), "shape mismatch for " #OPERATION " operation"); */ \
    Assert(A.Header->Dim > 0, "dimension of the tensor must greater than zero"); \
    Assert(B.Header->Dim > 0, "dimension of the tensor must greater than zero"); \
    if(A.Header->DType == B.Header->DType) \
    { \
        if(A.Header->DType == dtype_float32) \
        { \
            _BINARY_ELEMENTWISE_OP(A, B, Result, OPERATION, OP_TYPE, float32, float32, float32); \
            Result.Header->DType = dtype_float32; \
        } \
        else if(A.Header->DType == dtype_int32) \
        { \
            if(OP_TYPE == op_BinaryDiv) \
            { \
                _BINARY_ELEMENTWISE_OP(A, B, Result, OPERATION, OP_TYPE, int32, int32, float32); \
                Result.Header->DType = dtype_float32; \
            } \
            else \
            { \
                _BINARY_ELEMENTWISE_OP(A, B, Result, OPERATION, OP_TYPE, int32, int32, int32); \
                Result.Header->DType = dtype_int32; \
            } \
        } \
        else Assert(0, "invalid code path"); \
    } \
    else \
    { \
        if(B.Header->DType == dtype_float32) \
        { \
            _BINARY_ELEMENTWISE_OP(A, B, Result, OPERATION, OP_TYPE, int32, float32, float32); \
            Result.Header->DType = dtype_float32; \
        } \
        else if(A.Header->DType == dtype_float32) \
        { \
            _BINARY_ELEMENTWISE_OP(A, B, Result, OPERATION, OP_TYPE, float32, int32, float32); \
            Result.Header->DType = dtype_float32; \
        } \
        else Assert(0, "invalid code path"); \
    }

internal tensor32
T32Sub(tensor32 A, tensor32 B, tensor32 Result)
{
    _BINARY_ELEMENTWISE_OP_DTYPE(A, B, Result, -, op_BinarySub);
    return Result;
}

internal tensor32
T32Mul(tensor32 A, tensor32 B, tensor32 Result)
{
    _BINARY_ELEMENTWISE_OP_DTYPE(A, B, Result, *, op_BinaryMul);
    return Result;
}

internal tensor32
T32Div(tensor32 A, tensor32 B, tensor32 Result)
{
    _BINARY_ELEMENTWISE_OP_DTYPE(A, B, Result, /, op_BinaryDiv);
    return Result;
}
#endif

internal inline void
_CheckEndofDimAndUpdate(tensor_header *AHeader, tensor_header *BHeader, tensor_header *ResultHeader,
                        size_t *AOffset, size_t *BOffset, size_t *ResultOffset)
{
    uint32 DimIdx = 1;
    // NOTE(Abid): If we've reached the end of the this dim in result tensor
    while(ResultHeader->AccessSizes[ResultHeader->Dim- DimIdx] == ResultHeader->Sizes[ResultHeader->Dim- DimIdx])
    {
        // NOTE(Abid): If there is nothing left to calculate
        if(ResultHeader->Dim- DimIdx == 0) break;

        int32 ACurrentDim = (int32)AHeader->Dim - DimIdx;
        int32 BCurrentDim = (int32)BHeader->Dim - DimIdx;
        int32 ResultCurrentDim = (int32)ResultHeader->Dim - DimIdx;

        if(ACurrentDim <= 0)
        { 
            *AOffset = 0;
            AHeader->AccessSizes[0] = 0;
            // NOTE(Abid): Then B tensor is the one with greater dim
            BHeader->AccessSizes[BCurrentDim] = 0;
            ++BHeader->AccessSizes[BCurrentDim-1];
            *BOffset -= BHeader->Strides[BCurrentDim]*BHeader->Sizes[BCurrentDim];
            *BOffset += BHeader->Strides[BCurrentDim-1];
        }
        else if(BCurrentDim <= 0)
        {
            *BOffset = 0;
            BHeader->AccessSizes[0] = 0;
            // NOTE(Abid): Then A tensor is the one with greater dim
            AHeader->AccessSizes[ACurrentDim] = 0;
            ++AHeader->AccessSizes[ACurrentDim-1];
            *AOffset -= AHeader->Strides[ACurrentDim]*AHeader->Sizes[ACurrentDim];
            *AOffset += AHeader->Strides[ACurrentDim-1];
        }
        else
        {
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

internal tensor32
T32Add(tensor32 A, tensor32 B, tensor32 Result)
{
    // TODO(Abid): Do Assert here that result does match with the highest dim
    int64 ResDataLeft = Result.Header->DataStorageSize;
    uintptr AOffset = 0;
    uintptr BOffset = 0;
    uintptr ResultOffset = 0;
    uint32 DimIdx = 1;

    memset(Result.Header->AccessSizes, 0, Result.Header->Dim*sizeof(uintptr));
    memset(A.Header->AccessSizes, 0, A.Header->Dim*sizeof(uint32));
    memset(B.Header->AccessSizes, 0, B.Header->Dim*sizeof(uint32));
    
    // TODO(Abid): For optimizing this routine, check if we are in a broadcastable dim, then take the
    //             value that we must broadcast, and then run a for loop continuously operating it until
    //             the end of this dim.
    // TODO(Abid): Also, at the start of loop, check if we got 4 values at the tail dimension, and if we
    //             do, then get all 4 and operate on them instead of going through cache and check.
    while(ResDataLeft - 4 >= 0)
    {
        // NOTE(Abid): Procure 4 ops
        float32 AVals[4];
        float32 BVals[4];
        float32 *ResultPtr[4];

        ResultPtr[0] = (float32 *)Result.Data + ResultOffset;
        AVals[0] = *((float32 *)A.Data + AOffset);
        BVals[0] = *((float32 *)B.Data + BOffset);
        int32 ACurrentDim = (int32)A.Header->Dim - DimIdx;
        int32 BCurrentDim = (int32)B.Header->Dim - DimIdx;
        int32 ResultCurrentDim = (int32)Result.Header->Dim - DimIdx;
        AOffset += A.Header->Strides[ACurrentDim]; ++A.Header->AccessSizes[ACurrentDim];
        BOffset += B.Header->Strides[BCurrentDim]; ++B.Header->AccessSizes[BCurrentDim];
        ResultOffset += Result.Header->Strides[ResultCurrentDim]; ++Result.Header->AccessSizes[ResultCurrentDim];

        _CheckEndofDimAndUpdate(A.Header, B.Header, Result.Header, &AOffset, &BOffset, &ResultOffset);

        ResultPtr[1] = (float32 *)Result.Data + ResultOffset;
        AVals[1] = *((float32 *)A.Data + AOffset);
        BVals[1] = *((float32 *)B.Data + BOffset);
        ACurrentDim = (int32)A.Header->Dim - DimIdx;
        BCurrentDim = (int32)B.Header->Dim - DimIdx;
        ResultCurrentDim = (int32)Result.Header->Dim - DimIdx;
        AOffset += A.Header->Strides[ACurrentDim]; ++A.Header->AccessSizes[ACurrentDim];
        BOffset += B.Header->Strides[BCurrentDim]; ++B.Header->AccessSizes[BCurrentDim];
        ResultOffset += Result.Header->Strides[ResultCurrentDim]; ++Result.Header->AccessSizes[ResultCurrentDim];

        _CheckEndofDimAndUpdate(A.Header, B.Header, Result.Header, &AOffset, &BOffset, &ResultOffset);

        ResultPtr[2] = (float32 *)Result.Data + ResultOffset;
        AVals[2] = *((float32 *)A.Data + AOffset);
        BVals[2] = *((float32 *)B.Data + BOffset);
        ACurrentDim = (int32)A.Header->Dim - DimIdx;
        BCurrentDim = (int32)B.Header->Dim - DimIdx;
        ResultCurrentDim = (int32)Result.Header->Dim - DimIdx;
        AOffset += A.Header->Strides[ACurrentDim]; ++A.Header->AccessSizes[ACurrentDim];
        BOffset += B.Header->Strides[BCurrentDim]; ++B.Header->AccessSizes[BCurrentDim];
        ResultOffset += Result.Header->Strides[ResultCurrentDim]; ++Result.Header->AccessSizes[ResultCurrentDim];

        _CheckEndofDimAndUpdate(A.Header, B.Header, Result.Header, &AOffset, &BOffset, &ResultOffset);

        ResultPtr[3] = (float32 *)Result.Data + ResultOffset;
        AVals[3] = *((float32 *)A.Data + AOffset);
        BVals[3] = *((float32 *)B.Data + BOffset);
        ACurrentDim = (int32)A.Header->Dim - DimIdx;
        BCurrentDim = (int32)B.Header->Dim - DimIdx;
        ResultCurrentDim = (int32)Result.Header->Dim - DimIdx;
        AOffset += A.Header->Strides[ACurrentDim]; ++A.Header->AccessSizes[ACurrentDim];
        BOffset += B.Header->Strides[BCurrentDim]; ++B.Header->AccessSizes[BCurrentDim];
        ResultOffset += Result.Header->Strides[ResultCurrentDim]; ++Result.Header->AccessSizes[ResultCurrentDim];

        _CheckEndofDimAndUpdate(A.Header, B.Header, Result.Header, &AOffset, &BOffset, &ResultOffset);

        *ResultPtr[0] = AVals[0] + BVals[0];
        *ResultPtr[1] = AVals[1] + BVals[1];
        *ResultPtr[2] = AVals[2] + BVals[2];
        *ResultPtr[3] = AVals[3] + BVals[3];

        ResDataLeft -= 4;
    }

    while(ResDataLeft > 0)
    {
        _CheckEndofDimAndUpdate(A.Header, B.Header, Result.Header, &AOffset, &BOffset, &ResultOffset);

        float32 *ResultPtr = (float32 *)Result.Data + ResultOffset;
        float32 AVals = *((float32 *)A.Data + AOffset);
        float32 BVals = *((float32 *)B.Data + BOffset);
        int32 ACurrentDim = (int32)A.Header->Dim - DimIdx;
        int32 BCurrentDim = (int32)B.Header->Dim - DimIdx;
        int32 ResultCurrentDim = (int32)Result.Header->Dim - DimIdx;
        AOffset += A.Header->Strides[ACurrentDim]; ++A.Header->AccessSizes[ACurrentDim];
        BOffset += B.Header->Strides[BCurrentDim]; ++B.Header->AccessSizes[BCurrentDim];
        ResultOffset += Result.Header->Strides[ResultCurrentDim]; ++Result.Header->AccessSizes[ResultCurrentDim];

        *ResultPtr = AVals + BVals;

        --ResDataLeft;
    }
    
    return Result;
}


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
internal tensor32
T32Transpose(tensor32 A, int32 Dim1, int32 Dim2, tensor32 ResultTen)
{
    Dim1 = (A.Header->Dim + Dim1) % A.Header->Dim;
    Dim2 = (A.Header->Dim + Dim2) % A.Header->Dim;

    Assert(ResultTen.Header->DType == A.Header->DType, "tensor type mismatch");
    Assert(GetStorageSize(ResultTen.Header->Sizes, ResultTen.Header->Dim) == GetStorageSize(A.Header->Sizes, A.Header->Dim),
           "tensor storage size mismatch");
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

#if 0
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
                        int32 AVal = A.Data[ATenOffset + Idx*A.Header->Strides[A.Header->Dim-1]];
                        int32 BVal = B.Data[BTenOffset + Idx*B.Header->Strides[B.Header->Dim-2]];
                        Result += AVal*BVal;
                    }
                    ResultTen.Data[ResultOuterDimOffset + ResultOffset] = Result;
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

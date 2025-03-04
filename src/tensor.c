/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 10:19:09 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +==================================================| Sayed Abid Hashimi |==========+  */

/* TODO(Abid):
 * - Indexing Schemes
 * - Softmax
 * - View
 * - Convolution (https://github.com/vdumoulin/conv_arithmetic)
 */

#include "tensor.h"
#include "module.h"

/* NOTE(Abid): The minimum allocated space for Stride/Shape is 2*sizeof(u32),
 *             Since, it will ease up the math computations and allow reshape ops */
/* TODO(Abid): It doesn't make sense to have f32/i32 TYPE allocation here. We can have byte-sized
 *              allocation instead, where the tensor is allocated based on how many bytes each
 *              element will get. In this case both f32 and i32 takes 4 bytes. */
/* TODO(Abid): The default type should always be f32, unless another call changes that. */
#define __ALLOC_TENSOR_DTYPE(TYPE, StoreGrad, Arena) \
    t32 *Result = NULL; \
    \
    size_t DataSize = 1; \
    for (u32 i = 0; i < ShapeLength; ++i) { DataSize *= Shape[i]; } \
    assert(DataSize != 0, "wrong shape given, cannot be zero"); \
    \
    u32 AllocShapeLength = ShapeLength; \
    if(ShapeLength == 1) ++AllocShapeLength;\
    size_t FinalSize = sizeof(t32) + \
                       sizeof(tensor_header) + \
                       3*AllocShapeLength*sizeof(u32) + /* For Stride, Shape, SizesAccessPtr */ \
                       2*sizeof(t32 *) + /*  For tensor operands */ \
                       DataSize*sizeof(TYPE) + \
                       (i32)StoreGrad*DataSize*sizeof(f32); /* StoreGrad for backprop */ \
    \
    /* NOTE(Abid): Memory mapping */ \
    Result = (t32 *)gzMemPushSize(Arena, FinalSize); \
    Result->Header = (tensor_header *)(Result+1); \
    assert(Result->Header, "storage memory cannot be allocated"); \
    Result->Header->Sizes = (u32 *)(Result->Header+1); \
    Result->Header->Strides = (u32 *)(Result->Header->Sizes + AllocShapeLength); \
    Result->Header->AccessSizes = (u32 *)(Result->Header->Strides + AllocShapeLength); \
    Result->Data.DType = dtype_##TYPE; \
    Result->Grad.DType = dtype_f32; \
    Result->Header->IsContiguous = true; \
    Result->Header->StorageNumElements = DataSize; \
    /* NOTE(Abid): Setting whether to compute the backward pass or not */ \
    Result->Header->ShouldGrad = IS_GRAD_PRESERVE(); \
    Result->Header->DerivedOp.TensorOp = op_none; \
    Result->Header->DerivedOp.op_context = NULL;  \
    \
    Result->Header->DerivedOp.Operands = (t32 **)(Result->Header->AccessSizes + AllocShapeLength); \
    Result->Data.Ptr = (void *)((t32 **)Result->Header->DerivedOp.Operands + 2); /* 2 operands by default */ \
    if(StoreGrad) { \
        Result->Grad.Ptr = ((TYPE *)Result->Data.Ptr) + DataSize; \
        memset(Result->Grad.Ptr, 0, DataSize*sizeof(TYPE)); \
        \
    } \
    else Result->Grad.Ptr = NULL; \
    \
    /* NOTE(Abid): Setting Default Values */ \
    Result->Header->Offset = 0; \
    Result->Header->Dim = ShapeLength; \
    memcpy(Result->Header->Sizes, Shape, AllocShapeLength*sizeof(u32)); \
    \
    /* NOTE(Abid): Calculate the strides given the tensor shape */      \
    for(u32 Idx = 0; Idx < Result->Header->Dim; ++Idx) {                \
        if(Result->Header->Sizes[Idx] == 1) {                           \
            Result->Header->Strides[Idx] = 0;                           \
            continue;                                                   \
        } else Result->Header->Strides[Idx] = 1;                        \
        for(u32 Jdx = Idx+1; Jdx < Result->Header->Dim; ++Jdx) {        \
            Result->Header->Strides[Idx] *= Result->Header->Sizes[Jdx]; \
        }                                                               \
    }                                                                   \
    \
    if(Data) { \
        /* NOTE(Abid): Check if the DataLength makes sense with the shape */ \
        assert(DataLength == DataSize, "data and tensor shape mismatch"); \
        memcpy(Result->Data.Ptr, Data, DataSize*sizeof(TYPE)); \
    } \
    \
    if(ShouldZero) memset(Result->Data.Ptr, 0, DataSize*sizeof(TYPE)); \
    \
    return Result;

#define gz_tensor_from_array(Shape, Data, TYPE, ShouldGrad, Arena) \
    _gzTensorAlloc##TYPE(Shape, gz_array_length(Shape), Data, gz_array_length(Data), ShouldGrad, false, Arena)

#define _gz_tensor_empty(Shape, ShapeLength, TYPE, ShouldGrad, Arena) \
    _gzTensorAlloc##TYPE(Shape, ShapeLength, 0, 0, false, ShouldGrad, Arena)
#define gz_tensor_empty(Shape, TYPE, ShouldGrad, Arena) _gz_tensor_empty(Shape, gz_array_length(Shape), TYPE, ShouldGrad, Arena)

#define _gz_tensor_zero(Shape, ShapeLength, TYPE, ShouldGrad, Arena) \
    _gzTensorAlloc##TYPE(Shape, ShapeLength, 0, 0, true, ShouldGrad, Arena)
#define gz_tensor_zero(Shape, TYPE, ShouldGrad, Arena) _gz_tensor_zero(Shape, gz_array_length(Shape), TYPE, ShouldGrad, Arena)

internal void
gzCalculateStride(u32 *Strides, u32 *Sizes, u32 Dim) {
    /* NOTE(Abid): Calculate the strides given the tensor shape */
    for(u32 Idx = 0; Idx < Dim; ++Idx) {
        if(Sizes[Idx] == 1) {
            Strides[Idx] = 0;
            continue;
        } else Strides[Idx] = 1;
        for(u32 Jdx = Idx+1; Jdx < Dim; ++Jdx) {
            Strides[Idx] *= Sizes[Jdx];
        }
    }
}

internal inline t32 *
_gzTensorAllocf32(u32 *Shape, u32 ShapeLength, f32 *Data, size_t DataLength, bool ShouldGrad, bool ShouldZero, mem_arena *Arena)
{ __ALLOC_TENSOR_DTYPE(f32, ShouldGrad, Arena); }

internal inline t32 *
_gzTensorAlloci32(u32 *Shape, u32 ShapeLength, i32 *Data, size_t DataLength, bool ShouldGrad, bool ShouldZero, mem_arena *Arena)
{ __ALLOC_TENSOR_DTYPE(i32, ShouldGrad, Arena); }

#define __TO_TENSOR_TYPE(type) dtype_##type; 

internal inline t32 *
_gz_tensor_alloc_huskf32_batched(u64 *unbatched_shape, u64 unbatched_shape_length, u64 batch_size,
                                 mem_arena *arena) {
    t32 *Result = NULL;

    u64 batched_shape_length = unbatched_shape_length+1;
    size_t FinalSize = sizeof(t32) + sizeof(tensor_header) +
                       3*(unbatched_shape_length+1)*sizeof(u32) + /* For Stride, Shape, SizesAccessPtr */
                       2*sizeof(t32 *); /*  For tensor operands */

    /* NOTE(Abid): Memory mapping */
    Result = (t32 *)gzMemPushSize(arena, FinalSize); 
    assert(Result, "storage memory cannot be allocated"); 
    Result->Header = (tensor_header *)(Result+1); 
    Result->Header->Sizes = (u32 *)(Result->Header+1); 
    Result->Header->Strides = (u32 *)(Result->Header->Sizes + batched_shape_length); 
    Result->Header->AccessSizes = (u32 *)(Result->Header->Strides + batched_shape_length); 
    Result->Data.DType = __TO_TENSOR_TYPE(f32);
    Result->Grad.DType = dtype_f32; 
    Result->Header->IsContiguous = true; 
    /* NOTE(Abid): Setting whether to compute the backward pass or not */ 
    Result->Header->ShouldGrad = false; 
    Result->Header->DerivedOp.TensorOp = op_none; 
    Result->Header->DerivedOp.op_context = NULL;  
    
    Result->Header->DerivedOp.Operands = (t32 **)(Result->Header->AccessSizes + batched_shape_length); 
    Result->Header->Offset = 0; 
    Result->Header->Dim = (u32)batched_shape_length;
    /* TODO(abid): Convert `Sizes`, `Strides` and `Offset` to u64 at some point. */
    Result->Header->Sizes[0] = (u32)batch_size;
    memcpy(Result->Header->Sizes + 1, unbatched_shape, unbatched_shape_length*sizeof(u32)); 

    /* NOTE(abid): Calculate number of elements. */
    Result->Header->StorageNumElements = 1;
    for(u32 idx = 0; idx < unbatched_shape_length; ++idx)
        Result->Header->StorageNumElements *= unbatched_shape[idx];
    Result->Header->StorageNumElements *= batch_size;
    assert(Result->Header->StorageNumElements != 0, "wrong shape given, cannot be zero");
    
    /* NOTE(Abid): Calculate the strides given the tensor shape */      
    for(u32 Idx = 0; Idx < Result->Header->Dim; ++Idx) {                
        if(Result->Header->Sizes[Idx] == 1) {                           
            Result->Header->Strides[Idx] = 0;                           
            continue;                                                   
        } else Result->Header->Strides[Idx] = 1;                        

        for(u32 Jdx = Idx+1; Jdx < Result->Header->Dim; ++Jdx) {        
            Result->Header->Strides[Idx] *= Result->Header->Sizes[Jdx]; 
        }                                                               
    }                                                                   
    
    return Result;
}

internal inline t32 *
_gz_tensor_alloc_huskf32(u64 *shape, u64 shape_length, mem_arena *arena) { 
    t32 *Result = NULL;

    size_t DataSize = 1;
    for (u32 i = 0; i < shape_length; ++i) { DataSize *= shape[i]; }
    assert(DataSize != 0, "wrong shape given, cannot be zero");

    u64 AllocShapeLength = shape_length;
    if(shape_length == 1) ++AllocShapeLength;
    size_t FinalSize = sizeof(t32) + sizeof(tensor_header) +
                       3*AllocShapeLength*sizeof(u32) + /* For Stride, Shape, SizesAccessPtr */
                       2*sizeof(t32 *); /*  For tensor operands */

    /* NOTE(Abid): Memory mapping */
    Result = (t32 *)gzMemPushSize(arena, FinalSize); 
    assert(Result, "storage memory cannot be allocated"); 
    Result->Header = (tensor_header *)(Result+1); 
    Result->Header->Sizes = (u32 *)(Result->Header+1); 
    Result->Header->Strides = (u32 *)(Result->Header->Sizes + AllocShapeLength); 
    Result->Header->AccessSizes = (u32 *)(Result->Header->Strides + AllocShapeLength); 
    Result->Data.DType = __TO_TENSOR_TYPE(f32);
    Result->Grad.DType = dtype_f32; 
    Result->Header->IsContiguous = true; 
    Result->Header->StorageNumElements = DataSize; 
    /* NOTE(Abid): Setting whether to compute the backward pass or not */ 
    Result->Header->ShouldGrad = false; 
    Result->Header->DerivedOp.TensorOp = op_none; 
    Result->Header->DerivedOp.op_context = NULL;
    
    Result->Header->DerivedOp.Operands = (t32 **)(Result->Header->AccessSizes + AllocShapeLength); 
    Result->Header->Offset = 0; 
    /* TODO(abid): Convert `Sizes`, `Strides` and `Offset` to u64 at some point. */
    Result->Header->Dim = (u32)shape_length; 
    memcpy(Result->Header->Sizes, shape, AllocShapeLength*sizeof(u32)); 
    
    /* NOTE(Abid): Calculate the strides given the tensor shape */      
    for(u32 Idx = 0; Idx < Result->Header->Dim; ++Idx) {                
        if(Result->Header->Sizes[Idx] == 1) {                           
            Result->Header->Strides[Idx] = 0;                           
            continue;                                                   
        } else Result->Header->Strides[Idx] = 1;                        
        for(u32 Jdx = Idx+1; Jdx < Result->Header->Dim; ++Jdx) {        
            Result->Header->Strides[Idx] *= Result->Header->Sizes[Jdx]; 
        }                                                               
    }                                                                   
    
    return Result;
}

#define gzTensorNormal(Shape, Mean, Std, ShouldGrad, Arena) _gzTensorNormal(Shape, gz_array_length(Shape), Mean, Std, ShouldGrad, Arena)
internal inline t32 *
_gzTensorNormal(u32 *Shape, u32 ShapeLength, f64 Mean, f64 Std, bool ShouldGrad, mem_arena *Arena) {
    t32 *Tensor = _gzTensorAllocf32(Shape, ShapeLength, 0, 0, ShouldGrad, false, Arena);

    size_t NumData = Tensor->Header->StorageNumElements;
    for(size_t Idx = 0; Idx <= NumData; ++Idx) {
        *((f32 *)Tensor->Data.Ptr + Idx) = (f32)gzRandNormal(Mean, Std);
    }

    return Tensor;
}


#define ARR(...) __VA_ARGS__
#define TensorFromArrayLiteral(NAME, DTYPE, Shape, Values, ShouldGrad, Arena) \
    NULL; \
    do {  \
        u32 Shape_Arr[] = { Shape }; \
        DTYPE Values_Arr[] = { Values }; \
        NAME = _##DTYPE##AllocTensor(Shape_Arr, gzArrayLength(Shape_Arr), \
                                     Values_Arr, gzArrayLength(Values_Arr), \
                                     ShouldGrad, Arena); \
    } while(0);

internal inline size_t
gzGetStorageSize(u32 *Sizes, u32 Dim) {
    size_t Result = 1;

    for (u32 Idx = 0; Idx < Dim; ++Idx) {
        assert(Sizes[Idx], "tensor shape cannot be zero")
        Result *= Sizes[Idx];
    }

    return Result;
}

internal inline bool
gzIsArrayEqual(u32 *Array1, u32 *Array2, u32 Array1Length, u32 Array2Length) {
    if(Array1Length != Array2Length) return false;

    for (u32 Idx = 0; Idx < Array1Length; ++Idx)
        if (Array1[Idx] != Array2[Idx]) return false;

    return true;
}

internal inline bool
gzIsShapeEqual(tensor_header *A, tensor_header *B) { return gzIsArrayEqual(A->Sizes, B->Sizes, A->Dim, B->Dim); }

/* TODO(Abid): Implement for unit testing */
#if 0
internal inline bool
T32IsClose(t32 *A, t32 *B) {
    bool Result = true;

    assert(IsShapeEqual(A->Header, B->Header), "tensors of comparison must be the same shape");
    usize NumElements = A->Header->StorageNumElements;
    for(usize Idx = 0; Idx < NumElements; ++Idx) {
    }

    return Result;
}
#endif


#define __PRINT_DTYPE(TEN_NAME, PRINT_FORMAT, TYPE) \
    printf(TEN_NAME); \
    printf(" -> shape ("); \
    for (u32 Idx = 0; Idx < (A->Header->Dim-1); ++Idx) { printf("%d,", A->Header->Sizes[Idx]); } \
    printf("%d) :=\n",A->Header->Sizes[A->Header->Dim-1]); \
    size_t NumData = A->Header->StorageNumElements; \
    for (u32 Idx = 0; Idx < A->Header->Dim; ++Idx) printf("["); ++PrintCount; \
    u32 NumSpaceNewLine = A->Header->Dim-1; \
    \
    size_t Offset = 0; \
    for(size_t OpNum = 1; OpNum <= NumData; ++OpNum) { \
        if(PrintCount >= MaxPrintWidth) { \
            printf("\n"); PrintCount = 0; \
            for(u32 Idx = 0; Idx < NumSpaceNewLine; ++Idx) printf(" "); \
        } \
        printf(PRINT_FORMAT, *((TYPE *)A->Data.Ptr + Offset)); ++PrintCount; \
        \
        i32 DimMaxNumSoFar = 1; \
        u32 NumClosedBrackets = 0; \
        for(i32 DimIdx = 1; DimIdx <= (i32)A->Header->Dim; ++DimIdx) { \
            DimMaxNumSoFar *= A->Header->Sizes[A->Header->Dim-DimIdx]; \
            if(OpNum % DimMaxNumSoFar == 0) { \
                Offset -= A->Header->Strides[A->Header->Dim-DimIdx]*(A->Header->Sizes[A->Header->Dim-DimIdx]-1); \
                printf("]"); ++PrintCount; \
                ++NumClosedBrackets; \
                continue; \
            } \
            Offset += A->Header->Strides[A->Header->Dim-DimIdx]; \
            printf(", "); ++PrintCount; \
            if(PrintCount >= MaxPrintWidth) { \
                printf("\n"); PrintCount = 0; \
                for(u32 Idx = 0; Idx < NumSpaceNewLine; ++Idx) printf(" "); \
            } \
            for(u32 Idx = 0; Idx < NumClosedBrackets; ++Idx) printf("["); ++PrintCount; \
            break; \
        } \
    } \
    printf("\n\n")
internal void
gz_print(t32 *A) {
    i32 MaxPrintWidth = 20;
    i32 PrintCount = 0;

    /* NOTE(Abid): Print the header and data */
    tensor_dtype DType = A->Data.DType;
    switch(DType)
    {
        case dtype_i32: { __PRINT_DTYPE("tensor i32", "%d", i32); } break;
        case dtype_f32: { __PRINT_DTYPE("tensor f32", "%.4f", f32); } break;
        default: assert(0, "invalid code path");
    }
}
#undef __PRINT_DTYPE

/* TODO(Abid): Not so sure about this */
internal inline void
gzSwapDataGrad(t32 *Tensor) {
    storage Grad = Tensor->Grad;
    Tensor->Grad = Tensor->Data;
    Tensor->Data = Grad;
}

/* =======================================
 * NOTE(Abid): Math Operations
 * ======================================= */
/* TODO(Abid): Refactor all elementwise routines so that any vector (dim==1) gets converted to a matrix beforehand. */

internal inline void
gzReduceSumAll_(t32 *A, t32 *Result) {
    assert((Result->Header->Dim == 1) && (Result->Header->Sizes[0] == 1), "result tensor must be of shape (1)");
    assert(Result->Data.Ptr, "tensor storage not found");

    f32 ResSum = 0;
    switch (A->Data.DType) {
        case dtype_f32: {
            for(i32 Idx = 0; Idx < A->Header->StorageNumElements; ++Idx)
                ResSum += *(f32 *)A->Data.Ptr + Idx;
        } break;
        case dtype_i32: {
            for(i32 Idx = 0; Idx < A->Header->StorageNumElements; ++Idx)
                ResSum += *(f32 *)A->Data.Ptr + Idx;
        } break;
        default: assert(0, "invalid code path");
    }
    if(Result->Data.DType == dtype_i32) *(i32 *)Result->Data.Ptr = (i32)ResSum;
    else *(f32 *)Result->Data.Ptr = ResSum;

    Result->Header->DerivedOp.TensorOp = op_unary_reduce_sum_all;
    Result->Header->DerivedOp.Operands[0] = A;
}
internal inline t32 *
gzReduceSumAll(t32 *A, mem_arena *Arena) {
    u32 ResultShape[] = {1};
    t32 *Result = _gzTensorAllocf32(ResultShape, 1, 0, 0, A->Header->ShouldGrad, false, Arena);
    gzReduceSumAll_(A, Result);

    return Result;
}

/* NOTE(Abid): Main routines for elementwise binary operations. */
internal inline void
_CheckEndofDimAndUpdate(tensor_header *AHeader, tensor_header *BHeader, tensor_header *ResultHeader,
                        size_t *AOffset, size_t *BOffset, size_t *ResultOffset) {
    u32 DimIdx = 1;
    /* NOTE(Abid): If we've reached the end of the this dim in result tensor */
    while(ResultHeader->AccessSizes[ResultHeader->Dim- DimIdx] == ResultHeader->Sizes[ResultHeader->Dim- DimIdx]) {
        /* NOTE(Abid): If there is nothing left to calculate */
        if(ResultHeader->Dim - DimIdx == 0) break;

        i32 ACurrentDim = (i32)AHeader->Dim - DimIdx;
        i32 BCurrentDim = (i32)BHeader->Dim - DimIdx;
        i32 ResultCurrentDim = (i32)ResultHeader->Dim - DimIdx;

        if(ACurrentDim <= 0) { 
            *AOffset = 0;
            AHeader->AccessSizes[0] = 0;
            /* NOTE(Abid): Then B tensor is the one with greater dim */
            BHeader->AccessSizes[BCurrentDim] = 0;
            ++BHeader->AccessSizes[BCurrentDim-1];
            *BOffset -= BHeader->Strides[BCurrentDim]*BHeader->Sizes[BCurrentDim];
            *BOffset += BHeader->Strides[BCurrentDim-1];
        }
        else if(BCurrentDim <= 0) {
            *BOffset = 0;
            BHeader->AccessSizes[0] = 0;
            /* NOTE(Abid): Then A tensor is the one with greater dim */
            AHeader->AccessSizes[ACurrentDim] = 0;
            ++AHeader->AccessSizes[ACurrentDim-1];
            *AOffset -= AHeader->Strides[ACurrentDim]*AHeader->Sizes[ACurrentDim];
            *AOffset += AHeader->Strides[ACurrentDim-1];
        }
        else {
            /* NOTE(Abid): Both tensors have dim in this case */
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
    while(ResDataLeft > 0) { \
        *((R_DTYPE *)Result->Data.Ptr + ResultOffset) = (R_DTYPE)(*((A_DTYPE *)A->Data.Ptr + AOffset) OP \
                                                       *((B_DTYPE *)B->Data.Ptr + BOffset)); \
        i32 ACurrentDim = (i32)A->Header->Dim - DimIdx; \
        i32 BCurrentDim = (i32)B->Header->Dim - DimIdx; \
        i32 ResultCurrentDim = (i32)Result->Header->Dim - DimIdx; \
        AOffset += A->Header->Strides[ACurrentDim]; ++A->Header->AccessSizes[ACurrentDim]; \
        BOffset += B->Header->Strides[BCurrentDim]; ++B->Header->AccessSizes[BCurrentDim]; \
        ResultOffset += Result->Header->Strides[ResultCurrentDim]; ++Result->Header->AccessSizes[ResultCurrentDim]; \
        \
        _CheckEndofDimAndUpdate(A->Header, B->Header, Result->Header, &AOffset, &BOffset, &ResultOffset); \
        --ResDataLeft; \
    }

#define __BIN_ELEMENTWISE_OP(A, B, Result, OP) \
    /* NOTE(Abid): assert here that result does match the highest dim and sizes (broadcast size as well) */ \
    u32 GreaterDim = 0; \
    if (A->Header->Dim > B->Header->Dim) GreaterDim = A->Header->Dim; \
    else GreaterDim = B->Header->Dim; \
    assert(Result->Header->Dim == GreaterDim, "result-operand(s) dimension mismatch"); \
    for(u32 Idx = 1; Idx <= GreaterDim; Idx++) { \
        u32 ASize = 0; \
        u32 BSize = 0; \
        if((i32)(A->Header->Dim - Idx) >= 0) ASize = A->Header->Sizes[A->Header->Dim - Idx]; \
        if((i32)(B->Header->Dim - Idx) >= 0) BSize = B->Header->Sizes[B->Header->Dim - Idx]; \
        /* NOTE(Abid): Check for shape alignment for the operands */ \
        assert(((ASize > BSize) && ((BSize == 1) || (BSize == 0))) || \
               ((BSize > ASize) && ((ASize == 1) || (ASize == 0))) || \
               (BSize == ASize), "operand(s) shape mismatch"); \
        u32 GreaterSize = ASize > BSize ? ASize : BSize; \
        assert(Result->Header->Sizes[Result->Header->Dim - Idx] == GreaterSize, "result-operand(s) shape mismatch"); \
    } \
    \
    /* NOTE(Abid): Initialize variables */ \
    i64 ResDataLeft = Result->Header->StorageNumElements; \
    uintptr AOffset = 0; \
    uintptr BOffset = 0; \
    uintptr ResultOffset = 0; \
    u32 DimIdx = 1; \
    \
    memset(Result->Header->AccessSizes, 0, Result->Header->Dim*sizeof(u32)); \
    memset(A->Header->AccessSizes, 0, A->Header->Dim*sizeof(u32)); \
    memset(B->Header->AccessSizes, 0, B->Header->Dim*sizeof(u32)); \
    Result->Header->ShouldGrad = IS_GRAD_PRESERVE(); \
    \
    bin_op_dtypes OpDTypes = bin_op_dtypes_all_float; /* Assuming all f32 types initially. */ \
    if(A->Data.DType == B->Data.DType) { if(A->Data.DType == dtype_i32) OpDTypes = 2; } \
    else if(A->Data.DType == dtype_i32) { OpDTypes = 6; } /* A is i32, B is f32 */ \
    else OpDTypes = 4; /* A is f32, B is i32 */ \
    \
    if(Result->Data.DType == dtype_i32) OpDTypes += 1; /* Determining the result data type */ \
    \
    switch(OpDTypes) { \
        case bin_op_dtypes_all_float:       { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, f32, f32, f32, OP); } break; \
        case bin_op_dtypes_float_float_int: { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, f32, f32, i32, OP); } break; \
        case bin_op_dtypes_int_int_float:   { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, i32, i32, f32, OP); } break; \
        case bin_op_dtypes_all_int:         { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, i32, i32, i32, OP); } break; \
        case bin_op_dtypes_float_int_float: { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, f32, i32, f32, OP); } break; \
        case bin_op_dtypes_float_int_int:   { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, f32, i32, i32, OP); } break; \
        case bin_op_dtypes_int_float_float: { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, i32, f32, f32, OP); } break; \
        case bin_op_dtypes_int_float_int:   { __BIN_ELEMENTWISE_OP_DTYPE(A, B, Result, i32, f32, i32, OP); } break; \
        default:                            assert(0, "Invalid Code Path");                                         \
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
internal void gzAdd(t32 *A, t32 *B, t32 *Result) {
    __BIN_ELEMENTWISE_OP(A, B, Result, +);

    Result->Header->DerivedOp.TensorOp = op_binary_add;
    Result->Header->DerivedOp.Operands[0] = A;
    Result->Header->DerivedOp.Operands[1] = B;
}
internal void gzSub(t32 *A, t32 *B, t32 *Result) {
    __BIN_ELEMENTWISE_OP(A, B, Result, -);

    Result->Header->DerivedOp.TensorOp = op_binary_sub;
    Result->Header->DerivedOp.Operands[0] = A;
    Result->Header->DerivedOp.Operands[1] = B;
}
internal void gzMul(t32 *A, t32 *B, t32 *Result) {
    __BIN_ELEMENTWISE_OP(A, B, Result, *);

    Result->Header->DerivedOp.TensorOp = op_binary_mul;
    Result->Header->DerivedOp.Operands[0] = A;
    Result->Header->DerivedOp.Operands[1] = B;
}
internal void gzDiv(t32 *A, t32 *B, t32 *Result) {
    __BIN_ELEMENTWISE_OP(A, B, Result, /);

    Result->Header->DerivedOp.TensorOp = op_binary_div;
    Result->Header->DerivedOp.Operands[0] = A;
    Result->Header->DerivedOp.Operands[1] = B;
}
#undef __BIN_ELEMENTWISE_OP_DTYPE
#undef __BIN_ELEMENTWISE_OP

internal inline u32
gzGetIndex(u32 Dim, i32 Index) {
    i32 Result = (i32)Dim + Index;
    if(Result < 0) Result = 0;
    assert(Result < (i32)(2*Dim), "index out of bounds");
    return Result % Dim;
}

/* NOTE(Abid): 2. Main routines for MatMul operation */

/* TODO(Abid): Improve this routine by reshaping all operand tensors to be the same dim (expanding all with shape 1)
 *             and stride 0. For now, let's just do a simple hack with a few conditionals. */

#define __GZ_BIN_MATMUL_DTYPE(A, B, Result, A_DTYPE, B_DTYPE, R_DTYPE, OP) \
    while(ResDataLeft) { \
        /* NOTE(Abid): We are progressing row-wise on the result tensor */ \
        f32 DotResult = 0; \
        for(u32 Idx = 0; Idx < A->Header->Sizes[ALastIdx]; ++Idx) { \
            DotResult += (*((A_DTYPE *)A->Data.Ptr + AOffset) * *((B_DTYPE *)B->Data.Ptr + BOffset)); \
            \
            AOffset += A->Header->Strides[ALastIdx]; \
            BOffset += B->Header->Strides[BSecondLastIdx]; \
        } \
        *((R_DTYPE *)Result->Data.Ptr + ResultOffset) OP (R_DTYPE)DotResult; \
        ++Result->Header->AccessSizes[Result->Header->Dim-1]; \
        \
        AOffset -= A->Header->Strides[ALastIdx]*A->Header->Sizes[ALastIdx]; \
        BOffset -= B->Header->Strides[BSecondLastIdx]*B->Header->Sizes[BSecondLastIdx]; \
        if(--ResDataLeft == 0) break; \
        \
        /* NOTE(Abid): If Result dim = 1, then it won't go beyond this point. */ \
        /* NOTE(Abid): In case we have reached the end of the result tensor row. */ \
        if(Result->Header->AccessSizes[Result->Header->Dim-1] == Result->Header->Sizes[Result->Header->Dim-1]) { \
            ResultOffset -= Result->Header->Strides[Result->Header->Dim-1]*(Result->Header->Sizes[Result->Header->Dim-1]-1); \
            Result->Header->AccessSizes[Result->Header->Dim-1] = 0; \
            \
            if(ResLastBroadDim > 2) { \
                ++Result->Header->AccessSizes[Result->Header->Dim-2]; \
                if(Result->Header->AccessSizes[Result->Header->Dim-2] == Result->Header->Sizes[Result->Header->Dim-2]) { \
                    /* NOTE(Abid): We've reached the end, begin broadcast semantics for matrix... */ \
                    IsBroadcastDim = true; \
                    \
                    /* NOTE(Abid): Zero out matrix multiplication dimensions, and take offsets to the start. */ \
                    AOffset -= A->Header->Strides[ASecondLastIdx]*(A->Header->Sizes[ASecondLastIdx]-1); \
                    A->Header->AccessSizes[ALastIdx] = 0; A->Header->AccessSizes[ASecondLastIdx] = 0; \
                    \
                    BOffset -= B->Header->Strides[BLastIdx]*(B->Header->Sizes[BLastIdx]-1); \
                    B->Header->AccessSizes[BLastIdx] = 0; B->Header->AccessSizes[BSecondLastIdx] = 0; \
                    \
                    Result->Header->AccessSizes[Result->Header->Dim - (ResLastBroadDim-1)] = 0; \
                    ResultOffset -= Result->Header->Strides[Result->Header->Dim-2]*(Result->Header->Sizes[Result->Header->Dim-2]-1); \
                } else { \
                    AOffset += A->Header->Strides[ASecondLastIdx]; /* Going to the next column element (start of next row). */ \
                    BOffset -= B->Header->Strides[BLastIdx]*(B->Header->Sizes[BLastIdx]-1); \
                    ResultOffset += Result->Header->Strides[Result->Header->Dim-2]; \
                } \
            } \
            else { \
                /* NOTE(Abid): Result has only 1 matmul dimension, the rest is for broadcast. */ \
                IsBroadcastDim = true; \
                \
                if(A->Header->Dim > B->Header->Dim) { \
                    /* NOTE(Abid): BOffset needs to be zero'd out */ \
                    AOffset -= A->Header->Strides[ASecondLastIdx]*(A->Header->Sizes[ASecondLastIdx]-2); \
                    A->Header->AccessSizes[ALastIdx] = 0; A->Header->AccessSizes[ASecondLastIdx] = 0; \
                    BOffset = 0; \
                } else { \
                    /* NOTE(Abid): AOffset needs to be zero'd out */ \
                    BOffset -= B->Header->Strides[BLastIdx]*(B->Header->Sizes[BLastIdx]-2); \
                    B->Header->AccessSizes[BLastIdx] = 0; B->Header->AccessSizes[BSecondLastIdx] = 0; \
                    AOffset = 0; \
                } \
                Result->Header->AccessSizes[Result->Header->Dim - (ResLastBroadDim-1)] = 0; \
            } \
        } else { \
            BOffset += B->Header->Strides[B->Header->Dim-1]; /* Going to the next row element (start of next column). */ \
            ResultOffset += Result->Header->Strides[Result->Header->Dim-1]; \
        } \
        \
        if(IsBroadcastDim) { \
            for(u32 CurrentBroadIter = 0; CurrentBroadIter < ResLastBroadDim; ++CurrentBroadIter) { \
                i32 CurrenntDimIdx = ResLastBroadDim + CurrentBroadIter; \
                i32 CurResBroadIdx = (i32)Result->Header->Dim - CurrenntDimIdx; \
                i32 CurABroadIdx = (i32)A->Header->Dim - CurrenntDimIdx; \
                i32 CurBBroadIdx = (i32)B->Header->Dim - CurrenntDimIdx; \
                \
                if(Result->Header->AccessSizes[CurResBroadIdx]+1 == \
                   Result->Header->Sizes[CurResBroadIdx]) { \
                    /* NOTE(Abid): We are at the end of this broadcast dimension */ \
                    Result->Header->AccessSizes[CurResBroadIdx] = 0; \
                    ResultOffset -= Result->Header->Strides[CurResBroadIdx]*Result->Header->Sizes[CurResBroadIdx]; \
                    if(CurABroadIdx >= 0) { \
                        A->Header->AccessSizes[CurABroadIdx] = 0; \
                        AOffset -= A->Header->Strides[CurABroadIdx]*A->Header->Sizes[CurABroadIdx]; \
                    } else { \
                        memset(A->Header->AccessSizes, 0, A->Header->Dim*sizeof(u32)); \
                        AOffset = 0; \
                    } \
                    if(CurBBroadIdx >= 0) { \
                        B->Header->AccessSizes[CurBBroadIdx] = 0; \
                        BOffset -= B->Header->Strides[CurBBroadIdx]*B->Header->Sizes[CurBBroadIdx]; \
                    } else { \
                        memset(A->Header->AccessSizes, 0, A->Header->Dim*sizeof(u32)); \
                        BOffset = 0; \
                    } \
                } else { \
                    ++Result->Header->AccessSizes[CurResBroadIdx]; \
                    ResultOffset += Result->Header->Strides[CurResBroadIdx]; \
                    if(CurABroadIdx >= 0) { \
                        ++A->Header->AccessSizes[CurABroadIdx]; \
                        AOffset += A->Header->Strides[CurABroadIdx]; \
                    } \
                    if(CurBBroadIdx >= 0) { \
                        ++B->Header->AccessSizes[CurBBroadIdx]; \
                        BOffset += B->Header->Strides[CurBBroadIdx]; \
                    } \
                    break; \
                } \
            } \
            IsBroadcastDim = false; \
        } \
    }

/* TODO(Abid): This really needs to be refactored */
internal void 
gzMatMul(t32 *A, t32 *B, t32 *Result) {
    /* NOTE(Abid): assert greater dim is the same as the result tensors' dim */
    u32 GreaterDim = 0;
    u32 LesserDim = 0;
    u32 ResLastBroadDim = 0;

    if (A->Header->Dim > B->Header->Dim) { GreaterDim = A->Header->Dim; LesserDim = B->Header->Dim; }
    else { GreaterDim = B->Header->Dim; LesserDim = A->Header->Dim; }
    if(LesserDim > 1) ResLastBroadDim = 3;
    else ResLastBroadDim = 2;
    assert(Result->Header->Dim == GreaterDim, "result-operand(s) dimension mismatch");

    u32 ALastIdx = gzGetIndex(A->Header->Dim, -1);
    u32 ASecondLastIdx = gzGetIndex(A->Header->Dim, -2);

    u32 BLastIdx = gzGetIndex(B->Header->Dim, -1);
    u32 BSecondLastIdx = gzGetIndex(B->Header->Dim, -2);
    /* NOTE(Abid): Assert here the tensor operand(s) can be matrix-multiplied */
    assert(A->Header->Sizes[ALastIdx] == B->Header->Sizes[BSecondLastIdx],
           "operand(s) shape mismatch for MatMul operation");

    /* NOTE(Abid): Assert the shapes of the result tensor match with the MatMul operation's required shape.
     *             1. In case, the lesserDim is 1, then we expect the last dimension of result to be 1 as well
     *             2. Otherwise, the dimensions must match with the result of the MatMul operation. */
    assert(((LesserDim > 1) && (
                                    (A->Header->Sizes[gzGetIndex(A->Header->Dim, -2)] ==
                                     Result->Header->Sizes[gzGetIndex(Result->Header->Dim, -2)]) &&
                                    (B->Header->Sizes[gzGetIndex(B->Header->Dim, -1)] ==
                                     Result->Header->Sizes[gzGetIndex(Result->Header->Dim, -1)])
                               )
           ) || ((LesserDim == 1) && (B->Header->Dim == 1 ? Result->Header->Sizes[gzGetIndex(Result->Header->Dim, - 1)] == 1 :
                                                           Result->Header->Sizes[gzGetIndex(Result->Header->Dim, - 2)] == 1)),
           "result-operand(s) shape mismatch for MatMul operation");

    /* NOTE(Abid): Assert broadcast shape match */
    for(u32 Idx = 3; Idx <= GreaterDim; ++Idx) {
        u32 ASize = 0; u32 BSize = 0;
        if((i32)(A->Header->Dim - Idx) >= 0) ASize = A->Header->Sizes[A->Header->Dim - Idx];
        if((i32)(B->Header->Dim - Idx) >= 0) BSize = B->Header->Sizes[B->Header->Dim - Idx];
        /* NOTE(Abid): Check for shape alignment for the operands */
        bool IsAGreater = ASize > BSize;
        assert((IsAGreater && ((BSize == 1) || (BSize == 0))) ||
               ((BSize > ASize) && ((ASize == 1) || (ASize == 0))) ||
               (BSize == ASize), "operand(s) shape mismatch");
        u32 GreaterSize = IsAGreater ? ASize : BSize;
        assert(Result->Header->Sizes[Result->Header->Dim - Idx] == GreaterSize, "result-operand(s) shape mismatch");
    }

    /* NOTE(Abid): Initialize variables */
    i64 ResDataLeft = Result->Header->StorageNumElements;
    uintptr AOffset = 0;
    uintptr BOffset = 0;
    uintptr ResultOffset = 0;

    memset(Result->Header->AccessSizes, 0, Result->Header->Dim*sizeof(u32));
    memset(A->Header->AccessSizes, 0, A->Header->Dim*sizeof(u32));
    memset(B->Header->AccessSizes, 0, B->Header->Dim*sizeof(u32));

    /* NOTE(Abid): Are we allowed to backprop through this operation? */
    Result->Header->ShouldGrad = IS_GRAD_PRESERVE();

    i32 IsBroadcastDim = false; /* Last if we count from the right */
    
    bin_op_dtypes OpDTypes = bin_op_dtypes_all_float; /* Assuming all f32 types initially. */
    if(A->Data.DType == B->Data.DType) { if(A->Data.DType == dtype_i32) OpDTypes = 2; }
    else if(A->Data.DType == dtype_i32) { OpDTypes = 6; } /* A is i32, B is f32 */
    else OpDTypes = 4; /* A is f32, B is i32 */

    if(Result->Data.DType == dtype_i32) OpDTypes += 1; /* Determining the result data type */

    switch(OpDTypes) {
        case bin_op_dtypes_all_float:       { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, f32, f32, =); } break;
        case bin_op_dtypes_float_float_int: { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, f32, i32, =); } break;
        case bin_op_dtypes_int_int_float:   { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, i32, f32, =); } break;
        case bin_op_dtypes_all_int:         { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, i32, i32, =); } break;
        case bin_op_dtypes_float_int_float: { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, i32, f32, =); } break;
        case bin_op_dtypes_float_int_int:   { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, i32, i32, =); } break;
        case bin_op_dtypes_int_float_float: { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, f32, f32, =); } break;
        case bin_op_dtypes_int_float_int:   { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, f32, i32, =); } break;
        default:                            assert(0, "Invalid Code Path");
    }

    Result->Header->DerivedOp.TensorOp = op_binary_matmul;
    Result->Header->DerivedOp.Operands[0] = A;
    Result->Header->DerivedOp.Operands[1] = B;
}

internal void
__gzMatMulAccumulate(t32 *A, t32 *B, t32 *Result) {
    /* NOTE(Abid): assert greater dim is the same as the result tensors' dim */
    u32 GreaterDim = 0;
    u32 LesserDim = 0;
    u32 ResLastBroadDim = 0;

    if(A->Header->Dim > B->Header->Dim) { GreaterDim = A->Header->Dim; LesserDim = B->Header->Dim; }
    else { GreaterDim = B->Header->Dim; LesserDim = A->Header->Dim; }
    if(LesserDim > 1) ResLastBroadDim = 3;
    else ResLastBroadDim = 2;
    assert(Result->Header->Dim == GreaterDim, "result-operand(s) dimension mismatch");

    u32 ALastIdx = gzGetIndex(A->Header->Dim, -1);
    u32 ASecondLastIdx = gzGetIndex(A->Header->Dim, -2);

    u32 BLastIdx = gzGetIndex(B->Header->Dim, -1);
    u32 BSecondLastIdx = gzGetIndex(B->Header->Dim, -2);
    /* NOTE(Abid): Assert here the tensor operand(s) can be matrix-multiplied */
    assert(A->Header->Sizes[ALastIdx] == B->Header->Sizes[BSecondLastIdx],
           "operand(s) shape mismatch for MatMul operation");

    /* NOTE(Abid): Assert the shapes of the result tensor match with the MatMul operation's required shape.
     *             1. In case, the lesserDim is 1, then we expect the last dimension of result to be 1 as well
     *             2. Otherwise, the dimensions must match with the result of the MatMul operation. */
    assert(((LesserDim > 1) && (
                                    (A->Header->Sizes[gzGetIndex(A->Header->Dim, -2)] ==
                                     Result->Header->Sizes[gzGetIndex(Result->Header->Dim, -2)]) &&
                                    (B->Header->Sizes[gzGetIndex(B->Header->Dim, -1)] ==
                                     Result->Header->Sizes[gzGetIndex(Result->Header->Dim, -1)])
                               )
           ) || ((LesserDim == 1) && (B->Header->Dim == 1 ? Result->Header->Sizes[gzGetIndex(Result->Header->Dim, - 1)] == 1 :
                                                           Result->Header->Sizes[gzGetIndex(Result->Header->Dim, - 2)] == 1)),
           "result-operand(s) shape mismatch for MatMul operation");

    /* NOTE(Abid): Assert broadcast shape match */
    for(u32 Idx = 3; Idx <= GreaterDim; ++Idx) {
        u32 ASize = 0; u32 BSize = 0;
        if((i32)(A->Header->Dim - Idx) >= 0) ASize = A->Header->Sizes[A->Header->Dim - Idx];
        if((i32)(B->Header->Dim - Idx) >= 0) BSize = B->Header->Sizes[B->Header->Dim - Idx];
        /* NOTE(Abid): Check for shape alignment for the operands */
        bool IsAGreater = ASize > BSize;
        assert((IsAGreater && ((BSize == 1) || (BSize == 0))) ||
               ((BSize > ASize) && ((ASize == 1) || (ASize == 0))) ||
               (BSize == ASize), "operand(s) shape mismatch");
        u32 GreaterSize = IsAGreater ? ASize : BSize;
        assert(Result->Header->Sizes[Result->Header->Dim - Idx] == GreaterSize, "result-operand(s) shape mismatch");
    }

    /* NOTE(Abid): Initialize variables */
    i64 ResDataLeft = Result->Header->StorageNumElements;
    uintptr AOffset = 0;
    uintptr BOffset = 0;
    uintptr ResultOffset = 0;

    memset(Result->Header->AccessSizes, 0, Result->Header->Dim*sizeof(u32));
    memset(A->Header->AccessSizes, 0, A->Header->Dim*sizeof(u32));
    memset(B->Header->AccessSizes, 0, B->Header->Dim*sizeof(u32));

    /* NOTE(Abid): Are we allowed to backprop through this operation? */
    Result->Header->ShouldGrad = IS_GRAD_PRESERVE();

    i32 IsBroadcastDim = false; /* Last if we count from the right */
    
    bin_op_dtypes OpDTypes = bin_op_dtypes_all_float; /* Assuming all f32 types initially. */
    if(A->Data.DType == B->Data.DType) { if(A->Data.DType == dtype_i32) OpDTypes = 2; }
    else if(A->Data.DType == dtype_i32) { OpDTypes = 6; } /* A is i32, B is f32 */
    else OpDTypes = 4; /* A is f32, B is i32 */

    if(Result->Data.DType == dtype_i32) OpDTypes += 1; /* Determining the result data type */

    switch(OpDTypes) {
        case bin_op_dtypes_all_float:       { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, f32, f32, +=); } break;
        case bin_op_dtypes_float_float_int: { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, f32, i32, +=); } break;
        case bin_op_dtypes_int_int_float:   { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, i32, f32, +=); } break;
        case bin_op_dtypes_all_int:         { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, i32, i32, +=); } break;
        case bin_op_dtypes_float_int_float: { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, i32, f32, +=); } break;
        case bin_op_dtypes_float_int_int:   { __GZ_BIN_MATMUL_DTYPE(A, B, Result, f32, i32, i32, +=); } break;
        case bin_op_dtypes_int_float_float: { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, f32, f32, +=); } break;
        case bin_op_dtypes_int_float_int:   { __GZ_BIN_MATMUL_DTYPE(A, B, Result, i32, f32, i32, +=); } break;
        default:                            assert(0, "Invalid Code Path");
    }
}
#undef __BIN_MATMUL_DTYPE
/* NOTE(Abid): Main routines for Unary operations */
/* TODO(Abid): Implement T32ElementOp here */

internal void
__gzSigmoidOnStorage(tensor_header *AHead, f32 *SrcStorage, tensor_header *ResHead, f32 *ResStorage) {

    assert((SrcStorage != NULL) && (ResStorage != NULL), "null storage found");
    assert(gzIsShapeEqual(AHead, ResHead), "operand-result shape mismatch");

    size_t ANumData = AHead->StorageNumElements;
    size_t ResultOffset = 0;
    size_t AOffset = 0;
    for(size_t OpNum = 1; OpNum <= ANumData; ++OpNum) {
        f32 SrcData = *(SrcStorage + AOffset);
        *(ResStorage + ResultOffset) = 1 / (1 + expf(-SrcData));

        i32 DimMaxNumSoFar = 1;
        for(i32 DimIdx = 1; DimIdx <= (i32)AHead->Dim; ++DimIdx) {
            DimMaxNumSoFar *= AHead->Sizes[AHead->Dim-DimIdx];
            if(OpNum % DimMaxNumSoFar == 0) {
                AOffset -= AHead->Strides[AHead->Dim-DimIdx]*(AHead->Sizes[AHead->Dim-DimIdx]-1);
                ResultOffset -= ResHead->Strides[ResHead->Dim-DimIdx]*(ResHead->Sizes[ResHead->Dim-DimIdx]-1);
                continue;
            }
            AOffset += AHead->Strides[AHead->Dim-DimIdx];
            ResultOffset += ResHead->Strides[ResHead->Dim-DimIdx];
            break;
        }
    }
}

inline internal void
gz_sigmoid_(t32 *A, t32 *Result) {
    assert((A->Data.DType == Result->Data.DType) & (A->Data.DType == dtype_f32),
           "sigmoid require tensor(s) to be of type f32");
    __gzSigmoidOnStorage(A->Header, A->Data.Ptr, Result->Header, Result->Data.Ptr);
    Result->Header->DerivedOp.TensorOp = op_unary_sigmoid;
    Result->Header->DerivedOp.Operands[0] = A;
}

inline internal t32 *
gz_sigmoid(t32 *tensor, mem_arena *arena) {
    t32 *result = _gzTensorAllocf32(tensor->Header->Sizes, tensor->Header->Dim, 0, 0, tensor->Header->ShouldGrad, false, arena);
    gz_sigmoid_(tensor, result);

    return result;
}

internal void
_gz_relu(t32 *A, t32 *Result) {
    tensor_header *AHead = A->Header;
    f32 *AStorage = A->Data.Ptr;
    tensor_header *ResHead = Result->Header;
    f32 *ResStorage = Result->Data.Ptr;

    assert((AStorage != NULL) && (ResStorage != NULL), "null storage found");
    assert(gzIsShapeEqual(AHead, ResHead), "operand-result shape mismatch");

    size_t ANumData = AHead->StorageNumElements;
    size_t ResultOffset = 0;
    size_t AOffset = 0;
    for(size_t OpNum = 1; OpNum <= ANumData; ++OpNum) {
        f32 Value = *(AStorage + AOffset);
        *(ResStorage + ResultOffset) = (Value < 0) ? 0.f : Value;

        i32 DimMaxNumSoFar = 1;
        for(i32 DimIdx = 1; DimIdx <= (i32)AHead->Dim; ++DimIdx) {
            DimMaxNumSoFar *= AHead->Sizes[AHead->Dim-DimIdx];
            if(OpNum % DimMaxNumSoFar == 0) {
                AOffset -= AHead->Strides[AHead->Dim-DimIdx]*(AHead->Sizes[AHead->Dim-DimIdx]-1);
                ResultOffset -= ResHead->Strides[ResHead->Dim-DimIdx]*(ResHead->Sizes[ResHead->Dim-DimIdx]-1);
                continue;
            }
            AOffset += AHead->Strides[AHead->Dim-DimIdx];
            ResultOffset += ResHead->Strides[ResHead->Dim-DimIdx];
            break;
        }
    }

    Result->Header->DerivedOp.TensorOp = op_unary_relu;
    Result->Header->DerivedOp.Operands[0] = A;
}

internal inline t32 *
gz_relu(t32 *A, mem_arena *Arena) {
    t32 *Result = _gzTensorAllocf32(A->Header->Sizes, A->Header->Dim, 0, 0, A->Header->ShouldGrad, false, Arena);
    _gz_relu(A, Result);

    return Result;
}

internal void
__LossLogOnStorage(tensor_header *AHead, f32 *SrcStorage, tensor_header *ResHead, f32 *ResStorage) {
    assert((SrcStorage != NULL) && (ResStorage != NULL), "null storage found");
    assert(gzIsShapeEqual(AHead, ResHead), "operand-result shape mismatch");

    size_t ANumData = AHead->StorageNumElements;
    size_t ResultOffset = 0;
    size_t AOffset = 0;
    for(size_t OpNum = 1; OpNum <= ANumData; ++OpNum) {
        ResStorage[ResultOffset] = gz_logf(SrcStorage[AOffset]);

        i32 DimMaxNumSoFar = 1;
        for(i32 DimIdx = 1; DimIdx <= (i32)AHead->Dim; ++DimIdx) {
            DimMaxNumSoFar *= AHead->Sizes[AHead->Dim-DimIdx];
            if(OpNum % DimMaxNumSoFar == 0) {
                AOffset -= AHead->Strides[AHead->Dim-DimIdx]*(AHead->Sizes[AHead->Dim-DimIdx]-1);
                ResultOffset -= ResHead->Strides[ResHead->Dim-DimIdx]*(ResHead->Sizes[ResHead->Dim-DimIdx]-1);
                continue;
            }
            AOffset += AHead->Strides[AHead->Dim-DimIdx];
            ResultOffset += ResHead->Strides[ResHead->Dim-DimIdx];
            break;
        }
    }
}

/* =======================================
 * NOTE(Abid): Reshaping Operations
 * ======================================= */

/* NOTE(Abid): Check if the proposed view fits the data of the original tensor */
internal bool
gzValidateViewOnTensor(t32 *A, u32 *NewShape, u32 NewShapeLength) {
    usize ViewExpectedNumElements = 1;
    for(u32 Idx = 0; Idx < NewShapeLength; ++Idx) {
        ViewExpectedNumElements *= NewShape[Idx];
    }

    return ViewExpectedNumElements == A->Header->StorageNumElements;
}

#define gzNewView(A, NewShape, Arena) _gzNewView(A, NewShape, gzArrayLength(NewShape), Arena)
internal t32 *
_gzNewView(t32 *A, u32 *NewShape, u32 NewShapeLength, mem_arena *Arena) {
    assert(A->Header->IsContiguous, "view a of non-contiguous tensor not allowed");
    assert(NewShapeLength > 0, "invalid view, dim cannot be %d", NewShapeLength);
    assert(gzValidateViewOnTensor(A, NewShape, NewShapeLength), "invalid view, shape-storage mismatch")

    t32 *Result = gz_mem_push_struct(t32, Arena);
    Result->Header = gz_mem_push_struct(tensor_header, Arena);

    /* NOTE(Abid): Copy storage pointer and dtype */
    Result->Data.DType = A->Data.DType;
    Result->Data.Ptr = A->Data.Ptr;
    Result->Grad.DType = A->Grad.DType;
    Result->Grad.Ptr = A->Grad.Ptr;

    /* NOTE(Abid): Copy tensor_header data */
    Result->Header->Dim = NewShapeLength;
    Result->Header->Offset = A->Header->Offset;
    Result->Header->ShouldGrad = A->Header->ShouldGrad;
    Result->Header->IsContiguous = A->Header->IsContiguous;
    Result->Header->StorageNumElements = A->Header->StorageNumElements;

    /* NOTE(Abid): Stride and Sizes TODO: Must remove AccessSizes */
    Result->Header->Sizes = gzMemPushArray(Arena, u32, 3*NewShapeLength);
    Result->Header->Strides = Result->Header->Sizes + NewShapeLength;
    Result->Header->AccessSizes = Result->Header->Strides + NewShapeLength;
    memcpy(Result->Header->Sizes, NewShape, NewShapeLength*sizeof(u32));
    gzCalculateStride(Result->Header->Strides, Result->Header->Sizes, Result->Header->Dim);

    /* NOTE(Abid): Derived ops operand(s) */
    Result->Header->DerivedOp.TensorOp = op_unary_view;
    /* TODO(Abid): Could we just allocate ONE operand instead of two */
    Result->Header->DerivedOp.Operands = gzMemPushArray(Arena, t32 *, 2);
    Result->Header->DerivedOp.Operands[0] = A;

    return Result;
}

/* NOTE(Abid): Trim the trailing size of the tensor if the last size is unity (1) */
/* TODO(Abid): Calling `_gzNewView` just to trim trailing size seems wasteful. MUST change. */
internal t32 *
gz_trim_trailing_unit_size(t32 *A, mem_arena *Arena) {
    bool IsValid = (A->Header->Dim > 1) && (A->Header->Sizes[A->Header->Dim-1] == 1);
    t32 *result = A;

    // assert(IsValid, "cannot trim, trailing size is %d, with dim %d",
    //        A->Header->Sizes[A->Header->Dim-1], A->Header->Dim);

    if(IsValid) result = _gzNewView(A, A->Header->Sizes, A->Header->Dim-1, Arena);

    return result;
}

internal inline bool
gzIsContiguous(t32 A) {
    bool IsContiguous = true;
    for(u32 Idx = 0; Idx < A.Header->Dim; ++Idx) {
        if(A.Header->Sizes[Idx] == 1) continue;
        for(u32 CompIdx = 1; CompIdx < A.Header->Dim; ++CompIdx) {
            if(A.Header->Strides[Idx] <= A.Header->Strides[CompIdx]) {
                IsContiguous = false;
                break;
            }
        }
    }

    return IsContiguous;
}

#define gzReshapeInPlace(A, NEW_SHAPE) \
    do { \
        i32 Shape[] = { NEW_SHAPE }; \
        __gzReshapeInPlace(A, Shape, gzArrayLength(Shape)); \
    } while(0)
internal void
__gzReshapeInPlace(t32 A, i32 *NewSizes, u32 NewSizesLength) {

    assert(NewSizes && (NewSizesLength > 0), "must provide appropriate size values for reshaping operation");
    /* TODO(Abid): Support -1 indexing in reshaping for dimension that is leftover. */
    u32 TotalSizeProd = 1;
    for(u32 Idx = 0; Idx < NewSizesLength; ++Idx) {
        TotalSizeProd *= NewSizes[Idx];
    }
    assert(TotalSizeProd == A.Header->StorageNumElements, "invalid reshape sizes");

    /* TODO(Abid): Reshape here, make sure the allocation of sizes, strides, and dim are done separately. */
}

/* TODO(Abid): Must take care of non-contiguous tensors, especially when flattening */
#define _GZ_TRANSPOSED_COPY_TENSOR_DTYPE(DTYPE) \
    while(AccessDimIdx >= 0) \
    { \
        /* NOTE(Abid): Check if we are in outer dim and then update */ \
        bool IsOuterDim = AccessDimIdx < (i32)(A.Header->Dim-1); \
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
            u32 ATenOuterDimOffset = 0; \
            u32 ResultOuterDimOffset = 0; \
            for(u32 Idx = 0; Idx < A.Header->Dim-1; ++Idx) \
            { \
                ATenOuterDimOffset += AccessDims[Idx] * NewStrides[Idx]; \
                ResultOuterDimOffset += AccessDims[Idx] * ResultTen.Header->Strides[Idx]; \
            } \
            for(u32 Idx = 0; Idx < NewSizes[AccessDimIdx]; ++Idx) \
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

internal inline void
__gzTransposeInPlaceNoGrad(t32 *A, i32 Dim1, i32 Dim2)
{
    /* TODO(Abid): Properly check if the transpose could make the tensor contiguous again. */
    assert(A->Header->IsContiguous, "cannot transpose non-contiguous tensor");

    /* NOTE(Abid): Support for negative indexing. */
    Dim1 = (A->Header->Dim + Dim1) % A->Header->Dim;
    Dim2 = (A->Header->Dim + Dim2) % A->Header->Dim;

    assert(((i32)A->Header->Dim >= Dim1) && ((i32)A->Header->Dim >= Dim2), "dimension index out of bounds");

    /* NOTE(Abid): Swap the sizes and strides */
    u32 Temp = A->Header->Sizes[Dim1];
    A->Header->Sizes[Dim1] = A->Header->Sizes[Dim2];
    A->Header->Sizes[Dim2] = Temp;
    Temp = A->Header->Strides[Dim1];
    A->Header->Strides[Dim1] = A->Header->Strides[Dim2];
    A->Header->Strides[Dim2] = Temp;
}

inline internal void
gzTransposeInPlace(t32 *A, i32 Dim1, i32 Dim2) {
    __gzTransposeInPlaceNoGrad(A, Dim1, Dim2);
    A->Header->IsContiguous = false;
}

/* =================================
 * NOTE(Abid): TensorList operations
 * ================================= */

inline internal void
gz_tensor_list_add(t32 *Tensor, tensor_list *TensorList) {
    assert(TensorList->used < TensorList->size, "Index out of range. Not enough space");
    TensorList->array[TensorList->used++] = Tensor;
}

inline internal tensor_list
gz_tensor_list_allocate(usize Size, mem_arena *Arena) {
    tensor_list TensorList = {0};
    TensorList.size = Size;
    TensorList.array = gzMemPushArray(Arena, t32 *, TensorList.size);

    return TensorList;
}

internal tensor_list
gzTensorListFromModule(module *Module, mem_arena *Arena) {
    tensor_list TensorList = gz_tensor_list_allocate(Module->weights.used, Arena);
    for(u32 Idx = 0; Idx < Module->weights.used; ++Idx) {
        gz_tensor_list_add(Module->weights.array[Idx], &TensorList);
    }

    return TensorList;
}

internal tensor_list
gz_tensor_list_from_module_list(module **modules, u64 modules_length, mem_arena *arena) {
    u64 total_tensors = 0;
    for(u64 idx = 0; idx < modules_length; ++idx) {
        total_tensors += modules[idx]->weights.used;
    }

    tensor_list list = gz_tensor_list_allocate(total_tensors, arena);
    for(u32 idx = 0; idx < modules_length; ++idx) {
        module* module = modules[idx];
        for(u32 jdx = 0; jdx < module->weights.used; ++jdx) {
            gz_tensor_list_add(module->weights.array[jdx], &list);
        }
    }

    return list;
}

#if 0
internal t32
T32Transpose(t32 A, i32 Dim1, i32 Dim2, t32 ResultTen)
{
    /* NOTE(Abid): Support for negative indexing. */
    Dim1 = (A.Header->Dim + Dim1) % A.Header->Dim;
    Dim2 = (A.Header->Dim + Dim2) % A.Header->Dim;

    assert(ResultTen.Header->DType == A.Data.DType, "tensor type mismatch");
    assert(ResultTen.Header->StorageNumElements == A.Header->StorageNumElements, "tensor storage size mismatch");
    assert(Dim1 != Dim2, "incorrect dimensions for transpose; select two different dimensions");
    assert(((i32)A.Header->Dim >= Dim1) && ((i32)A.Header->Dim >= Dim2),
           "dimension index out of bounds");

    u32 *NewSizes = ResultTen.Header->Sizes;
    u32 *NewStrides = ResultTen.Header->Strides;
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(u32));
    memcpy(NewStrides, A.Header->Strides, A.Header->Dim*sizeof(u32));

    /* NOTE(Abid): Swap the sizes and strides */
    NewSizes[Dim1] = A.Header->Sizes[Dim2];
    NewSizes[Dim2] = A.Header->Sizes[Dim1];
    NewStrides[Dim1] = A.Header->Strides[Dim2];
    NewStrides[Dim2] = A.Header->Strides[Dim1];

    u32 *AccessDims = (u32 *)Calloc(A.Header->Dim, sizeof(u32));
    i32 AccessDimIdx = 0;

    /* Copy the tensor to the new one based on DType */
    if (A.Data.DType == dtype_f32) { _TRANSPOSED_COPY_TENSOR_DTYPE(f32) }
    else if (A.Data.DType == dtype_i32) { _TRANSPOSED_COPY_TENSOR_DTYPE(i32) }
    else assert(0, "invalid code path");

    /* NOTE(Abid): Add information for backpropagation */
    ResultTen.Header->DerivedOp.TensorOp = op_unary_tranpose;
    t32 *Operands = ResultTen.Header->DerivedOp.Operands;
    Operands[0] = A;
    i32 *TranposedDims = (i32 *)(Operands+1);
    TranposedDims[0] = Dim1;
    TranposedDims[1] = Dim2;
    ResultTen.Header->DerivedOp.Operands = Operands;
    ResultTen.Header->DerivedOp.op_context = TranposedDims;

    Free(AccessDims);
    return ResultTen;
}

internal t32
T32TransposeAll(t32 A, t32 ResultTen)
{
    assert(ResultTen.Header->DType == A.Data.DType, "tensor type mismatch");
    assert(GetStorageSize(ResultTen.Header->Sizes, ResultTen.Header->Dim) == GetStorageSize(A.Header->Sizes, A.Header->Dim),
           "tensor storage size mismatch");

    u32 *NewSizes = ResultTen.Header->Sizes;
    u32 *NewStrides = ResultTen.Header->Strides;
    memcpy(NewSizes, A.Header->Sizes, A.Header->Dim*sizeof(u32));
    memcpy(NewStrides, A.Header->Strides, A.Header->Dim*sizeof(u32));

    / NOTE(Abid): Swap the sizes and strides */
    for(u32 Idx = 0; Idx < A.Header->Dim; ++Idx)
    {
        NewSizes[Idx] = A.Header->Sizes[A.Header->Dim-Idx-1];
        NewStrides[Idx] = A.Header->Strides[A.Header->Dim-Idx-1];
    }

    u32 *AccessDims = (u32 *)Calloc(A.Header->Dim, sizeof(u32));
    i32 AccessDimIdx = 0;

    if (A.Data.DType == dtype_f32) { _TRANSPOSED_COPY_TENSOR_DTYPE(f32) }
    else if (A.Data.DType == dtype_i32) { _TRANSPOSED_COPY_TENSOR_DTYPE(i32) }
    else assert(0, "invalid code path");

    /* NOTE(Abid): Add information for backpropagation */
    ResultTen.Header->DerivedOp.TensorOp = op_unary_tranpose_all;
    t32 *Operands = ResultTen.Header->DerivedOp.Operands;
    Operands[0] = A;
    ResultTen.Header->DerivedOp.Operands = Operands;

    Free(AccessDims);
    return ResultTen;
}
#endif

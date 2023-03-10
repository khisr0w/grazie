/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 10:19:09 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "tensor.h"

#define F32Tensor(Shape, Data) F32Tensor_(Shape, ArrayLength(Shape), Data, ArrayLength(Data))
tensor_f32
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

#define I32Tensor(Shape, Data) I32Tensor_(Shape, ArrayLength(Shape), Data, ArrayLength(Data))
tensor_i32
I32Tensor_(int32 *Shape, int32 ShapeLength, int32 *Data = NULL, size_t DataLength = 0, boolean Intialize = true)
{
    tensor_i32 Result = {};

    size_t StorageSize = 1;
    for (int32 i = 0; i < ShapeLength; ++i) { StorageSize *= Shape[i]; }
    Assert(StorageSize != 0, "Wrong shape given, cannot be zero");

    size_t FinalSize = StorageSize*sizeof(int32);

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

    Result.Storage = (int32 *)Malloc(FinalSize);
    Assert(Result.Storage, "Storage memory cannot be allocated");

    // NOTE(Abid): Check if the DataLength makes sense with the shape
    if(Intialize)
    {
        Assert(DataLength == StorageSize, "Mismatch of data and tensor shape");
        memcpy(Result.Storage, Data, StorageSize*sizeof(int32));
    }
    
    return Result;
}

inline size_t
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

inline size_t
GetValueMemOffset(tensor_header *Header, int32 *AccessShape, boolean IsSparseTensor = false)
{
    // TODO(Abid): Change this, Assert here assumes our AccessShape starts from 1 which is not true
#if 0 
    if(!IsSparseTensor) Assert(GetStorageSize(Header->Sizes, Header->Dim) ==
                               GetStorageSize(AccessShape, Header->Dim), "shape out of bound of storage");
#else
    if(!IsSparseTensor) IsSparseTensor = false;
#endif

    size_t Result = 0;
    for (int32 Idx = 0; Idx < Header->Dim; ++Idx) Result += AccessShape[Idx] * Header->Strides[Idx];

    return Result;
}

inline boolean
IsShapeEqual(tensor_header *A, tensor_header *B)
{
    if(A->Dim != B->Dim) return false;

    for (int32 Idx = 0; Idx < A->Dim; ++Idx)
    {
        if (A->Sizes[Idx] != B->Sizes[Idx]) return false;
    }

    return true;
}

inline void
_PrintTensorHeader(char *Name, tensor_header *TenHeader)
{
    printf(Name); printf(" -> shape (");

    for (int32 i = 0; i < (TenHeader->Dim-1); ++i) { printf("%d,", TenHeader->Sizes[i]); }
    printf("%d) =\n\n",TenHeader->Sizes[TenHeader->Dim-1]);
}

inline boolean
_PrintIsOuterAndUpdateDims(int32 *AccessDimIdx, int32 *AccessDims, tensor_header *TenHeader, int32 *PrintCount)
{
    boolean IsOuterDim = *AccessDimIdx != (TenHeader->Dim-1);
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
            ++AccessDims[(*AccessDimIdx)];
        } else ++(*AccessDimIdx);
    }

    return IsOuterDim;
}

void
PrintTensor(tensor_f32 *Tensor, int32 MaxPrintWidth = 20)
{
    int32 PrintCount = 0;

    tensor_header *TenHeader = &Tensor->Header;
    _PrintTensorHeader("tensor f32", TenHeader);

    int32 AccessDims[MAX_SHAPE_LENGTH] = {0};
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        if(PrintCount >= MaxPrintWidth) { printf("\n"); PrintCount = 0; }
        if(_PrintIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader, &PrintCount)) continue;

        // NOTE(Abid): Main printing of the last dimension
        printf("["); ++PrintCount;
        ++PrintCount;
        for (int32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;
            size_t Pos = GetValueMemOffset(TenHeader, AccessDims);
            float32 Value = Tensor->Storage[Pos];
            printf("%.3f", Value); ++PrintCount;
            if(AccessDims[AccessDimIdx] < (TenHeader->Sizes[AccessDimIdx]-1)) { printf(", "); ++PrintCount; }
        }
        printf("]"); ++PrintCount;
        if(AccessDims[AccessDimIdx-1] < (TenHeader->Sizes[AccessDimIdx-1]-1)) { printf(", "); ++PrintCount; }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }
    printf("\n");
}

void
PrintTensor(tensor_i32 *Tensor, int32 MaxPrintWidth = 20)
{
    int32 PrintCount = 0;

    tensor_header *TenHeader = &Tensor->Header;
    _PrintTensorHeader("tensor i32", TenHeader);

    int32 AccessDims[MAX_SHAPE_LENGTH] = {0};
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        if(PrintCount >= MaxPrintWidth) { printf("\n"); PrintCount = 0; }
        if(_PrintIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader, &PrintCount)) continue;

        // NOTE(Abid): Main printing of the last dimension
        printf("["); ++PrintCount;
        ++PrintCount;
        for (int32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;
            size_t Pos = GetValueMemOffset(TenHeader, AccessDims);
            int32 Value = Tensor->Storage[Pos];
            printf("%d", Value); ++PrintCount;
            if(AccessDims[AccessDimIdx] < (TenHeader->Sizes[AccessDimIdx]-1)) { printf(", "); ++PrintCount; }
        }
        printf("]"); ++PrintCount;
        if(AccessDims[AccessDimIdx-1] < (TenHeader->Sizes[AccessDimIdx-1]-1)) { printf(", "); ++PrintCount; }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }

    printf("\n");
}


// =======================================
// NOTE(Abid): Math Operations tensor_i32
// =======================================
inline boolean
_OpsIsOuterAndUpdateDims(int32 *AccessDimIdx, int32 *AccessDims, tensor_header *TenHeader)
{
    boolean IsOuterDim = *AccessDimIdx != (TenHeader->Dim-1);
    if (IsOuterDim)
    {
        // NOTE(Abid): In case we have reached the end of this dimension
        if (AccessDims[*AccessDimIdx] == TenHeader->Sizes[*AccessDimIdx])
        {
            AccessDims[(*AccessDimIdx)--] = 0;
            ++AccessDims[*AccessDimIdx];
        }
        else ++(*AccessDimIdx);
    }
    return IsOuterDim;
}

tensor_i32
operator+(tensor_i32 A, tensor_i32 B)
{
    Assert(IsShapeEqual(&A.Header, &B.Header), "shape mismatch with + operation");
    tensor_i32 Result = I32Tensor_(A.Header.Sizes, B.Header.Dim, 0, 0, false);

    tensor_header *TenHeader = &A.Header;
    int32 AccessDims[MAX_SHAPE_LENGTH] = {0};
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        if (_OpsIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader)) continue;

        for (int32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;
            size_t Pos = GetValueMemOffset(TenHeader, AccessDims);
            int32 AVal = A.Storage[Pos];
            int32 BVal = B.Storage[Pos];
            Result.Storage[Pos] = AVal+BVal;
        }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }

    return Result;
}

tensor_i32
operator*(tensor_i32 A, tensor_i32 B)
{
    Assert(IsShapeEqual(&A.Header, &B.Header), "shape mismatch with + operation");
    tensor_i32 Result = I32Tensor_(A.Header.Sizes, B.Header.Dim, 0, 0, false);

    tensor_header *TenHeader = &A.Header;
    int32 AccessDims[MAX_SHAPE_LENGTH] = {0};
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        // NOTE(Abid): Checks if we are in outer dim and then updates
        if (_OpsIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader)) continue;

        for (int32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;
            size_t Pos = GetValueMemOffset(TenHeader, AccessDims);
            int32 AVal = A.Storage[Pos];
            int32 BVal = B.Storage[Pos];
            Result.Storage[Pos] = AVal*BVal;
        }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }

    return Result;
}

tensor_i32
operator~(tensor_i32 A)
{
    tensor_i32 Result = I32Tensor_(A.Header.Sizes, A.Header.Dim, 0, 0, false);

    tensor_header *TenHeader = &A.Header;
    int32 AccessDims[MAX_SHAPE_LENGTH] = {0};
    int32 AccessDimIdx = 0;

    while(AccessDimIdx >= 0)
    {
        if (_OpsIsOuterAndUpdateDims(&AccessDimIdx, AccessDims, TenHeader)) continue;

        for (int32 Idx = 0; Idx < TenHeader->Sizes[AccessDimIdx]; ++Idx)
        {
            AccessDims[AccessDimIdx] = Idx;
            size_t Pos = GetValueMemOffset(TenHeader, AccessDims);
            int32 AVal = A.Storage[Pos];
            Result.Storage[Pos] = -1*AVal;
        }

        // NOTE(Abid): Go to the previous dimension while setting this one to zero
        AccessDims[AccessDimIdx--] = 0;
        ++AccessDims[AccessDimIdx];
    }

    return Result;
}

/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  12/20/2023 5:00:51 PM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +==============================================| Sayed Abid Hashimi |==============+  */

internal void
_gz_loss_binary_cross_entropy(t32 *A, t32 *B, t32 *Result, reduce_method *ReduceMethod) {
    /* NOTE(Abid): We expect the input to be probabilities. The tensors are as follows:
     *             A : Prediction
     *             B : Ground
     */

    /* TODO(Abid): Perhaps it would be best to check if A and B are probabilites? Not now though. */
    assert((A->Data.DType == B->Data.DType) && (B->Data.DType == Result->Data.DType) &&
           (A->Data.DType == dtype_f32), "unexpected dtype, f32 expected");
    assert(gzIsShapeEqual(A->Header, B->Header), "operand(s) shape mismatch");
    assert((*ReduceMethod == reduce_none) ? gzIsShapeEqual(Result->Header, B->Header) :
                                           (Result->Header->Dim == 1) && (Result->Header->Sizes[0] == 1),
           "operand-result shape mismatch");
    u32 IsNotNone = (*ReduceMethod != reduce_none);
    size_t AOffset = 0;
    size_t BOffset = 0;
    size_t ResultOffset = 0;
    size_t ExpectedNumOps = A->Header->StorageNumElements;

    /* NOTE(Abid): In case we are reduce_mean or reduce_sum. */
    *((f32 *)Result->Data.Ptr + ResultOffset) = 0;

    for(size_t OpNum = 1; OpNum <= ExpectedNumOps; ++OpNum) {
        f32 X = ((f32 *)A->Data.Ptr)[AOffset];
        f32 Y = ((f32 *)B->Data.Ptr)[BOffset];

        /* TODO(Abid): Maybe we want to do a rescaling weight as well? Maybe not. */
        f32 Loss = Y*gz_clamp(gz_logf(X), -100, INFINITY) + (1-Y)*gz_clamp(gz_logf(1-X), -100, INFINITY);

        /* NOTE(Abid): If we do not have reduce_none, then we will discard any value inside result,
         *             otherwise we add to it. */
        *((f32 *)Result->Data.Ptr + ResultOffset) *= IsNotNone;
        *((f32 *)Result->Data.Ptr + ResultOffset) += -Loss;

        i32 DimMaxNumSoFar = 1;
        /* NOTE(Abid): If we have reached the end of the current dim in A/B */
        for(i32 DimIdx = 1; DimIdx <= (i32)A->Header->Dim; ++DimIdx) {
            DimMaxNumSoFar *= A->Header->Sizes[A->Header->Dim-DimIdx];
            if(OpNum % DimMaxNumSoFar == 0) {
                AOffset -= A->Header->Strides[A->Header->Dim-DimIdx] * (A->Header->Sizes[A->Header->Dim-DimIdx]-1);
                BOffset -= B->Header->Strides[B->Header->Dim-DimIdx] * (B->Header->Sizes[B->Header->Dim-DimIdx]-1);
                if((i32)Result->Header->Dim - DimIdx < 0) ResultOffset = 0;
                else ResultOffset -= Result->Header->Strides[Result->Header->Dim-DimIdx]*
                                     (Result->Header->Sizes[Result->Header->Dim-DimIdx]-1);
                continue;
            }
            AOffset += A->Header->Strides[A->Header->Dim-DimIdx];
            BOffset += B->Header->Strides[B->Header->Dim-DimIdx];
            if((i32)Result->Header->Dim - DimIdx >= 0) ResultOffset += Result->Header->Strides[Result->Header->Dim-DimIdx];
            break;
        }
    }
    if(*ReduceMethod == reduce_mean) {
        *((f32 *)Result->Data.Ptr) = *((f32 *)Result->Data.Ptr) / ExpectedNumOps;
    }

    Result->Header->DerivedOp.TensorOp = op_binary_loss_cross_entropy;
    Result->Header->DerivedOp.Operands[0] = A;
    Result->Header->DerivedOp.Operands[1] = B;
    Result->Header->DerivedOp.op_context = ReduceMethod;
}

/* TODO(abid): Maybe it is better to have two options, one for `reduce_none` and one for others. */
inline internal t32 *
gz_loss_binary_cross_entropy(t32 *a, t32 *b, reduce_method method, mem_arena *arena) {
    t32 *result;
    if(method == reduce_none) {
        result = _gzTensorAllocf32(a->Header->Sizes, a->Header->Dim, 0, 0, a->Header->ShouldGrad, false, arena);
    } else {
        u32 shape1[] = {1};
        result = _gzTensorAllocf32(shape1, 1, 0, 0, a->Header->ShouldGrad, false, arena);
    }

    reduce_method *method_context = gz_mem_push_struct(reduce_method, arena);
    *method_context = method;
    _gz_loss_binary_cross_entropy(a, b, result, method_context);

    return result;
}

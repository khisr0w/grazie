/*
    TODO(Abid):

      EDIT: Fixed memory scheme doesn't seem to make sense here, its not an end product
            but a tool for development and so fixed memory will create headaches, not
            to mention memory fragmentation as stuff are allocated and deallocated.
      EDIT: An optimization issue could arise if we are allocating new memory inside
            an operation. e.g. we are currently allocating memory for `AccessDims`,
            which I would like to get rid of. Maybe, some small bits of memory should
            be allocated beforehand so that we don't have to do this; something akin
            to an `operational memory`.
      EDIT: One idea could be to have buckets of memory that would be required for
            some operations, maybe buckets of 32 bytes, 64 bytes, ... and since the
            usage is temporary, we don't have to worry about memory fragmentation.
            If the program fills the memory to a certain point, we can allocate more.

    - Assign operands to each tensor that goes through a tensor operation, along with the op type.
      - Check that after operations, only the oldest non-leaf tensor gets overwritten.
    - Better (effecient) implementation of the tensor operations, maybe start with the the MatMul
    - Make the tensor struct more generic so we can have float and int at the same time.
*/

#include "grazie.h"

int main()
{
#if 0
    tensor32 *Ten1 = TensorFromArrayLiteral(Ten1, float32,
                                            ARR(2, 2, 2), // Shape
                                            ARR(-2.4f, 1.43f,
                                                 5.8f,  1.7f,
                                                12.14f, -3.4f,
                                                2.43f, 6.8f), true);
#else
    tensor32 *Ten1 = TensorFromArrayLiteral(Ten1, float32,
                                            ARR(2, 2, 2), // Shape
                                            ARR(-2.4f, 1.43f,
                                                 5.8f,  1.7f,
                                                12.14f, -3.4f,
                                                2.43f, 6.8f), true);
#endif

#if 0
    tensor32 *Ten2 = TensorFromArrayLiteral(Ten2, float32,
                                            ARR(2, 2),
                                            ARR(1, // 4, 3,
                                                -2, // 7, 5,
                                                3, // 4, 3,
                                                5,), true); // 7, 5));
#else
    tensor32 *Ten2 = TensorFromArrayLiteral(Ten2, float32, ARR(2), ARR(2, 1.2f), true);
#endif
    uint32 Shape1[] = {2, 2, 2};
    tensor32 *AddResult = T32Empty(Shape1, float32, true);
    // uint32 ReduceResultShape[] = {1};
    // tensor32 *ReduceResult = T32Empty(ReduceResultShape, float32, true);

    T32Div(Ten1, Ten2, AddResult);
    // PrintTensor32(AddResult);
    // T32ReduceSumAll(AddResult, ReduceResult);

    Backward(AddResult, true);

    SwapDataGrad(Ten1);
    SwapDataGrad(Ten2);
    SwapDataGrad(AddResult);
        printf("Ten1 Grad\n");
        PrintTensor32(Ten1);
        printf("Ten2 Grad\n");
        PrintTensor32(Ten2);
        printf("Result Grad\n");
        PrintTensor32(AddResult);
    SwapDataGrad(Ten1);
    SwapDataGrad(Ten2);

    // T32ReshapeInPlace(Result, ARR(3, 2, 2));
    // T32Transpose(Ten2, 0, -1, Result);
    // T32MatMul(Ten1, Ten2, Result);

    return 0;
}


#if 0
    // NOTE(Abid): Test for when one of the dimension is 1
    tensor32 Ten1 = TensorFromArrayLiteral(Ten1, float32,
                                           SHAPE(2, 3, 2),
                                           ARRAY(-2.4f, 1.43f,
                                                  5.8f, 12.14f,
                                                  5.5f, 3.2f,

                                                 -3.4f, 2.43f,
                                                  6.8f, 13.14f,
                                                  6.5f, 4.2f));

    tensor32 Ten2 = TensorFromArrayLiteral(Ten2, float32,
                                           SHAPE(2, 1, 2),
                                           ARRAY(-1, 4,

                                                  2, 7,));
#endif


#if 0
    uint32 Temp;
    Temp = Ten1.Header->Strides[0];
    Ten1.Header->Strides[0] = Ten1.Header->Strides[1];
    Ten1.Header->Strides[1] = Temp; 

    Temp = Ten1.Header->Sizes[0];
    Ten1.Header->Sizes[0] = Ten1.Header->Sizes[1];
    Ten1.Header->Sizes[1] = Temp; 

    uint32 Shape1[] = {2, 2, 3};
    tensor32 ResultAdd = T32Empty(Shape1, float32);
    T32Mul(Ten1, Ten2, ResultAdd);
    PrintTensor32(ResultAdd);
#endif

#if 0
    printf("Grad before setting it: \n");
    CallOnGradStorage(Ten2, PrintTensor32(Ten2));
    CallOnGradStorage(Ten2, T32SetElementsInPlace(Ten2, 15.4f));
    printf("Grad after setting it: \n");
    CallOnGradStorage(Ten2, PrintTensor32(Ten2));
    printf("Now the data: \n");
    PrintTensor32(Ten2);

    tensor32 ResultTranpose = _float32AllocTensor(Shape1, ArrayLength(Shape1), 0, 0, false);

    ResultAdd = T32Add(Ten1, Ten2, ResultAdd);
    ResultTranpose = T32TransposeAll(ResultAdd, ResultTranpose);
    tensor_i32 Ten2 = I32Tensor(Shape2, Value2);

    tensor_i32 Result = I32TenMatMul(Ten1, I32TenTransposeAll(Ten2));
    printf("First Operand\n");
    PrintI32Tensor(((tensor_i32 *)Result.Header->DerivedOp.Operands)[0]);

    printf("Second Operand\n");
    PrintI32Tensor(((tensor_i32 *)Result.Header->DerivedOp.Operands)[1]);

    printf("Transpose Operand of the Second Operand\n");
    PrintI32Tensor(((tensor_i32 *)(((tensor_i32 *)Result.Header->DerivedOp.Operands)[1]).Header->DerivedOp.Operands)[0]);

    PrintTensor32Grad(ResultTranpose);
#endif



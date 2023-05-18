/*
    TODO(Abid):

    - Create a fixed memory scheme that would be done at the start up
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
      - Check that after operations, the only the oldest non-leaf tensor gets overwritten.
    - Better (effecient) implementation of the tensor operations, maybe start with the the MatMul
    - Make the tensor struct more generic so we can have float and int at the same time.
*/

#include "grazie.h"

int main()
{
    TensorFromArrayLiteral(Ten1, float32,
                           SHAPE(2, 3),
                           ARRAY(-2.4f, 1.43f, 5.8f,
                                  12.14f, 5.5f, 3.2f));

    TensorFromArrayLiteral(Ten2, float32,
                           SHAPE(2, 3),
                           ARRAY(-1, 4, 8,
                                  2, 7, 9));

    PrintTensor32Data(Ten2);

    uint32 Shape1[] = {2, 3};
    tensor32 ResultAdd = _int32AllocTensor(Shape1, ArrayLength(Shape1), 0, 0, false);
    tensor32 ResultTranpose = _float32AllocTensor(Shape1, ArrayLength(Shape1), 0, 0, false);

    ResultAdd = T32Add(Ten1, Ten2, ResultAdd);
    ResultTranpose = T32TransposeAll(ResultAdd, ResultTranpose);

#if 0
    tensor_i32 Ten2 = I32Tensor(Shape2, Value2);

    tensor_i32 Result = I32TenMatMul(Ten1, I32TenTransposeAll(Ten2));
    printf("First Operand\n");
    PrintI32Tensor(((tensor_i32 *)Result.Header->DerivedOp.Operands)[0]);

    printf("Second Operand\n");
    PrintI32Tensor(((tensor_i32 *)Result.Header->DerivedOp.Operands)[1]);

    printf("Transpose Operand of the Second Operand\n");
    PrintI32Tensor(((tensor_i32 *)(((tensor_i32 *)Result.Header->DerivedOp.Operands)[1]).Header->DerivedOp.Operands)[0]);
#endif

    PrintTensor32Data(ResultTranpose);

    return 0;
}

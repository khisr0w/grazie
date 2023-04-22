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

    - TODO(Abid): Math Ops
                  - View
                  - Convolution

    DODO:
    - Storing operations history for autograd
    - MatMul implemented
    - Transpose implemented
    - Change the structure of the tensor so that they are pointers, avoid copying them around
    - How to resolve the issue of freeing the memory of intermediate results, in case we call 'No grad' on them.

*/

#include "grazie.h"

int main()
{
    uint32 Shape1[] = {2, 3};
    int32 Value1[] = {-2, 1, 5,
                      12, 55, 3};
    tensor_i32 Ten1 = I32Tensor(Shape1, Value1);

    tensor_i32 Result = I32TenNeg(Ten1);
#if 0
    uint32 Shape2[] = {2, 3};
    int32 Value2[] = {-1, 4, 8,
                      2, 7, 9};
    tensor_i32 Ten2 = I32Tensor(Shape2, Value2);

    tensor_i32 Result = I32TenMatMul(Ten1, I32TenTransposeAll(Ten2));
    printf("First Operand\n");
    PrintI32Tensor(((tensor_i32 *)Result.Header->DerivedOp.Operands)[0]);

    printf("Second Operand\n");
    PrintI32Tensor(((tensor_i32 *)Result.Header->DerivedOp.Operands)[1]);

    printf("Transpose Operand of the Second Operand\n");
    PrintI32Tensor(((tensor_i32 *)(((tensor_i32 *)Result.Header->DerivedOp.Operands)[1]).Header->DerivedOp.Operands)[0]);
#endif

    PrintI32Tensor(Result);

    return 0;
}

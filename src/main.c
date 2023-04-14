/*
    TODO(Abid):

    - Create a fixed memory scheme that would be done at the start up
    - Assign operands to each tensor who goes through a tensor operation, along with the op type.
      - Check that after operations, the only the oldest non-leaf tensor gets overwritten.
    - Better (effecient) implementation of the tensor operations, maybe start with the the MatMul
    - Make the tensor struct more generic so we can have float and int at the same time.

    - Math Ops:
      - Implement matrix multiplication
      - Implement transpose
      - ...

    DODO:
    - Change the structure of the tensor so that they are pointers, avoid copying them around
    - How to resolve the issue of freeing the memory of intermediate results, in case we call 'No grad' on them.

*/

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// NOTE(Abid): Unity includes
#include "utils.h"
#include "memory.c"
#include "tensor.c"

int main()
{
    InitGrazie(Gigabyte(2));
    int32 Shape1[] = {3, 4};
    int32 Data1[] = {1, 2, 3, 4,
                     5, 6, 7, 8,
                     9, 10, 11, 12};

    int32 Shape2[] = {3, 4};
    int32 Data2[] = {3, 4, 8, -1,
                     15, 26, 7, 2,
                     9, 15, 16, 2};

    // float32 Data3[] = {3, 4, 8, -1,
    //                    15, 26, 7, 2,
    //                    9, 15, 16, 2};

    int32 Sh1[] = {2, 2};
    int32 Val1[] = {1, 2,
                    3, 4};

    int32 Sh2[] = {2, 2};
    int32 Val2[] = {1, 2,
                    3, 4};

    int32 Sh3[] = {2, 2};
    int32 Val3[] = {2, 2,
                    2, 2};

    tensor_i32 Ten1 = I32Tensor(Shape1, Data1);
    tensor_i32 Ten2 = I32Tensor(Shape2, Data2);
    I32TenAdd(Ten1, Ten2);

    tensor_i32 ValTen1 = I32Tensor(Sh1, Val1);
    tensor_i32 ValTen2 = I32Tensor(Sh2, Val2);
    tensor_i32 ValTen3 = I32Tensor(Sh3, Val3);

    GRAD_PRESERVE(false);

    tensor_i32 TenMul = I32TenMul(
                                  I32TenAdd(ValTen1, ValTen2),
                                  ValTen3
                                  );

    GRAD_PRESERVE_TOGGLE();

    PrintI32Tensor(TenMul);

    // PrintTensor(Ten1);
    // printf("\n\t*\n\n");
    // PrintTensor(Ten2);
    // printf("\n\t=\n\n");
    // PrintTensor(TenMul);
    // printf("\n\nFloat Tensor:\n"); PrintTensor(&Ten3);

    boolean B1 = IS_GRAD_PRESERVE();
    boolean B2 = GRAD_PRESERVE(false);

    printf("Grad is: %d\n", IS_GRAD_PRESERVE());
    GRAD_PRESERVE(true);
    printf("Grad is: %d\n", IS_GRAD_PRESERVE());

    printf("StateMemUsed Before: %zu\n", GrazieState()->MemByteUsed);
    GrazieState()->MemByteUsed = 12;
    printf("StateMemUsed Before: %zu\n", GrazieState()->MemByteUsed);
    printf("StateMemSize is: %zu\n", GrazieState()->MemMaxByteSize);


    return 0;
}

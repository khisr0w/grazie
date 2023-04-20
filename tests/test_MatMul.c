/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /tests                                                        |
    |    Creation date:  4/20/2023 6:25:39 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "../src/grazie.h"

int main()
{
    uint32 LShape1[] = {3, 4, 5};
    int32 Large1[] = {38, 88, 30, 50, 64,
                      33, 64, 21, 93, 62,
                      15, 99, 37, 39, 87,
                      88, 32, 98, 98, 56,

                      72, 93, 17, 46, 21,
                      77, 74, 42, 37, 40,
                      66, 80, 85, 81,  5,
                      31, 24, 92, 52,  2,

                      68, 74, 48, 37, 44,
                      83, 26, 66, 18, 46,
                      47, 76, 10, 97, 24,
                      67, 46, 89, 52, 75};

    uint32 LShape2[] = {3, 5, 7};
    int32 Large2[] = {42, 41, 99, 32,  4, 77,  2,
                      35, 56, 40, 51, 93, 51, 85,
                      20, 39, 67, 51,  5, 13, 80,
                      50,  2,  4, 23, 56, 14, 70,
                      30, 87, 65, 99, 51, 39, 12,

                       4, 97, 59, 46, 67, 78, 53,
                      93, 86, 18, 37, 21, 34, 36,
                      43, 99, 62, 16, 66, 24, 10,
                      71, 45, 88, 14, 55, 49, 62,
                      21, 76, 64, 81, 35, 86, 15,

                      87, 79, 34, 69, 55, 59, 70,
                      24, 55, 18, 71,  4, 44, 32,
                      89, 79,  9, 11, 78, 14, 24,
                      59, 88, 53, 45, 31, 41, 29,
                      14,  1, 36, 91, 14, 83, 65};

    tensor_i32 LargeValTen1 = I32Tensor(LShape1, Large1);
    tensor_i32 LargeValTen2 = I32Tensor(LShape2, Large2);

    tensor_i32 TenMul = I32TenMatMul(LargeValTen1, LargeValTen2);

    PrintI32Tensor(TenMul);

    return 0;
}

#if 0
    InitGrazie(Gigabyte(2));

    uint32 Shape1[] = {3, 4};
    int32 Data1[] = {1, 2, 3, 4,
                     5, 6, 7, 8,
                     9, 10, 11, 12};

    uint32 Shape2[] = {3, 4};
    int32 Data2[] = {3, 4, 8, -1,
                     15, 26, 7, 2,
                     9, 15, 16, 2};

    tensor_i32 Ten1 = I32Tensor(Shape1, Data1);
    tensor_i32 Ten2 = I32Tensor(Shape2, Data2);
    
    PrintI32Tensor(I32TenAdd(Ten1, Ten2));

    // float32 Data3[] = {3, 4, 8, -1,
    //                    15, 26, 7, 2,
    //                    9, 15, 16, 2};

    uint32 Sh1[] = {2, 2, 3};
    int32 Val1[] = {12, 11, 10,
                    9,  8,  7,

                    6,  5,  4,
                    3,  2,  1};

    uint32 Sh2[] = {2, 4, 12};
    int32 Val2[] = {1,
                    2,
                    3,

                    4,
                    5,
                    6};


    // int32 Sh3[] = {2, 2};
    // int32 Val3[] = {2, 2,
    //                 2, 2};

    // tensor_i32 ValTen1 = I32Tensor(Sh1, Val1);
    // tensor_i32 ValTen2 = I32Tensor(Sh2, Val2);
    // tensor_i32 ValTen3 = I32Tensor(Sh3, Val3);
    // PrintI32Tensor(ValTen1);

    GRAD_PRESERVE(false);

    tensor_i32 TenMul = I32TenMul(
                                  I32TenAdd(ValTen1, ValTen2),
                                  ValTen3
                                  );
    GRAD_PRESERVE_TOGGLE();

    // PrintTensor(Ten1);
    // printf("\n\t*\n\n");
    // PrintTensor(Ten2);
    // printf("\n\t=\n\n");
    // PrintTensor(TenMul);
    // printf("\n\nFloat Tensor:\n"); PrintTensor(&Ten3);

    // boolean B1 = IS_GRAD_PRESERVE();
    // boolean B2 = GRAD_PRESERVE(false);

    printf("Grad is: %d\n", IS_GRAD_PRESERVE());
    GRAD_PRESERVE(true);
    printf("Grad is: %d\n", IS_GRAD_PRESERVE());

    printf("StateMemUsed Before: %zu\n", GrazieState()->MemByteUsed);
    GrazieState()->MemByteUsed = 12;
    printf("StateMemUsed Before: %zu\n", GrazieState()->MemByteUsed);
    printf("StateMemSize is: %zu\n", GrazieState()->MemMaxByteSize);
#endif

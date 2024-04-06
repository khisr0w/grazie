/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 10:19:09 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#include "grazie.h"

i32 main() {
    /* NOTE(Abid): To implement XOR NN, we need:
     *             1. Sigmoid (DONE)
     *             2. (Binary) CrossEntropy Loss
     *             3. SGD Optimizer 
     *             4. Non-Linearities
     *             5. Linear Module 
     */

    t32 *A = TensorFromArrayLiteral(A, f32, ARR(3, 4),
                                    ARR(5.0f,    4.0f,  2.0f,
                                        6.0f,   -3.0f, -5.0f,
                                        6.45f, 5.0f, 1.3f,
                                        -1.4f, 44.14f, 11.9f), true);

    t32 *B = TensorFromArrayLiteral(B, f32,
                                    ARR(3, 4),
                                    ARR(19.234f, 18.007f, 90.562f, 18.204f,
                                        30.362f, 31.658f, 60.179f, 77.811f,
                                        89.703f, 60.655f, 15.55f, 97.477f), true);
    u32 LShape[] = {1};
    t32 *LossRes = T32Empty(LShape, f32, true);
    tensor_list OptimList = T32AllocateTensorList(1024);
    __T32AddToTensorList(A, &OptimList);
    //__T32AddToTensorList(B, &OptimList);

    OptimList.Size;

    T32Sigmoid(A, B);
    T32Print(B);
    T32ReduceSumAll(B, LossRes);
    T32Backprop(LossRes);

    SwapDataGrad(A);
    SwapDataGrad(B);
        T32Print(B);
        T32Print(A);
    SwapDataGrad(A);
    SwapDataGrad(B);

    /* NOTE(Abid): SGD seems to be working for this example but TODO: we need to test it on more 
     *             examples. */
    T32SGDOptim(OptimList, 0.1f);
    printf("\nAfter SDG: \n");
    T32Print(A);

    return(0);
}

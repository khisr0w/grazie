/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 10:19:09 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#define T32Data(Shape, Data, TYPE, ShouldGrad, Arena) _##TYPE##AllocTensor(Shape, ArrayLength(Shape), Data, ArrayLength(Data), ShouldGrad, Arena)
#define T32Empty(Shape, TYPE, ShouldGrad, Arena) _##TYPE##AllocTensor(Shape, ArrayLength(Shape), 0, 0, ShouldGrad, Arena)
#include "grazie.h"

i32 main() {
    /* NOTE(Abid): To implement XOR NN, we need:
     *             1. Sigmoid                    (DONE)
     *             2. (Binary) CrossEntropy Loss (DONE)
     *             3. SGD Optimizer              (DONE)
     *             4. Non-Linearities(ReLU)      (DONE)
     *             5. Linear Module              (DONE)
     */

    mem_arena MainArena = AllocateArena(Megabyte(100));

    /* NOTE(Abid): Model definition */
    module *Lin1 = T32Linear(20, 50, &MainArena);

    /* NOTE(Abid): Training loop */
    temp_memory TempSession = BeginTempMemory(&MainArena); {
        u32 Shape[] = {100, 20};
        t32 *Input = T32Empty(Shape, f32, true, &MainArena);
        t32 *Output = RunModule(Lin1, Input, &MainArena);
    } EndTempMemory(TempSession);

    return(0);
}

#if 0
    t32 *A = TensorFromArrayLiteral(A, f32, ARR(3, 4),
                                    ARR(5.0f,  4.0f,  2.0f,
                                        6.0f, -3.0f, -5.0f,
                                        6.45f, 5.0f,  1.3f,
                                       -1.4f, 44.14f, 11.9f), true, &MainArena);

    t32 *B = TensorFromArrayLiteral(B, f32, ARR(3, 4),
                                    ARR(19.234f, 18.007f, 90.562f, 18.204f,
                                        30.362f, 31.658f, 60.179f, 77.811f,
                                        89.703f, 60.655f, 15.55f, 97.477f), true, &MainArena);
    u32 LShape[] = {1};
    t32 *LossRes = T32Empty(LShape, f32, true, &MainArena);

    tensor_list OptimList = T32AllocateTensorList(1024, &MainArena);
    __T32AddToTensorList(&OptimList, A);
    //__T32AddToTensorList(&OptimList, B);

    T32ReLU(A, B);
    T32Print(B);
    T32ReduceSumAll(B, LossRes);
    T32Backprop(LossRes);

    SwapDataGrad(A); SwapDataGrad(B); {
        T32Print(B);
        T32Print(A);
    } SwapDataGrad(A); SwapDataGrad(B);

    /* NOTE(Abid): SGD seems to be working for this example but TODO: we need to test it on more 
     *             examples. */
    T32SGDOptim(OptimList, 0.1f);
    printf("\nAfter SDG: \n");
    T32Print(A);
#endif
#if 0
    t32 *A = TensorFromArrayLiteral(A, f32, ARR(3, 4),
                                    ARR(5.0f,  4.0f,  2.0f,
                                        6.0f, -3.0f, -5.0f,
                                        6.45f, 5.0f,  1.3f,
                                       -1.4f, 44.14f, 11.9f), true, &MainArena);

    t32 *B = TensorFromArrayLiteral(B, f32, ARR(3, 4),
                                    ARR(19.234f, 18.007f, 90.562f, 18.204f,
                                        30.362f, 31.658f, 60.179f, 77.811f,
                                        89.703f, 60.655f, 15.55f, 97.477f), true, &MainArena);
    u32 LShape[] = {1};
    t32 *LossRes = T32Empty(LShape, f32, true, &MainArena);

    tensor_list OptimList = T32AllocateTensorList(1024, &MainArena);
    __T32AddToTensorList(&OptimList, A);
    //__T32AddToTensorList(&OptimList, B);

    T32ReLU(A, B);
    T32Print(B);
    T32ReduceSumAll(B, LossRes);
    T32Backprop(LossRes);

    SwapDataGrad(A); SwapDataGrad(B); {
        T32Print(B);
        T32Print(A);
    } SwapDataGrad(A); SwapDataGrad(B);

    /* NOTE(Abid): SGD seems to be working for this example but TODO: we need to test it on more 
     *             examples. */
    T32SGDOptim(OptimList, 0.1f);
    printf("\nAfter SDG: \n");
    T32Print(A);
#endif

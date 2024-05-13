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
     *             1. Sigmoid                    (DONE)
     *             2. (Binary) CrossEntropy Loss (DONE)
     *             3. SGD Optimizer              (DONE)
     *             4. Non-Linearities(ReLU)      (DONE)
     *             5. Linear Module              (DONE)
     */

    mem_arena MainArena = AllocateArena(Megabyte(100));

    /* NOTE(Abid): Input */
    f32 TrainX[][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    };

    f32 TrainY[][1] = {
        {0},
        {1},
        {1},
        {0},
    };

    u32 InputShape[] = {5, 7};
    f32 InputData[] = {
        0.3978f, -1.1573f,  0.5564f,  1.4209f,  0.6619f,  1.2710f,  1.0088f,
        0.6431f, -0.7379f, -0.2102f, -0.1002f, -2.2516f, -1.1144f, -1.2046f,
        0.5182f, -0.7396f,  0.1325f,  2.8119f, -0.2471f, -0.2388f,  1.0221f,
        0.2247f, -0.0725f,  0.6489f,  0.6703f, -2.3605f, -0.9891f,  0.3283f,
       -0.2900f,  0.9854f,  0.7036f,  0.4051f, -0.2086f, -0.7467f, -0.5946f
    };
    t32 *Input = T32Data(InputShape, InputData, f32, true, &MainArena);

    /* NOTE(Abid): Model definition */
    module *Lin1 = T32Linear(7, 6, &MainArena);
    tensor_list OptimList = T32AllocateTensorList(1024, &MainArena);
    __T32AddToTensorList(&OptimList, Lin1->TensorList.Array[0]);
    __T32AddToTensorList(&OptimList, Lin1->TensorList.Array[1]);

    /* NOTE(Abid): Training loop */
    for(u32 Idx = 0; Idx < 10; ++Idx) {
        temp_memory TempSession = BeginTempMemory(&MainArena);

        T32ZeroGrad(OptimList);
        t32 *Output = RunModule(Lin1, Input, &MainArena);
        u32 LossShape[] = {1};
        t32 *Loss = T32Empty(LossShape, f32, true, &MainArena);
        T32ReduceSumAll(Output, Loss);

        T32Backprop(Loss);
        T32SGDOptim(OptimList, 0.1f);

        EndTempMemory(TempSession);
    }
    T32Print(Lin1->TensorList.Array[0]);
    T32Print(Lin1->TensorList.Array[1]);

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

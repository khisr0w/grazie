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
     *             0. Non-Linearities(ReLU)      (DONE)
     */

    mem_arena MainArena = gzMemArenaAllocate(gzMegabyte(100));

    /* NOTE(Abid): Input */
    // f32 TrainX[][2] = {
    //     {0, 0},
    //     {0, 1},
    //     {1, 0},
    //     {1, 1},
    // };

    // f32 TrainY[][1] = {
    //     {0},
    //     {1},
    //     {1},
    //     {0},
    // };

    u32 InputShape[] = {5, 7};
    f32 InputData[] = {
        0.3978f, -1.1573f,  0.5564f,  1.4209f,  0.6619f,  1.2710f,  1.0088f,
        0.6431f, -0.7379f, -0.2102f, -0.1002f, -2.2516f, -1.1144f, -1.2046f,
        0.5182f, -0.7396f,  0.1325f,  2.8119f, -0.2471f, -0.2388f,  1.0221f,
        0.2247f, -0.0725f,  0.6489f,  0.6703f, -2.3605f, -0.9891f,  0.3283f,
       -0.2900f,  0.9854f,  0.7036f,  0.4051f, -0.2086f, -0.7467f, -0.5946f
    };
    t32 *Input = gzTensorFromArray(InputShape, InputData, f32, true, &MainArena);
    // Input = gzTensorNormal(InputShape, 0, 1, true, &MainArena);
    // gzPrint(Input);

    /* NOTE(Abid): Model definition */
    module *Lin1 = gzLinear(7, 6, &MainArena);
    tensor_list OptimList = gzTensorListFromModule(Lin1, &MainArena);

    /* NOTE(Abid): Training loop */
    for(u32 Idx = 0; Idx < 10; ++Idx) {
        temp_memory TempSession = gzMemTempBegin(&MainArena);

        gzGradZero(OptimList);
        t32 *Out = gzRunModule(Lin1, Input, &MainArena);
        Out = gzReLU(Out, &MainArena);
        t32 *Loss = gzReduceSumAll(Out, &MainArena);

        gzBackprop(Loss);
        gzOptimSGD(OptimList, 0.1f);

        gzMemTempEnd(TempSession);
    }
    // gzPrint(Lin1->TensorList.Array[0]);
    // gzPrint(Lin1->TensorList.Array[1]);

    f32 a_data[] = {
        1, 2, 3,
        4, 5, 6,
    };
    u32 a_shape[] = {2, 3};
    t32 *a = gzTensorFromArray(a_shape, a_data, f32, true, &MainArena);

    f32 b_data[] = {
        7, 8, 9, 10,
        11, 12, 13, 14,
        14, 15, 16, 17,
    };
    u32 b_shape[] = {3, 4};
    t32 *b = gzTensorFromArray(b_shape, b_data, f32, true, &MainArena);

    u32 c_shape[] = {2, 4};
    t32 *c = gzTensorEmpty(c_shape, f32, true, &MainArena);

    gzMatMul(a, b, c);
    gzBackprop(c);

    gzPrint(c);
    gzSwapDataGrad(a);
    gzSwapDataGrad(b);
    gzPrint(a);
    gzPrint(b);
    gzSwapDataGrad(a);
    gzSwapDataGrad(b);


    return(0);
}

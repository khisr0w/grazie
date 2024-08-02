/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/25/2024 6:18:00 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#include "module.h"

internal module *
gzLinear(u32 InDim, u32 OutDim, mem_arena *Arena) {
    module *Module = gzMemPushStruct(Arena, module);
    Module->Type = module_Linear;

    u32 WShape[] = {1, OutDim, InDim};
    u32 BShape[] = {1, OutDim};

#if 0
    f32 WData[] = {
         0.1156f, -0.1106f, -0.2412f, -0.3419f,  0.2909f,  0.2414f,  0.0282f,
        -0.2990f, -0.3239f, -0.0045f,  0.3586f, -0.1603f, -0.3139f,  0.0940f,
        -0.0951f, -0.0665f,  0.3455f,  0.1468f, -0.0944f,  0.1256f,  0.2166f,
         0.2839f, -0.1071f,  0.3134f, -0.3428f,  0.1482f,  0.1772f, -0.3114f,
        -0.1763f, -0.1420f,  0.1839f, -0.1723f,  0.0435f,  0.0620f, -0.1480f,
         0.1925f, -0.0642f, -0.2648f,  0.3748f,  0.2646f,  0.2446f,  0.3162f
    };
    f32 BData[] = { -0.1549f, -0.0739f, -0.2686f, -0.2907f,  0.1559f, -0.2320f };
    t32 *W = gzTensorFromArray(WShape, WData, f32, true, Arena)
    t32 *B = gzTensorFromArray(BShape, BData, f32, true, Arena)
#endif

    t32 *W = gzTensorNormal(WShape, 0.f, 1.f, true, Arena);
    t32 *B = gzTensorNormal(BShape, 0.f, 1.f, true, Arena);

    Module->TensorList = gzTensorListAllocate(2, Arena);
    gzTensorListAdd(&Module->TensorList, W);
    gzTensorListAdd(&Module->TensorList, B);

    return Module;
}

internal t32 *
gzRunModule(module *Module, t32 *Input, mem_arena *Arena) {
    t32 *Result = NULL;

    switch(Module->Type) {
        case module_Linear: {
            /* TODO(Abid): Implement backward for addmm to make it more efficient. */
            t32 *W = Module->TensorList.Array[0];
            t32 *B = Module->TensorList.Array[1];
            Assert(Input->Header->Dim == 2, "expected input dim to be 2");
            Assert(Input->Header->Sizes[1] == W->Header->Sizes[2], "input-linear shape mismatch");

            u32 BatchSize = Input->Header->Sizes[0];
            u32 InDim = Input->Header->Sizes[1];
            u32 OutDim = W->Header->Sizes[1];

            u32 WXShape[] = {BatchSize, OutDim, 1};
            t32 *WX = gzTensorEmpty(WXShape, f32, true, Arena);
            u32 InputViewShape[] = {BatchSize, InDim, 1};
            t32 *InputView = _gzNewView(Input, InputViewShape, gzArrayLength(InputViewShape), Arena);
            gzMatMul(W, InputView, WX);

            u32 ResultShape[] = {BatchSize, OutDim};
            Result = gzTensorEmpty(ResultShape, f32, true, Arena);
            gzAdd(gzTrimUnitSize(WX, Arena), B, Result);
        } break;
        default: Assert(0, "invalid code path"); break;
    }

    return Result;
}

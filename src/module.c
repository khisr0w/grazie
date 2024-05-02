/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/25/2024 6:18:00 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

typedef enum {
    module_None,
    module_Linear,
} module_type;

typedef struct {
    tensor_list TensorList;
    module_type Type;
} module;

internal module *
T32Linear(u32 InDim, u32 OutDim, mem_arena *Arena) {
    module *Module = PushStruct(Arena, module);
    Module->Type = module_Linear;

    u32 WShape[] = {OutDim, InDim};
    u32 BShape[] = {OutDim};
    Module->TensorList = T32AllocateTensorList(2, Arena);

    Module->TensorList.Array[0] = T32Empty(WShape, f32, true, Arena);
    Module->TensorList.Array[1] = T32Empty(BShape, f32, true, Arena);

    return Module;
}

internal t32 *
RunModule(module *Module, t32 *Input, mem_arena *Arena) {
    t32 *Result = NULL;

    switch(Module->Type) {
        case module_Linear: {
            t32 *W = Module->TensorList.Array[0];
            t32 *B = Module->TensorList.Array[1];
            Assert(Input->Header->Dim == 2, "expected input dim to be 2");
            Assert(Input->Header->Sizes[1] == W->Header->Sizes[1], "input-linear shape mismatch");

            u32 BatchSize = Input->Header->Sizes[0];
            u32 Shape[] = {BatchSize, W->Header->Sizes[0]};
            t32 *WX = T32Empty(Shape, f32, true, Arena);
            T32MatMul(W, Input, WX);
            Result = T32Empty(Shape, f32, true, Arena);
        } break;
        default: Assert(0, "invalid code path"); break;
    }

    return Result;
}

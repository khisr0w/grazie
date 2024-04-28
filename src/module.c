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

internal module
AllocateLinear(u32 BatchSize, u32 InDim, u32 OutDim) {
    module Module = {0};
    Module.Type = module_Linear;

    u32 WShape[3] = {BatchSize, OutDim, InDim};
    u32 BShape[2] = {BatchSize, OutDim};
    Module.TensorList = T32AllocateTensorList(2);

    Module.TensorList.Array[0] = T32Empty(WShape, f32, true);
    Module.TensorList.Array[1] = T32Empty(BShape, f32, true);

    return Module;
}

#if 0
internal void
RunModule(module Module, t32 *Dest) {
}
#endif

/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/25/2024 6:18:00 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#include "module.h"

internal module *
gz_module_linear(u32 InDim, u32 OutDim, mem_arena *Arena) {
    module *Module = gz_mem_push_struct(module, Arena);
    Module->type = module_linear;

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

    Module->weights = gz_tensor_list_allocate(2, Arena);
    gz_tensor_list_add(W, &Module->weights);
    gz_tensor_list_add(B, &Module->weights);

    return Module;
}

internal module *
gz_module_sigmoid(mem_arena *arena) {
    module *mod = gz_mem_push_struct(module, arena);
    mod->type = module_sigmoid;

    return mod;
}

internal module *
gz_module_relu(mem_arena *arena) {
    module *mod = gz_mem_push_struct(module, arena);
    mod->type = module_relu;

    return mod;
}

internal t32 *
gz_module_run(module *module, t32 *input, mem_arena *arena) {
    t32 *result = NULL;

    switch(module->type) {
        case module_linear: {
            /* TODO(Abid): Implement backward for addmm to make it more efficient. */
            t32 *w = module->weights.array[0];
            t32 *b = module->weights.array[1];
            assert(input->Header->Dim == 2, "expected input dim to be 2");
            assert(input->Header->Sizes[1] == w->Header->Sizes[2], "input-linear shape mismatch");

            u32 batch_size = input->Header->Sizes[0];
            u32 in_dim = input->Header->Sizes[1];
            u32 out_dim = w->Header->Sizes[1];

            u32 wx_shape[] = {batch_size, out_dim, 1};
            t32 *wx = gz_tensor_empty(wx_shape, f32, true, arena);
            u32 input_view_shape[] = {batch_size, in_dim, 1};
            t32 *input_view = _gzNewView(input, input_view_shape, gz_array_length(input_view_shape), arena);
            gzMatMul(w, input_view, wx);

            u32 result_shape[] = {batch_size, out_dim};
            result = gz_tensor_empty(result_shape, f32, true, arena);
            gzAdd(gz_trim_trailing_unit_size(wx, arena), b, result);
        } break;
        case module_sigmoid: { result = gz_sigmoid(input, arena); } break;
        case module_relu: { result = gz_relu(input, arena); } break;
        default: assert(0, "invalid code path"); break;
    }

    return result;
}

inline internal t32 *
gz_module_run_all(module **modules, u64 module_length, t32 *input, mem_arena *arena) {
    t32 *result = input;
    for(u64 idx = 0; idx < module_length; ++idx)
        result = gz_module_run(modules[idx], result, arena);

    return result;
}

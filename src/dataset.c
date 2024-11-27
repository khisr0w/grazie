/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 25 Nov 2024 15:09:36 CET                                   |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

/* NOTE(abid): The base tensor serves as the output of a dataset.
 * and `stride` is for getting new datapoints from `stream` into `base_tensor`. */
typedef struct {
    t32 *base_tensor;
    u64 stride; /* TODO(abid): This is possibly redundant, since we can calculate from tensor. */

    f32 *stream;
    u64 length;
} dataset;

inline internal dataset
dataset_build(f32 *stream, u64 stream_length, u64 batch_size, u64 *shape, u64 shape_length, mem_arena *arena) {
    dataset result = {0};

    /* TODO(abid): Have more types than just `f32`. */
    /* TODO(abid): Implement shuffling for random batches. */
    result.base_tensor = _gz_tensor_alloc_huskf32_batched(shape, shape_length, batch_size, arena);
    result.stream = stream;
    /* NOTE(abid): Ignoring data not fit into batch. */

    result.stride = 1;
    for(u64 idx = 0; idx < result.base_tensor->Header->Dim; ++idx) {
        result.stride *= result.base_tensor->Header->Sizes[idx];
    }
    result.length = stream_length/result.stride;

    return result;
}

inline internal t32 *
dataset_index(dataset *dataset, u64 idx) {
    assert(idx < dataset->length, "dataset index out of bounds");
    dataset->base_tensor->Data.Ptr = dataset->stream + dataset->stride*idx;

    return dataset->base_tensor;
}

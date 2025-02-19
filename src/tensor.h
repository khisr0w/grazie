/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 3:16:32 PM                                           |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(TENSOR_H)

typedef enum {
    op_none = 0,

    op_unary_begin,

    op_unary_negate,
    op_unary_broadcast,
    op_unary_tranpose,
    op_unary_tranpose_all,
    op_unary_reduce_sum_all,
    op_unary_sigmoid,
    op_unary_relu,
    op_unary_view,

    op_unary_end, /* NOTE(Abid): Marks the num after the end of unary ops, WARNING: should not be moved! */

    op_binary_begin,

    op_binary_add,
    op_binary_sub,
    op_binary_mul,
    op_binary_div,
    op_binary_matmul,

    op_binary_end, /* NOTE(Abid): Marks the num after the end of binary ops, WARNING: should not be moved! */

    op_loss_begin,

    op_binary_loss_cross_entropy,

    op_loss_end, /* NOTE(Abid): Marks the num after the end of binary ops, WARNING: should not be moved! */

} tensor_op;

typedef enum {
    dtype_f32 = 0,
    dtype_i32 = 1,
} tensor_dtype;

typedef struct t32 t32;
typedef struct {
    tensor_op TensorOp;
    t32 **Operands;

    /* NOTE(Abid): This is used for storing context data related to operations,
     *             One of the main uses is to store the dimensions that transposed. */
    void *op_context;
} op_info;

typedef struct {
    u32 *Sizes;
    u32 *Strides;
    u32 Dim;
    u32 Offset;

    /* NOTE(Abid): This is stricly for optimizing math ops, so we don't allocate. TODO: Must remove */
    u32 *AccessSizes; 
    size_t StorageNumElements;

    bool ShouldGrad;
    bool IsContiguous;

    op_info DerivedOp;
} tensor_header;

typedef struct {

    void *Ptr;
    tensor_dtype DType;
} storage;

struct t32 {
    tensor_header *Header;

    storage Data;
    storage Grad;
};

typedef struct {
    t32 **array;
    usize size;
    usize used;
} tensor_list;

typedef enum {
    reduce_none = 0,
    reduce_mean = 1,
    reduce_sum  = 2,
} reduce_method;

#define TENSOR_H
#endif

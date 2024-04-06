/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 3:16:32 PM                                           |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(TENSOR_H)

typedef enum {
    storage_Data,
    storage_Grad,

} storage_type;

typedef enum {
    op_None = 0,

    op_UnaryNegate,
    op_UnaryBroadcast,
    op_UnaryTranpose,
    op_UnaryTranposeAll,
    op_UnaryReduceSumAll,
    op_UnarySigmoid,

    op_UnaryEnd, /* NOTE(Abid): Marks the num after the end of unary ops, WARNING: should not be moved! */

    /* NOTE(Abid): Element wise ops */
    op_BinaryAdd,
    op_BinarySub,
    op_BinaryMul,
    op_BinaryDiv,

    op_BinaryMatmul,


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
    void *OpContext;
} op_info;

typedef struct {
    u32 *Sizes;
    u32 *Strides;
    u32 Dim;
    u32 Offset;

    /* NOTE(Abid): This is stricly for optimizing math ops, so we don't allocate. */
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

#define TENSOR_H
#endif

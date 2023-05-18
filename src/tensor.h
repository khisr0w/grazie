/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 3:16:32 PM                                           |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */
#if !defined(TENSOR_H)

typedef enum
{
    op_None,

    op_UnaryNegate,
    op_UnaryBroadcast,
    op_UnaryTranpose,
    op_UnaryTranposeAll,

    // NOTE(Abid): Element wise ops
    op_BinaryAdd,
    op_BinarySub,
    op_BinaryMul,
    op_BinaryDiv,

    op_BinaryMatmul,

} tensor_op;

typedef enum
{
    dtype_int32 = 1,
    dtype_float32 = 2,
} tensor_dtype;

typedef struct
{
    tensor_op TensorOp;
    void *Operands;

    // NOTE(Abid): This is used for storing context data related to operations,
    //             One of the main uses is to store the dimensions that transposed.
    void *OpContext;
} op_info;

typedef struct
{
    uint32 *Sizes;
    uint32 *Strides;
    uint32 Dim;
    uint32 Offset;

    // boolean IsPersist;
    boolean ShouldGrad;
    tensor_dtype DType;

    op_info DerivedOp;
} tensor_header;

typedef struct tensor32 tensor32;
struct tensor32
{
    tensor_header *Header;

    void *Data;
    void *Grad;
};

#define TENSOR_H
#endif

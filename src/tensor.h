/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 3:16:32 PM                                           |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright � All rights reserved |======+  */
#if !defined(TENSOR_H)

typedef enum
{
    storage_Data,
    storage_Grad,

} storage_type;

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
    brule_NotExist = 0,
    brule_SameSize,
    brule_LTERepeat,
    brule_GTERepeat,
} broadcast_rules;

typedef enum
{
    dtype_int32 = 1,
    dtype_float32 = 2,
} tensor_dtype;

typedef struct tensor32 tensor32;
typedef struct
{
    tensor_op TensorOp;
    tensor32 *Operands;

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

    uint32 *AccessSizes; // NOTE(Abid): This is stricly for optimizing math ops.
    size_t DataStorageSize;

    // boolean IsPersist;
    boolean ShouldGrad;
    boolean GradStorageInit;

    boolean IsContiguous;

    tensor_dtype DType;

    op_info DerivedOp;
} tensor_header;

struct tensor32
{
    tensor_header *Header;

    void *Data;
    void *Grad;
};

#define TENSOR_H
#endif

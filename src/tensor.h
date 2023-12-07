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

    op_ReduceSumAll,

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
    dtype_float32 = 0,
    dtype_int32 = 1,
} tensor_dtype;

typedef struct tensor32 tensor32;
typedef struct
{
    tensor_op TensorOp;
    tensor32 **Operands;

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

    // NOTE(Abid): This is stricly for optimizing math ops, so we don't allocate.
    uint32 *AccessSizes; 
    size_t StorageNumElements;

    // boolean IsPersist;
    boolean ShouldGrad;
    boolean IsContiguous;

    op_info DerivedOp;
} tensor_header;

typedef struct {

    void *Ptr;
    tensor_dtype DType;
} storage;

struct tensor32
{
    tensor_header *Header;

    storage Data;
    storage Grad;
};


#define TENSOR_H
#endif

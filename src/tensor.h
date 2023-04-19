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
    ops_None,

    ops_UnaryNegate,
    ops_UnaryBroadcast,

    // NOTE(Abid): Element wise ops
    ops_BinaryAdd,
    ops_BinaryMult,
    ops_BinarySub,

    ops_BinaryMatmul
} tensor_op;

typedef struct
{
    tensor_op TenOp;

    void **Parents;
} operands;

typedef struct
{
    uint32 *Sizes;
    uint32 *Strides;
    uint32 Dim;
    uint32 Offset;

    // boolean IsPersist;
    boolean ShouldGrad;

    operands Parents;
} tensor_header;

typedef struct
{
    tensor_header *Header;

    int32 *Storage;
} tensor_i32;

typedef struct
{
    tensor_header *Header;

    float32 *Storage;
} tensor_f32;

#define TENSOR_H
#endif

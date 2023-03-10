/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/9/2023 3:16:32 PM                                           |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */
#if !defined(TENSOR_H)

#define MAX_SHAPE_LENGTH 64

typedef struct
{
    int32 Sizes[MAX_SHAPE_LENGTH];
    int32 Strides[MAX_SHAPE_LENGTH];
    int32 Dim;

    int32 Offset;
} tensor_header;

typedef struct
{
    tensor_header Header;

    int32 *Storage;
} tensor_i32;

typedef struct
{
    tensor_header Header;

    float32 *Storage;
} tensor_f32;

#define TENSOR_H
#endif

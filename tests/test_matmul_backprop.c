/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /tests                                                        |
    |    Creation date:  12/20/2023 6:06:06 PM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "grazie.h"

int main() {
    tensor32 *Ten1 = TensorFromArrayLiteral(Ten1, float32,
                                            ARR(2, 3, 2), // Shape
                                            ARR(-2.4f, 1.43f,
                                                 5.8f,  1.7f,
                                                12.14f, 0.4f,
                                                -3.55f, 14.73f,
                                                22.34f,  2.3f,
                                                2.43f, 6.8f), true);
    tensor32 *Ten2 = TensorFromArrayLiteral(Ten2, float32,
                                            ARR(2, 4),
                                            ARR(2.2f, 4.76f, 3.01f, -2.93f,
                                                7.45f, -6.11f, 11.08f, 5.3f), true);
    tensor32 *Ten3 = TensorFromArrayLiteral(Ten3, float32,
                                            ARR(4, 5),
                                            ARR(19.234f, 18.007f, 63.91f, 90.562f, 18.204f,
                                                30.362f, 31.658f, 85.982f, 60.179f, 77.811f,
                                                44.786f, 30.6f, 66.859f, 47.448f, 52.97f,
                                                89.703f, 60.655f, 80.247f, 15.55f, 97.477f), true);
    uint32 RShape1[] = {2, 3, 4};
    tensor32 *Result1 = T32Empty(RShape1, float32, true);
    uint32 RShape2[] = {2, 3, 5};
    tensor32 *Result2 = T32Empty(RShape2, float32, true);
    uint32 ReShape[] = {1};
    tensor32 *ReduceResult = T32Empty(ReShape, float32, true);

    T32MatMul(Ten1, Ten2, Result1);
    T32MatMul(Result1, Ten3, Result2);

    T32ReduceSumAll(Result2, ReduceResult);

    __BackwardT32SetElements(Ten1, 0.f);
    __BackwardT32SetElements(Ten2, 0.f);
    __BackwardT32SetElements(Result1, 0.f);
    __BackwardT32SetElements(Result2, 0.f);
    __BackwardT32SetElements(ReduceResult, 0.f);

    Backward(ReduceResult);

    SwapDataGrad(Ten1);
    SwapDataGrad(Ten2);
    SwapDataGrad(Ten3);
    SwapDataGrad(ReduceResult);
        printf("< Ten1 Grad >\n");
        PrintTensor32(Ten1);
        printf("< Ten2 Grad >\n");
        PrintTensor32(Ten2);
        printf("< Ten3 Grad >\n");
        PrintTensor32(Ten3);
        printf("< ReduceResult Grad >\n");
        PrintTensor32(ReduceResult);
    SwapDataGrad(Ten1);
    SwapDataGrad(Ten2);
    SwapDataGrad(Ten3);
    SwapDataGrad(ReduceResult);

    return(0);
}

/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/21/2023 10:49:09 PM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

internal void
Backward(tensor32 A)
{
    tensor32 Ten = A;
    // tensor32 *FirstOperand = (tensor32 *)(Ten.Header->DerivedOp.Operands);
    // tensor32 *SecondOperand = FirstOperand+1;
    tensor_op Operation = Ten.Header->DerivedOp.TensorOp;

    while(Operation != op_None)
    {
        switch (Operation)
        {
            case op_UnaryNegate:
            {
            } break;
            case op_UnaryBroadcast:
            {
            } break;
            case op_UnaryTranpose:
            {
            } break;
            case op_UnaryTranposeAll:
            {
            } break;

            case op_BinaryAdd:
            {
            } break;
            case op_BinarySub:
            {
            } break;
            case op_BinaryMul:
            {
            } break;
            case op_BinaryDiv:
            {
            } break;
            case op_BinaryMatmul:
            {
            } break;
            default: Assert(0, "invalid code path");
        }
    }
}

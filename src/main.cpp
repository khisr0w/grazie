#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define Assert(Expr, ErrorStr) if(!(Expr)) {fprintf(stderr, "ASSERTION ERROR: " ErrorStr "\nExiting...\n"); *(int *)0 = 0;}
#define ArrayLength(Array) (sizeof(Array)/sizeof(Array[0]))

#define Free(ptr) free(ptr)
#define Malloc(ptr) malloc(ptr)
#define Calloc(ptr, size) calloc(ptr, size)

typedef uint32_t unint32;
typedef int32_t int32;
typedef uintptr_t uintptr;
typedef float float32;
typedef double float64;
typedef int8_t boolean;

// NOTE(Abid): Unity includes
#include "tensor.cpp"

/*
void inline
Backward(value_f32 *From)
{
    switch (From->Op)
    {
        case OP::ADD:
        {
            From->Operands[0]->Grad += 1.0f * From->Grad;
            Backward(From->Operands[0]);
            From->Operands[1]->Grad = 1.0f * From->Grad;
            Backward(From->Operands[1]);
        } break;

        case OP::MUL:
        {
            From->Operands[0]->Grad += From->Operands[1]->Data * From->Grad;
            Backward(From->Operands[0]);
            From->Operands[1]->Grad += From->Operands[0]->Data * From->Grad;
            Backward(From->Operands[1]);
        } break;
        case OP::NONE: return;
    }
}
*/

int main()
{
    int32 Shape[] = {3, 4};

    int32 Data1[] = {1, 2, 3, 4,
                     5, 6, 7, 8,
                     9, 10, 11, 12};
    int32 Data2[] = {3, 4, 8, -1,
                       15, 26, 7, 2,
                       9, 15, 16, 2};

    float32 Data3[] = {3, 4, 8, -1,
                       15, 26, 7, 2,
                       9, 15, 16, 2};

    tensor_i32 Ten1 = I32Tensor(Shape, Data1);
    tensor_i32 Ten2 = I32Tensor(Shape, Data2);
    tensor_f32 Ten3 = F32Tensor(Shape, Data3);

    tensor_i32 TenMul = Ten1 * Ten2;

    PrintTensor(&Ten1);
    printf("\n\t*\n\n");
    PrintTensor(&Ten2);
    printf("\n\t=\n\n");
    PrintTensor(&TenMul);
    printf("\n\nFloat Tensor:\n"); PrintTensor(&Ten3);

    return 0;
}

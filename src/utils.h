/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/16/2023 6:25:57 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(UTILS_H)

#define Assert(Expr, ErrorStr) if(!(Expr)) {fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\nExiting...\n", __FILE__, __LINE__); *(int *)0 = 0;}
#define InvalidCodePath() *(int *)0 = 0


#define ArrayLength(Array) (sizeof(Array)/sizeof(Array[0]))

// TODO(Abid): The custum allocators should be defined here
#define Free(ptr) free(ptr)
#define Malloc(ptr) malloc(ptr)
#define Calloc(ptr, size) calloc(ptr, size)

// NOTE(Abid): Byte Macros
#define Kilobyte(Value) ((Value)*1024LL)
#define Megabyte(Value) (Kilobyte(Value)*1024LL)
#define Gigabyte(Value) (Megabyte(Value)*1024LL)
#define Terabyte(Value) (Gigabyte(Value)*1024LL)

// NOTE(Abid): typedef and static define for ease of use
typedef uint32_t unint32;
typedef int32_t int32;
typedef uintptr_t uintptr;
typedef float float32;
typedef double float64;
typedef int8_t boolean;
#define internal static
#define local_persist static
#define global_var static
#define true 1
#define false 0

// NOTE(Abid): Defines if op history should be preserved for gradient calculation
global_var boolean GLOBAL_grazie_grad_history = true;
#define GRAD_PRESERVE(Value) GLOBAL_grazie_grad_history = Value
#define GRAD_PRESERVE_TOGGLE() GLOBAL_grazie_grad_history = !GLOBAL_grazie_grad_history
#define IS_GRAD_PRESERVE() GLOBAL_grazie_grad_history

#define UTILS_H
#endif
